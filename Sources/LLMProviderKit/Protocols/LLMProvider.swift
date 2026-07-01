import Foundation

/// A unified interface for any LLM provider.
///
/// Adding a new provider means creating a type that conforms to `LLMProvider`,
/// implementing the four requirements below, and optionally registering it with
/// `LLMService`.
public protocol LLMProvider: Sendable {
    /// Provider name. Used for logging and by `LLMService` lookups.
    static var name: String { get }

    /// Configuration for this provider instance.
    var configuration: LLMProviderConfiguration { get }

    /// Optional `URLSession` for advanced customization (cache, proxies, etc.).
    /// Defaults to `.shared` if not implemented.
    var urlSession: URLSession { get }

    /// URL request builder. Takes a generic request and returns a provider-specific
    /// `URLRequest` and an accompanying decoder closure.
    func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest

    /// Optional: resolve the model identifier for a request before the request is built.
    ///
    /// Some providers (e.g. Ollama) can discover the default model at runtime by
    /// querying the local server. The default implementation returns the request's
    /// `model` unchanged.
    func resolvedModel(for request: LLMRequest) async throws -> String

    /// Parse a single server-sent stream line (SSE) into zero or more chunks.
    ///
    /// - Parameters:
    ///   - line: One line of text received from the streaming endpoint.
    ///   - request: The original request, for correlation.
    /// - Returns: An array of chunks. Returning an empty array means “keep going”.
    func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk]

    /// Parse a non-streaming response body into a finished `LLMResponse`.
    func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse

    /// Optional: fetch the list of models available from this provider.
    ///
    /// Not every provider exposes a model list endpoint. The default
    /// implementation throws `LLMError.unsupportedOperation`.
    func availableModels() async throws -> [LLMModelInfo]
}

extension LLMProvider {
    public var urlSession: URLSession { .shared }

    public func resolvedModel(for request: LLMRequest) async throws -> String {
        if !request.model.isEmpty { return request.model }
        if let defaultModel = configuration.defaultModel, !defaultModel.isEmpty { return defaultModel }
        return request.model
    }

    public func availableModels() async throws -> [LLMModelInfo] {
        throw LLMError.unsupportedOperation("\(Self.name) does not support model listing.")
    }

    /// Non-streaming completion.
    public func complete(_ request: LLMRequest) async throws -> LLMResponse {
        let model = try await resolvedModel(for: request)
        var resolvedRequest = request
        resolvedRequest.model = model
        let urlRequest = try prepareRequest(resolvedRequest, stream: false)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await urlSession.data(for: urlRequest)
            Self.debugLogHTTPResponse(response, data: data)
        } catch {
            throw LLMError.networkError(error.localizedDescription)
        }

        try Self.verifyHTTPResponse(response, data: data)

        do {
            return try parseResponse(data, request: resolvedRequest)
        } catch let error as LLMError {
            throw error
        } catch {
            let bodyPreview = String(data: data, encoding: .utf8)
                .map { String($0.prefix(2_000)) }
                ?? "<non-UTF8 response: \(data.count) bytes>"
            throw LLMError.invalidResponse("\(error)\nRaw response preview: \(bodyPreview)")
        }
    }

    /// Streaming completion.
    ///
    /// Emits `.text` chunks as they arrive and a final `.finish` chunk. Network
    /// and parsing errors are emitted as `.error` chunks.
    public func stream(_ request: LLMRequest) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let model = try await self.resolvedModel(for: request)
                    var resolvedRequest = request
                    resolvedRequest.model = model
                    let urlRequest = try self.prepareRequest(resolvedRequest, stream: true)
                    let (bytes, response) = try await self.urlSession.bytes(for: urlRequest)
                    try Self.verifyHTTPResponse(response, data: nil)

                    var pendingLine = ""
                    for try await byte in bytes {
                        let char = Character(UnicodeScalar(byte))
                        if char.isNewline {
                            if !pendingLine.isEmpty {
                                let chunks = try self.parseStreamLine(pendingLine, request: resolvedRequest)
                                for chunk in chunks {
                                    continuation.yield(chunk)
                                    if case .finish = chunk { break }
                                }
                            }
                            pendingLine = ""
                        } else {
                            pendingLine.append(char)
                        }
                    }

                    if !pendingLine.isEmpty {
                        let chunks = try self.parseStreamLine(pendingLine, request: resolvedRequest)
                        for chunk in chunks { continuation.yield(chunk) }
                    }

                    continuation.finish()
                } catch {
                    continuation.yield(with: .failure(error))
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    public static func verifyHTTPResponse(_ response: URLResponse, data: Data?) throws {
        guard let http = response as? HTTPURLResponse else {
            throw LLMError.networkError("Non-HTTP response received.")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw LLMError.httpError(http.statusCode, data)
        }
    }

    public static func debugLogHTTPResponse(_ response: URLResponse, data: Data) {
        let env = ProcessInfo.processInfo.environment
        guard env["LLM_PROVIDERKIT_DEBUG_HTTP"] == "1" || env["LLM_PROVIDERKIT_DEBUG_HTTP"] == "true" else {
            return
        }

        let status = (response as? HTTPURLResponse)?.statusCode.description ?? "<non-HTTP>"
        let url = response.url?.absoluteString ?? "<url>"
        let bodyText = String(data: data, encoding: .utf8) ?? "<non-UTF8 body: \(data.count) bytes>"
        print("""
        \n========== LLMProviderKit HTTP Response =========
        Provider: \(Self.name)
        Status: \(status)
        URL: \(url)
        Body bytes: \(data.count)
        Body:
        \(bodyText)
        =========================================\n
        """)
    }
}
