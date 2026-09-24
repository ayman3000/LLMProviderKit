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

    /// Which effort levels this provider accepts for `model`, or nil when it
    /// takes no effort level at all.
    ///
    /// Declared per model because a vocabulary belongs to a model *served by
    /// this endpoint*: the same model reached through a different gateway can
    /// accept a different set. The default returns nil, so a provider that has
    /// not been taught about effort silently sends none — the safe direction,
    /// since an unsupported level is an HTTP 400 on several wires.
    func effortVocabulary(for model: String) -> LLMEffortVocabulary?

    /// Optional: fetch the list of models available from this provider.
    ///
    /// Not every provider exposes a model list endpoint. The default
    /// implementation throws `LLMError.unsupportedOperation`.
    func availableModels() async throws -> [LLMModelInfo]

    /// Non-streaming completion. Declared as a requirement (with a default HTTP
    /// implementation below) so an **in-process** provider — e.g. an on-device
    /// MLX/llama.cpp backend — can override it directly instead of going through
    /// `prepareRequest`/`parseResponse`. HTTP providers rely on the default.
    func complete(_ request: LLMRequest) async throws -> LLMResponse

    /// Streaming completion. A requirement for the same reason as `complete`.
    func stream(_ request: LLMRequest) -> AsyncThrowingStream<LLMStreamChunk, Error>
}

extension LLMProvider {
    public var urlSession: URLSession { LLMNetworking.session }

    public func effortVocabulary(for model: String) -> LLMEffortVocabulary? { nil }

    /// The level to actually put on the wire for `request`: the caller's ask,
    /// resolved onto what this model accepts. Providers call this instead of
    /// reading `request.reasoningEffort` directly, so a level a wire cannot
    /// express is clamped at the boundary rather than sent and refused.
    public func wireEffort(for request: LLMRequest) -> LLMReasoningEffort? {
        guard let vocabulary = effortVocabulary(for: request.model) else { return nil }
        return vocabulary.clamp(request.reasoningEffort)
    }

    // Default (throwing) implementations of the HTTP-shaped hooks, so a provider
    // that overrides `complete`/`stream` (in-process, no URLRequest) doesn't have
    // to implement them. HTTP providers implement all three as before.
    public func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        throw LLMError.unsupportedOperation("\(Self.name) does not build URL requests (override complete/stream instead).")
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        []
    }

    public func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        throw LLMError.unsupportedOperation("\(Self.name) does not parse HTTP responses (override complete instead).")
    }

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
                    var urlRequest = try self.prepareRequest(resolvedRequest, stream: true)
                    // Stalled-stream watchdog (off unless request.stallTimeout is set).
                    // The connection's own idle timeout must not fire first:
                    // the watchdog is the policy, the idle timeout a backstop.
                    if let limit = resolvedRequest.stallTimeout {
                        urlRequest.timeoutInterval = max(urlRequest.timeoutInterval, limit + 30)
                    }
                    let clock = LLMProgressClock()
                    try await LLMStreamWatchdog.run(limit: resolvedRequest.stallTimeout, clock: clock) {
                    let (bytes, response) = try await self.urlSession.bytes(for: urlRequest)
                    // A failing stream still has a body, and it is the only
                    // thing that says WHY. Passing nil here left every provider
                    // reporting a bare "HTTP error: 404" — a number the user
                    // can do nothing with — while the server was explaining
                    // itself in the response the byte stream still held.
                    try await Self.verifyStreamingResponse(response, bytes: bytes)

                    // Fragmented tool calls (OpenAI-style wires) are joined
                    // here, so every consumer sees whole calls.
                    var assembler = StreamToolCallAssembler()
                    var pendingLineBytes = Data()
                    for try await byte in bytes {
                        if byte == 0x0A { // newline
                            if !pendingLineBytes.isEmpty {
                                if pendingLineBytes.last == 0x0D { pendingLineBytes.removeLast() }
                                guard let line = String(data: pendingLineBytes, encoding: .utf8) else {
                                    throw LLMError.invalidResponse("Streaming response contained a non-UTF-8 line.")
                                }
                                if !LLMStreamWatchdog.isKeepAlive(line) { clock.touch() }
                                let chunks = try self.parseStreamLine(line, request: resolvedRequest)
                                for chunk in chunks.flatMap({ assembler.consume($0) }) {
                                    continuation.yield(chunk)
                                    if case .finish = chunk { break }
                                }
                            }
                            pendingLineBytes.removeAll(keepingCapacity: true)
                        } else {
                            pendingLineBytes.append(byte)
                        }
                    }

                    if !pendingLineBytes.isEmpty {
                        if pendingLineBytes.last == 0x0D { pendingLineBytes.removeLast() }
                        guard let line = String(data: pendingLineBytes, encoding: .utf8) else {
                            throw LLMError.invalidResponse("Streaming response contained a non-UTF-8 line.")
                        }
                        let chunks = try self.parseStreamLine(line, request: resolvedRequest)
                        for chunk in chunks.flatMap({ assembler.consume($0) }) { continuation.yield(chunk) }
                    }
                    for chunk in assembler.flush() { continuation.yield(chunk) }
                    }

                    continuation.finish()
                } catch let error as URLError where error.code == .timedOut && request.stallTimeout != nil {
                    // A silent connection is a stall too: one treatment either way.
                    continuation.yield(with: .failure(LLMStreamStalled(seconds: request.stallTimeout ?? 0)))
                } catch {
                    continuation.yield(with: .failure(error))
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    /// Verify a streaming response, reading the error body when there is one.
    ///
    /// The body is drained only on a non-2xx, and capped: an error payload is
    /// small, and a stream that keeps talking must not be buffered whole.
    public static func verifyStreamingResponse(
        _ response: URLResponse,
        bytes: URLSession.AsyncBytes,
        maxErrorBodyBytes: Int = 64 * 1024
    ) async throws {
        guard let http = response as? HTTPURLResponse else {
            throw LLMError.networkError("Non-HTTP response received.")
        }
        guard (200..<300).contains(http.statusCode) else {
            var body = Data()
            // Best-effort: a body that fails mid-read still beats no body, so
            // a throw here must not replace the status the caller needs.
            do {
                for try await byte in bytes {
                    body.append(byte)
                    if body.count >= maxErrorBodyBytes { break }
                }
            } catch { /* keep whatever arrived */ }
            throw LLMError.httpError(http.statusCode, body.isEmpty ? nil : body)
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

/// The session every provider uses unless it is given another one.
///
/// `URLSession.shared` gives up after 60 s without data. A large model that
/// is serving several requests at once can take longer than that before its
/// first token, and every retry then hits the same wall — a review with three
/// concurrent sub-agents on Ollama Cloud lost two of them exactly this way
/// (2026-09-08). Generous per-request idle timeout; a long overall ceiling so
/// a slow streamed answer is never cut off mid-way.
public enum LLMNetworking {
    public static let requestTimeout: TimeInterval = 300
    public static let resourceTimeout: TimeInterval = 1800

    public static let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = requestTimeout
        config.timeoutIntervalForResource = resourceTimeout
        return URLSession(configuration: config)
    }()
}
