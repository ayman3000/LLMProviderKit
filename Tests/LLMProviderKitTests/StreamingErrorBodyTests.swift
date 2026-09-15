import Testing
import Foundation
@testable import LLMProviderKit

/// A failing stream still has a body, and it is the only thing that says why.
/// Before this, every provider reported a bare "HTTP error: 404" — a number
/// the user can do nothing with — while the server was explaining itself in a
/// response the byte stream still held.
private final class FailingStreamURLProtocol: URLProtocol {
    nonisolated(unsafe) static var status = 404
    nonisolated(unsafe) static var body = Data()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(url: request.url!, statusCode: Self.status,
                                       httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if !Self.body.isEmpty { client?.urlProtocol(self, didLoad: Self.body) }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private struct StreamingProvider: LLMProvider {
    static let name = "streaming-test"
    let configuration: LLMProviderConfiguration
    let urlSession: URLSession

    init(session: URLSession) {
        configuration = LLMProviderConfiguration(
            name: Self.name, baseURL: URL(string: "https://example.invalid/v1")!,
            apiKey: "k", defaultModel: "m")
        urlSession = session
    }

    func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        var r = URLRequest(url: configuration.baseURL)
        r.httpMethod = "POST"
        r.httpBody = Data("{}".utf8)
        return r
    }
    func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] { [] }
    func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        LLMResponse(text: "", finishReason: .stop, request: request, providerName: Self.name)
    }
}

/// Serialized: the stub's response is static, so parallel cases would overwrite
/// each other's status and body mid-request.
@Suite(.serialized)
struct StreamingErrorBodyTests {
    private func provider() -> StreamingProvider {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FailingStreamURLProtocol.self]
        return StreamingProvider(session: URLSession(configuration: configuration))
    }

    private func streamError(status: Int, body: String) async -> (any Error)? {
        FailingStreamURLProtocol.status = status
        FailingStreamURLProtocol.body = Data(body.utf8)
        let request = LLMRequest(model: "m", messages: [LLMMessage(role: .user, content: "hi")])
        do {
            for try await _ in provider().stream(request) {}
            return nil
        } catch { return error }
    }

    /// The whole point: the server's explanation reaches the user.
    @Test func theServersExplanationSurvives() async throws {
        let message = #"{"error":{"message":"models/gemini-2.5-pro is not found","code":404}}"#
        let error = try #require(await streamError(status: 404, body: message))
        let text = (error as? LLMError)?.errorDescription ?? "\(error)"
        #expect(text.contains("404"))
        #expect(text.contains("is not found"), "the body was discarded: \(text)")
    }

    /// The status must still be right when the body is empty — a bare number
    /// is poor, but losing the status would be worse.
    @Test func anEmptyBodyStillReportsTheStatus() async throws {
        let error = try #require(await streamError(status: 500, body: ""))
        #expect(((error as? LLMError)?.errorDescription ?? "").contains("500"))
    }

    /// A runaway error body must not be buffered whole.
    @Test func theErrorBodyIsCapped() async throws {
        let huge = String(repeating: "x", count: 400_000)
        let error = try #require(await streamError(status: 400, body: huge))
        let text = (error as? LLMError)?.errorDescription ?? "\(error)"
        #expect(text.contains("400"))
        // errorDescription already truncates for display; the cap is what stops
        // an unbounded read, and the error still arrives.
        #expect(text.count < 100_000)
    }
}
