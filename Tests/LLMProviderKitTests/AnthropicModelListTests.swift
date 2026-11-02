import Foundation
import Testing
@testable import LLMProviderKit
@testable import LLMProviderKitAnthropic

/// The model list: a refused key is reported, not hidden behind the built-in
/// list (2026-11-02: an expired key showed a stale list with no explanation);
/// an unreachable endpoint still falls back.
@Suite(.serialized)
struct AnthropicModelListTests {
    private func provider(status: Int, body: String) -> AnthropicProvider {
        StatusMockURLProtocol.status = status
        StatusMockURLProtocol.body = Data(body.utf8)
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StatusMockURLProtocol.self]
        return AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "expired", model: "claude-haiku-5-5"),
                                 urlSession: URLSession(configuration: config))
    }

    @Test func aRefusedKeyIsReported() async {
        for status in [401, 403] {
            do {
                _ = try await provider(status: status, body: #"{"type":"error","error":{"type":"authentication_error"}}"#).availableModels()
                Issue.record("expected an error for \(status)")
            } catch LLMError.httpError(let code, _) {
                #expect(code == status)
                #expect(LLMError.httpError(code, nil).userMessage.summary.contains("may have expired"))
            } catch {
                Issue.record("unexpected \(error)")
            }
        }
    }

    @Test func anOverloadedEndpointStillGivesTheBuiltInList() async throws {
        let models = try await provider(status: 529, body: "{}").availableModels()
        #expect(models.map(\.id) == AnthropicProvider.curatedModels.map(\.id))
    }

    @Test func theBuiltInListHasTheClaude5Family() {
        let ids = AnthropicProvider.curatedModels.map(\.id)
        for id in ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5", "claude-haiku-5-5"] {
            #expect(ids.contains(id), "\(id)")
        }
        #expect(Set(ids).count == ids.count, "no duplicates")
    }
}

private final class StatusMockURLProtocol: URLProtocol {
    nonisolated(unsafe) static var status = 200
    nonisolated(unsafe) static var body = Data()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let response = HTTPURLResponse(url: request.url!, statusCode: Self.status, httpVersion: nil,
                                       headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
