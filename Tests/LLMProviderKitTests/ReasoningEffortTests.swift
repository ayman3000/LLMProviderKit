import Testing
import Foundation
import LLMProviderKit
@testable import LLMProviderKitAnthropic

/// Effort is additive: a request that does not ask for a level must produce the
/// byte-for-byte body it produced before the parameter existed, because every
/// existing caller passes nothing.
struct ReasoningEffortTests {
    private static func provider() -> AnthropicProvider {
        AnthropicProvider(configuration: LLMProviderConfiguration(
            name: "anthropic",
            baseURL: URL(string: "https://api.anthropic.com")!,
            apiKey: "test-key"
        ))
    }

    private static func body(_ request: LLMRequest) throws -> [String: Any] {
        let urlRequest = try provider().prepareRequest(request, stream: false)
        let data = try #require(urlRequest.httpBody)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func request(effort: LLMReasoningEffort?) -> LLMRequest {
        LLMRequest(
            model: "claude-sonnet-4-6",
            messages: [LLMMessage(role: .user, content: "hi")],
            reasoningEffort: effort
        )
    }

    @Test func noEffortSendsNoOutputConfig() throws {
        let body = try Self.body(Self.request(effort: nil))
        #expect(body["output_config"] == nil)
    }

    @Test func effortIsSentAsOutputConfig() throws {
        let body = try Self.body(Self.request(effort: .medium))
        let config = try #require(body["output_config"] as? [String: Any])
        #expect(config["effort"] as? String == "medium")
    }

    /// The wire values are Anthropic's own spelling. A rename here is an API
    /// error, not a cosmetic change.
    @Test func everyLevelKeepsItsWireSpelling() {
        #expect(LLMReasoningEffort.allCases.map(\.rawValue).sorted()
                == ["high", "low", "max", "medium", "xhigh"])
    }

    /// Effort must not disturb what was already in the body.
    @Test func effortLeavesTheRestOfTheBodyAlone() throws {
        let plain = try Self.body(Self.request(effort: nil))
        let withEffort = try Self.body(Self.request(effort: .low))
        for key in plain.keys {
            #expect(withEffort[key] != nil, "effort dropped \(key) from the body")
        }
        #expect(withEffort.count == plain.count + 1)
    }
}
