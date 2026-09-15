import Testing
import Foundation
import LLMProviderKit
@testable import LLMProviderKitAnthropic
import LLMProviderKitOpenRouter

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

/// OpenRouter takes effort one level up from the OpenAI-compatible body, so it
/// is edited after the inner provider builds it. The untouched path matters
/// most: with no level asked for, the body must be exactly what it was.
struct OpenRouterReasoningEffortTests {
    private static func provider() -> OpenRouterProvider {
        OpenRouterProvider(configuration: LLMProviderConfiguration(
            name: "openrouter",
            baseURL: URL(string: "https://openrouter.ai/api/v1")!,
            apiKey: "test-key"
        ))
    }

    private static func body(effort: LLMReasoningEffort?) throws -> [String: Any] {
        let request = LLMRequest(
            model: "anthropic/claude-sonnet-4",
            messages: [LLMMessage(role: .user, content: "hi")],
            reasoningEffort: effort
        )
        let data = try #require(provider().prepareRequest(request, stream: false).httpBody)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func noEffortLeavesTheBodyUntouched() throws {
        #expect(try Self.body(effort: nil)["reasoning"] == nil)
    }

    @Test func effortBecomesTheReasoningObject() throws {
        let reasoning = try #require(try Self.body(effort: .xhigh)["reasoning"] as? [String: Any])
        #expect(reasoning["effort"] as? String == "xhigh")
    }

    @Test func effortAddsExactlyOneKey() throws {
        let plain = try Self.body(effort: nil)
        let withEffort = try Self.body(effort: .low)
        #expect(withEffort.count == plain.count + 1)
        for key in plain.keys { #expect(withEffort[key] != nil, "effort dropped \(key)") }
    }
}

/// Anthropic advertises no per-model signal for effort, so the gate is a
/// documented id list. It must fail CLOSED: an unknown model hides the control
/// rather than sending a level that answers 400.
struct AnthropicEffortCapabilityTests {
    @Test func documentedFamiliesAccessEffort() {
        for id in ["claude-fable-5", "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5",
                   "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
                   "claude-opus-4-5-20251101", "claude-sonnet-4-6", "claude-mythos-5-1"] {
            #expect(AnthropicProvider.acceptsReasoningEffort(id), "\(id) should accept effort")
        }
    }

    @Test func everythingElseIsRefused() {
        for id in ["claude-haiku-4-5-20251001", "claude-sonnet-4-20250514",
                   "claude-3-opus-20240229", "claude-3-5-sonnet-20241022",
                   "some-model-nobody-has-heard-of"] {
            #expect(!AnthropicProvider.acceptsReasoningEffort(id), "\(id) must NOT accept effort")
        }
    }

    @Test func curatedModelsCarryTheCapability() {
        let byID = Dictionary(uniqueKeysWithValues:
            AnthropicProvider.curatedModels.map { ($0.id, $0) })
        #expect(byID[AnthropicModel.sonnet46]?.capabilities.contains(.reasoningEffort) == true)
        #expect(byID[AnthropicModel.opus47]?.capabilities.contains(.reasoningEffort) == true)
        #expect(byID[AnthropicModel.haiku45]?.capabilities.contains(.reasoningEffort) == false)
    }
}
