import Foundation
import Testing
@testable import LLMProviderKit
@testable import LLMProviderKitOpenAI

/// gpt-5.6-luna on api.openai.com (checked live, 2026-09-29): `max_tokens` →
/// "Unsupported parameter … use 'max_completion_tokens'"; temperature 0.7 →
/// "Only the default (1) value is supported"; reasoning_effort none/low/
/// medium/high/xhigh accepted, minimal rejected.
struct OpenAIReasoningModelTests {
    private func body(model: String, base: String = "https://api.openai.com/v1",
                      effort: LLMReasoningEffort? = nil) throws -> [String: Any] {
        let provider = OpenAIProvider(configuration: LLMProviderConfiguration(
            name: "openai", baseURL: URL(string: base)!, apiKey: "k", defaultModel: model))
        let request = LLMRequest(model: model, messages: [LLMMessage(role: .user, content: "hi")],
                                 temperature: 0.7, maxTokens: 200, reasoningEffort: effort)
        let data = try #require(provider.prepareRequest(request, stream: false).httpBody)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func openAIGetsMaxCompletionTokensAndNoTemperatureForReasoningModels() throws {
        for model in ["gpt-5.6-luna", "gpt-6-sol", "o3-mini", "gpt-5"] {
            let b = try body(model: model)
            #expect(b["max_completion_tokens"] as? Int == 200, "\(model)")
            #expect(b["max_tokens"] == nil, "\(model)")
            #expect(b["temperature"] == nil, "\(model)")
        }
    }

    @Test func olderOpenAIModelsKeepTheirTemperature() throws {
        let b = try body(model: "gpt-4o")
        #expect(b["temperature"] as? Double == 0.7)
        #expect(b["max_completion_tokens"] as? Int == 200)   // OpenAI accepts it for every model
    }

    @Test func compatibleServicesKeepTheOldShape() throws {
        let b = try body(model: "gpt-5.6-luna", base: "https://api.groq.com/openai/v1")
        #expect(b["max_tokens"] as? Int == 200 && b["max_completion_tokens"] == nil)
        #expect(b["temperature"] as? Double == 0.7)
    }

    @Test func theEffortLevelIsSent() throws {
        #expect(try body(model: "gpt-5.6-luna", effort: .high)["reasoning_effort"] as? String == "high")
        #expect(try body(model: "gpt-5.6-luna", effort: .off)["reasoning_effort"] as? String == "none")
        #expect(try body(model: "gpt-5.6-luna")["reasoning_effort"] == nil)
        let v = try #require(OpenAIProvider(configuration: LLMProviderConfiguration(
            name: "openai", baseURL: URL(string: "https://api.openai.com/v1")!, apiKey: "k", defaultModel: "m"))
            .effortVocabulary(for: "gpt-5.6-luna"))
        #expect(v.supported.contains(.xhigh) && !v.supported.contains(.minimal))
        #expect(OpenAIProvider(configuration: LLMProviderConfiguration(
            name: "openai", baseURL: URL(string: "https://api.openai.com/v1")!, apiKey: "k", defaultModel: "m"))
            .effortVocabulary(for: "gpt-4o") == nil)
    }
}
