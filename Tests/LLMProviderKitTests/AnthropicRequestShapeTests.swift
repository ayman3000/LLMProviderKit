import Foundation
import Testing
@testable import LLMProviderKit
@testable import LLMProviderKitAnthropic

/// Two rules of the Messages API that a tool-using agent trips over
/// (compared against Hermes's Anthropic adapter, 2026-09-30):
/// - every tool_result for one assistant turn's tool_use blocks must sit in
///   ONE user message right after it, or the request is refused;
/// - `max_tokens` is required, and 4,096 cuts a long file edit short.
struct AnthropicRequestShapeTests {

    private func body(_ messages: [LLMMessage], model: String = "claude-sonnet-4-6",
                      maxTokens: Int? = nil) throws -> [String: Any] {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "k", model: model))
        var request = LLMRequest(model: model, messages: messages)
        request.maxTokens = maxTokens
        let data = try #require(provider.prepareRequest(request, stream: false).httpBody)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func parallelToolResultsShareOneUserMessage() throws {
        let calls = [LLMToolCall(id: "a1", name: "read_file", arguments: "{}"),
                     LLMToolCall(id: "b2", name: "list_dir", arguments: "{}")]
        var assistant = LLMMessage(role: .assistant, content: "")
        assistant.toolCalls = calls
        var r1 = LLMMessage(role: .tool, content: "file text"); r1.toolCallId = "a1"
        var r2 = LLMMessage(role: .tool, content: "dir listing"); r2.toolCallId = "b2"
        let b = try body([LLMMessage(role: .user, content: "go"), assistant, r1, r2,
                          LLMMessage(role: .assistant, content: "done")])
        let messages = try #require(b["messages"] as? [[String: Any]])
        // user, assistant(tool_use ×2), user(tool_result ×2), assistant
        #expect(messages.count == 4)
        let results = try #require(messages[2]["content"] as? [[String: Any]])
        #expect(messages[2]["role"] as? String == "user")
        #expect(results.map { $0["tool_use_id"] as? String } == ["a1", "b2"])
        #expect(results.allSatisfy { $0["type"] as? String == "tool_result" })
    }

    @Test func aSingleToolResultIsUnchanged() throws {
        var assistant = LLMMessage(role: .assistant, content: "")
        assistant.toolCalls = [LLMToolCall(id: "a1", name: "read_file", arguments: "{}")]
        var r1 = LLMMessage(role: .tool, content: "file text"); r1.toolCallId = "a1"
        let b = try body([LLMMessage(role: .user, content: "go"), assistant, r1])
        let messages = try #require(b["messages"] as? [[String: Any]])
        #expect(messages.count == 3)
        #expect((messages[2]["content"] as? [[String: Any]])?.count == 1)
    }

    @Test func theOutputLimitFollowsTheModelWhenTheCallerSetsNone() throws {
        #expect(try body([LLMMessage(role: .user, content: "hi")], model: "claude-sonnet-4-6")["max_tokens"] as? Int == 32_000)
        #expect(try body([LLMMessage(role: .user, content: "hi")], model: "claude-opus-4-8")["max_tokens"] as? Int == 32_000)
        #expect(try body([LLMMessage(role: .user, content: "hi")], model: "claude-3-5-sonnet-20241022")["max_tokens"] as? Int == 8_192)
        #expect(try body([LLMMessage(role: .user, content: "hi")], model: "claude-3-opus-20240229")["max_tokens"] as? Int == 4_096)
        // An explicit limit still wins.
        #expect(try body([LLMMessage(role: .user, content: "hi")], maxTokens: 500)["max_tokens"] as? Int == 500)
    }

    /// Checked live (2026-10-01): `thinking: {type: adaptive}` with
    /// `output_config.effort` is accepted, the reply carries a signed thinking
    /// block, and the next turn is accepted with that block left out.
    @Test func modelsThatTakeAnEffortAlsoGetAdaptiveThinking() throws {
        var r = LLMRequest(model: "claude-sonnet-4-6", messages: [LLMMessage(role: .user, content: "hi")])
        r.reasoningEffort = .low
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "k", model: "claude-sonnet-4-6"))
        let data = try #require(provider.prepareRequest(r, stream: false).httpBody)
        let b = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect((b["thinking"] as? [String: Any])?["type"] as? String == "adaptive")
        #expect((b["output_config"] as? [String: Any])?["effort"] as? String == "low")
    }

    @Test func noEffortMeansNoThinkingField() throws {
        let b = try body([LLMMessage(role: .user, content: "hi")], model: "claude-sonnet-4-6")
        #expect(b["thinking"] == nil)
        // An older model that takes no effort level never gets the field either.
        var r = LLMRequest(model: "claude-3-5-sonnet-20241022", messages: [LLMMessage(role: .user, content: "hi")])
        r.reasoningEffort = .high
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "k", model: "claude-3-5-sonnet-20241022"))
        let data = try #require(provider.prepareRequest(r, stream: false).httpBody)
        let old = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(old["thinking"] == nil)
    }
}
