import Testing
import Foundation
import LLMProviderKit
import LLMProviderKitAnthropic
import LLMProviderKitGemini

/// A tool call parsed from a Gemini or Anthropic response is later embedded
/// as-is into an OpenAI-wire request body (OpenAIProvider.swift re-sends the
/// `arguments` string verbatim). If the parser doesn't sort the re-serialized
/// keys, a conversation that started on Gemini/Anthropic and continues on an
/// OpenAI-wire provider sends a non-deterministic argument string, defeating
/// byte-identical requests and prompt caching.
struct ToolCallArgumentsSortedKeysTests {
    let request = LLMRequest(model: "m", messages: [LLMMessage(role: .user, content: "hi")])

    @Test func geminiToolCallArgumentsComeOutSorted() throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "k", model: "gemini-2.0-flash"))
        let body = #"""
        {"candidates":[{"content":{"parts":[{"functionCall":{"name":"tool_1","args":{"path":"/tmp","command":"ls","mode":"x","limit":"5"}}}]},"finishReason":"STOP"}]}
        """#
        let first = try provider.parseResponse(Data(body.utf8), request: request)
        let second = try provider.parseResponse(Data(body.utf8), request: request)
        let args = try #require(first.toolCalls.first?.arguments)
        #expect(args == #"{"command":"ls","limit":"5","mode":"x","path":"\/tmp"}"#)
        #expect(args == second.toolCalls.first?.arguments)
    }

    @Test func anthropicToolCallArgumentsComeOutSorted() throws {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "k", model: "claude-sonnet-4-6"))
        let body = #"""
        {"id":"m","type":"message","role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"tool_1","input":{"path":"/tmp","command":"ls","mode":"x","limit":"5"}}],"stop_reason":"tool_use"}
        """#
        let first = try provider.parseResponse(Data(body.utf8), request: request)
        let second = try provider.parseResponse(Data(body.utf8), request: request)
        let args = try #require(first.toolCalls.first?.arguments)
        #expect(args == #"{"command":"ls","limit":"5","mode":"x","path":"\/tmp"}"#)
        #expect(args == second.toolCalls.first?.arguments)
    }
}
