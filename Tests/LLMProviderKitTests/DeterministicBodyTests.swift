import Testing
import Foundation
import LLMProviderKit
import LLMProviderKitAnthropic
import LLMProviderKitOpenAI
import LLMProviderKitOllama
import LLMProviderKitGemini
import LLMProviderKitOpenRouter

/// Prompt caching reuses the longest identical start of a request. A body whose
/// JSON keys come out in another order on the next call is a different request
/// to the cache, so every provider must turn the same request into the same
/// bytes, every time.
struct DeterministicBodyTests {
    /// Thirty tools with eight-property schemas and a tool-calling history:
    /// enough dictionaries that unsorted key order shows within a few builds.
    static func request(model: String, effort: LLMReasoningEffort? = nil) -> LLMRequest {
        var properties: [String: Any] = [:]
        for name in ["path", "command", "query", "limit", "offset", "mode", "pattern", "encoding"] {
            properties[name] = ["type": "string", "description": "the \(name)"]
        }
        let schema: [String: Any] = ["type": "object", "properties": properties, "required": ["path"]]
        let tools = (0..<30).map {
            LLMToolDefinition(name: "tool_\($0)", description: "Tool \($0).", parameters: schema)
        }
        var assistant = LLMMessage(role: .assistant, content: "")
        assistant.toolCalls = [LLMToolCall(id: "call_1", name: "tool_1",
                                           arguments: #"{"command":"ls","path":"/tmp","limit":"5","mode":"x"}"#)]
        var result = LLMMessage(role: .tool, content: "ok")
        result.toolCallId = "call_1"
        return LLMRequest(model: model, messages: [
            LLMMessage(role: .system, content: "be brief"),
            LLMMessage(role: .user, content: "list /tmp"),
            assistant,
            result,
        ], tools: tools, reasoningEffort: effort)
    }

    /// Distinct bodies over `count` builds, with unrelated allocations in
    /// between so each build's dictionaries land at new addresses.
    static func distinctBodies(_ count: Int = 60, _ build: () throws -> Data?) rethrows -> Int {
        var bodies = Set<Data>()
        var noise: [[Int]] = []
        for i in 0..<count {
            noise.append([Int](repeating: i, count: 1 + (i * 37) % 257))
            if noise.count > 16 { noise.removeFirst(8) }
            if let body = try build() { bodies.insert(body) }
        }
        return bodies.count
    }

    @Test func anthropicBodiesAreByteIdentical() throws {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "k", model: "claude-sonnet-4-6"))
        let request = Self.request(model: "claude-sonnet-4-6")
        #expect(try Self.distinctBodies { try provider.prepareRequest(request, stream: false).httpBody } == 1)
    }

    @Test func openAIBodiesAreByteIdentical() throws {
        let provider = OpenAIProvider(configuration: OpenAIProvider.openAI(apiKey: "k", model: "gpt-4o"))
        let request = Self.request(model: "gpt-4o")
        #expect(try Self.distinctBodies { try provider.prepareRequest(request, stream: true).httpBody } == 1)
    }

    @Test func ollamaBodiesAreByteIdentical() throws {
        let provider = OllamaProvider(configuration: OllamaProvider.local(model: "llama3.2"))
        let request = Self.request(model: "llama3.2")
        #expect(try Self.distinctBodies { try provider.prepareRequest(request, stream: true).httpBody } == 1)
    }

    @Test func geminiBodiesAreByteIdentical() throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "k", model: "gemini-2.0-flash"))
        let request = Self.request(model: "gemini-2.0-flash")
        #expect(try Self.distinctBodies { try provider.prepareRequest(request, stream: false).httpBody } == 1)
    }

    /// With an effort level OpenRouter re-serializes the inner body: that pass
    /// must sort too.
    @Test func openRouterBodiesAreByteIdenticalWithAnEffort() throws {
        let provider = OpenRouterProvider(configuration: LLMProviderConfiguration(
            name: "openrouter", baseURL: URL(string: "https://openrouter.ai/api/v1")!, apiKey: "k"))
        let request = Self.request(model: "anthropic/claude-sonnet-4", effort: .xhigh)
        #expect(try Self.distinctBodies { try provider.prepareRequest(request, stream: false).httpBody } == 1)
    }

    @Test func keysComeOutSorted() throws {
        let provider = OpenAIProvider(configuration: OpenAIProvider.openAI(apiKey: "k", model: "gpt-4o"))
        let body = try #require(try provider.prepareRequest(Self.request(model: "gpt-4o"), stream: false).httpBody)
        let text = String(decoding: body, as: UTF8.self)
        #expect(text.hasPrefix(#"{"messages":[{"content":"be brief","role":"system"}"#))
    }
}
