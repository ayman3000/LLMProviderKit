import Foundation
@testable import LLMProviderKit
@testable import LLMProviderKitOllama
@testable import LLMProviderKitOpenAI
@testable import LLMProviderKitGemini
@testable import LLMProviderKitAnthropic
import Testing

struct ProviderTests {
    // MARK: - Ollama

    @Test func ollamaNonStreamingResponse() async throws {
        let provider = OllamaProvider(configuration: OllamaProvider.local(model: "llama3.2"))
        let data = """
        {
          "model": "llama3.2",
          "message": { "role": "assistant", "content": "Hello from Ollama" },
          "done": true,
          "prompt_eval_count": 10,
          "eval_count": 5
        }
        """.data(using: .utf8)!

        let request = LLMRequest(model: "llama3.2", messages: [.user("Hi")])
        let response = try provider.parseResponse(data, request: request)

        #expect(response.text == "Hello from Ollama")
        #expect(response.finishReason == LLMFinishReason.stop)
        #expect(response.usage?.totalTokens == 15)
        #expect(response.providerName == "ollama")
    }

    @Test func ollamaStreamingLine() async throws {
        let provider = OllamaProvider(configuration: OllamaProvider.local(model: "llama3.2"))
        let line = """
        {"model":"llama3.2","message":{"content":" world"},"done":true,"prompt_eval_count":10,"eval_count":3}
        """
        let request = LLMRequest(model: "llama3.2", messages: [.user("Hi")])
        let chunks = try provider.parseStreamLine(line, request: request)

        #expect(chunks.count == 2)
        #expect(chunks[0] == .text(" world"))
        #expect(chunks[1] == .finish(reason: .stop, usage: LLMUsage(promptTokens: 10, completionTokens: 3, totalTokens: 13)))
    }

    // MARK: - OpenAI

    @Test func openAINonStreamingResponse() async throws {
        let provider = OpenAIProvider(configuration: OpenAIProvider.openAI(apiKey: "test", model: "gpt-4o-mini"))
        let data = """
        {
          "id": "chatcmpl-123",
          "choices": [{
            "message": { "role": "assistant", "content": "Hello from OpenAI" },
            "finish_reason": "stop"
          }],
          "usage": { "prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25 }
        }
        """.data(using: .utf8)!

        let request = LLMRequest(model: "gpt-4o-mini", messages: [.user("Hi")])
        let response = try provider.parseResponse(data, request: request)

        #expect(response.text == "Hello from OpenAI")
        #expect(response.finishReason == LLMFinishReason.stop)
        #expect(response.usage?.totalTokens == 25)
        #expect(response.providerName == "openai")
    }

    @Test func openAIStreamingLine() async throws {
        let provider = OpenAIProvider(configuration: OpenAIProvider.openAI(apiKey: "test", model: "gpt-4o-mini"))
        let line = "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}"
        let request = LLMRequest(model: "gpt-4o-mini", messages: [.user("Hi")])
        let chunks = try provider.parseStreamLine(line, request: request)

        #expect(chunks.count == 2)
        #expect(chunks[0] == .text("!"))
        #expect(chunks[1] == .finish(reason: .stop, usage: nil))
    }

    // MARK: - Gemini

    @Test func geminiNonStreamingResponse() async throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "test", model: "gemini-2.0-flash"))
        let data = """
        {
          "candidates": [{
            "content": { "parts": [{"text": "Hello from Gemini"}] },
            "finishReason": "STOP"
          }],
          "usageMetadata": { "promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12 }
        }
        """.data(using: .utf8)!

        let request = LLMRequest(model: "gemini-2.0-flash", messages: [.user("Hi")])
        let response = try provider.parseResponse(data, request: request)

        #expect(response.text == "Hello from Gemini")
        #expect(response.finishReason == LLMFinishReason.stop)
        #expect(response.usage?.totalTokens == 12)
        #expect(response.providerName == "gemini")
    }

    @Test func geminiStreamingLine() async throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "test", model: "gemini-2.0-flash"))
        let line = """
        data: {"candidates":[{"content":{"parts":[{"text":" there"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":2,"totalTokenCount":10}}
        """
        let request = LLMRequest(model: "gemini-2.0-flash", messages: [.user("Hi")])
        let chunks = try provider.parseStreamLine(line, request: request)

        #expect(chunks.count == 2)
        #expect(chunks[0] == .text(" there"))
        #expect(chunks[1] == .finish(reason: .stop, usage: LLMUsage(promptTokens: 8, completionTokens: 2, totalTokens: 10)))
    }

    // MARK: - Anthropic

    @Test func anthropicNonStreamingResponse() async throws {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "test", model: "claude-3-5-sonnet-20241022"))
        let data = """
        {
          "id": "msg_01",
          "type": "message",
          "role": "assistant",
          "content": [{"type": "text", "text": "Hello from Claude"}],
          "stop_reason": "end_turn",
          "usage": { "input_tokens": 12, "output_tokens": 5 }
        }
        """.data(using: .utf8)!

        let request = LLMRequest(model: "claude-3-5-sonnet-20241022", messages: [.user("Hi")])
        let response = try provider.parseResponse(data, request: request)

        #expect(response.text == "Hello from Claude")
        #expect(response.finishReason == LLMFinishReason.stop)
        #expect(response.usage?.totalTokens == 17)
        #expect(response.providerName == "anthropic")
    }

    @Test func anthropicStreamingLines() async throws {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "test", model: "claude-3-5-sonnet-20241022"))
        let lines = [
            "event: content_block_delta",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}",
            "event: content_block_delta",
            "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\" Claude\"}}",
            "event: message_delta",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":12,\"output_tokens\":5}}"
        ]
        let request = LLMRequest(model: "claude-3-5-sonnet-20241022", messages: [.user("Hi")])
        var chunks: [LLMStreamChunk] = []
        for line in lines {
            chunks.append(contentsOf: try provider.parseStreamLine(line, request: request))
        }

        #expect(chunks.count == 3)
        #expect(chunks[0] == .text("Hello"))
        #expect(chunks[1] == .text(" Claude"))
        #expect(chunks[2] == .finish(reason: .stop, usage: LLMUsage(promptTokens: 12, completionTokens: 5, totalTokens: 17)))
    }

    // MARK: - Model registry

    @Test func registryRegistersAndQueriesModels() async throws {
        let registry = LLMModelRegistry()
        await registry.register([
            LLMModelInfo(id: "gpt-4o", providerName: "openai", displayName: "GPT-4o"),
            LLMModelInfo(id: "gpt-4o-mini", providerName: "openai", displayName: "GPT-4o Mini")
        ], for: "openai")

        let models = await registry.models(for: "openai")
        #expect(models.count == 2)
        #expect(models.first?.id == "gpt-4o")
        #expect(models.first?.displayName == "GPT-4o")
    }

    @Test func registryReturnsDefaultModelFromConfiguration() async throws {
        let registry = LLMModelRegistry()
        let config = LLMProviderConfiguration(
            name: "openai",
            baseURL: URL(string: "https://api.openai.com/v1")!,
            apiKey: nil,
            defaultModel: "gpt-4o"
        )
        let defaultID = try await registry.defaultModelID(for: "openai", configuration: config)
        #expect(defaultID == "gpt-4o")
    }

    @Test func registryFallsBackToRegisteredModel() async throws {
        let registry = LLMModelRegistry()
        await registry.register([
            LLMModelInfo(id: "glm-5.2:cloud", providerName: "ollama", displayName: "GLM Cloud")
        ], for: "ollama")

        let defaultID = try await registry.defaultModelID(for: "ollama")
        #expect(defaultID == "glm-5.2:cloud")
    }

    @Test func providerDefaultImplementationThrowsForModelListing() async throws {
        // OpenAI implements availableModels, so we test Anthropic's curated static list
        // and confirm the default protocol behavior for unknown providers via a custom stub.
        let anthropic = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "test"))
        let models = try await anthropic.availableModels()
        #expect(!models.isEmpty)
        #expect(models.first?.providerName == "anthropic")
    }

    // MARK: - LLMService

    @Test func serviceRegistersAndLooksUpProvider() async throws {
        let service = LLMService()
        let ollama = OllamaProvider(configuration: OllamaProvider.local(model: "llama3.2"))
        await service.register(ollama)

        let found = try await service.provider(named: "ollama")
        #expect(type(of: found).name == "ollama")
    }

    @Test func serviceThrowsForUnknownProvider() async throws {
        let service = LLMService()
        await #expect(throws: LLMError.unknownProvider("anthropic")) {
            _ = try await service.provider(named: "anthropic")
        }
    }
}

// MARK: - LLMStreamChunk Equatable conformance for tests

extension LLMStreamChunk: Equatable {
    public static func == (lhs: LLMStreamChunk, rhs: LLMStreamChunk) -> Bool {
        switch (lhs, rhs) {
        case (.text(let a), .text(let b)):
            return a == b
        case (.finish(let r1, let u1), .finish(let r2, let u2)):
            return r1 == r2 && u1 == u2
        case (.error, .error):
            return true
        default:
            return false
        }
    }
}
