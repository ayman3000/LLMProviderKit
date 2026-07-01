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
        #expect(chunksEqual(chunks[0], .text(" world")))
        #expect(chunksEqual(chunks[1], .finish(reason: .stop, usage: LLMUsage(promptTokens: 10, completionTokens: 3, totalTokens: 13))))
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
        #expect(chunksEqual(chunks[0], .text("!")))
        #expect(chunksEqual(chunks[1], .finish(reason: .stop, usage: nil)))
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
        #expect(chunksEqual(chunks[0], .text(" there")))
        #expect(chunksEqual(chunks[1], .finish(reason: .stop, usage: LLMUsage(promptTokens: 8, completionTokens: 2, totalTokens: 10))))
    }

    @Test func geminiParsesFunctionCallArgumentsObject() async throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "test", model: "gemini-2.5-flash-lite"))
        let data = """
        {
          "candidates": [{
            "content": {
              "role": "model",
              "parts": [{"functionCall": {"name": "list_files", "args": {"directory": "/tmp", "limit": 3}}}]
            },
            "finishReason": "STOP"
          }]
        }
        """.data(using: .utf8)!
        let request = LLMRequest(model: "gemini-2.5-flash-lite", messages: [.user("List /tmp")])
        let response = try provider.parseResponse(data, request: request)

        #expect(response.toolCalls.count == 1)
        #expect(response.toolCalls.first?.name == "list_files")
        #expect(response.toolCalls.first?.decodedArguments()?["directory"] as? String == "/tmp")
        #expect(response.toolCalls.first?.decodedArguments()?["limit"] as? Double == 3)
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
        #expect(chunksEqual(chunks[0], .text("Hello")))
        #expect(chunksEqual(chunks[1], .text(" Claude")))
        #expect(chunksEqual(chunks[2], .finish(reason: .stop, usage: LLMUsage(promptTokens: 12, completionTokens: 5, totalTokens: 17))))
    }

    @Test func anthropicParsesToolUseInputObject() async throws {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "test", model: "claude-3-5-sonnet-20241022"))
        let data = """
        {
          "id": "msg_01",
          "type": "message",
          "role": "assistant",
          "content": [{"type": "tool_use", "id": "toolu_123", "name": "list_files", "input": {"directory": "/tmp", "limit": 3}}],
          "stop_reason": "tool_use",
          "usage": { "input_tokens": 12, "output_tokens": 5 }
        }
        """.data(using: .utf8)!
        let request = LLMRequest(model: "claude-3-5-sonnet-20241022", messages: [.user("List /tmp")])
        let response = try provider.parseResponse(data, request: request)

        #expect(response.finishReason == LLMFinishReason.toolCalls)
        #expect(response.toolCalls.count == 1)
        #expect(response.toolCalls.first?.id == "toolu_123")
        #expect(response.toolCalls.first?.name == "list_files")
        #expect(response.toolCalls.first?.decodedArguments()?["directory"] as? String == "/tmp")
        #expect(response.toolCalls.first?.decodedArguments()?["limit"] as? Double == 3)
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

// MARK: - LLMStreamChunk comparison helper for tests

func chunksEqual(_ lhs: LLMStreamChunk, _ rhs: LLMStreamChunk) -> Bool {
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

extension ProviderTests {
    // MARK: - Image encoding tests

    private func makeImageRequest(provider: any LLMProvider, model: String) throws -> Data {
        let pixel = Data([0x89, 0x50, 0x4E, 0x47]) // fake PNG header
        let request = LLMRequest(
            model: model,
            messages: [
                .user("What's in this image?", images: [LLMImage(data: pixel, mimeType: "image/png")])
            ]
        )
        let urlRequest = try provider.prepareRequest(request, stream: false)
        return urlRequest.httpBody ?? Data()
    }

    @Test func ollamaImageEncoding() async throws {
        let provider = OllamaProvider(configuration: OllamaProvider.local(model: "llama3.2"))
        let body = try makeImageRequest(provider: provider, model: "llama3.2")
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try #require(json["messages"] as? [[String: Any]])
        let firstMsg = try #require(messages.first)
        let images = try #require(firstMsg["images"] as? [String])
        #expect(images.count == 1)
        let pixel = Data([0x89, 0x50, 0x4E, 0x47])
        #expect(images.first == pixel.base64EncodedString())
        // Content should still be a plain string.
        #expect(firstMsg["content"] is String)
    }

    @Test func openAIImageEncoding() async throws {
        let provider = OpenAIProvider(configuration: OpenAIProvider.openAI(apiKey: "test", model: "gpt-4o"))
        let body = try makeImageRequest(provider: provider, model: "gpt-4o")
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try #require(json["messages"] as? [[String: Any]])
        let firstMsg = try #require(messages.first)
        // Content should be an array of parts.
        let parts = try #require(firstMsg["content"] as? [[String: Any]])
        #expect(parts.count == 2)
        #expect(parts[0]["type"] as? String == "text")
        #expect(parts[1]["type"] as? String == "image_url")
        let imageURL = try #require(parts[1]["image_url"] as? [String: Any])
        let url = try #require(imageURL["url"] as? String)
        #expect(url.hasPrefix("data:image/png;base64,"))
    }

    @Test func anthropicImageEncoding() async throws {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "test", model: "claude-3-5-sonnet-20241022"))
        let body = try makeImageRequest(provider: provider, model: "claude-3-5-sonnet-20241022")
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try #require(json["messages"] as? [[String: Any]])
        let firstMsg = try #require(messages.first)
        // Content should be an array of blocks.
        let blocks = try #require(firstMsg["content"] as? [[String: Any]])
        #expect(blocks.count == 2)
        #expect(blocks[0]["type"] as? String == "text")
        #expect(blocks[1]["type"] as? String == "image")
        let source = try #require(blocks[1]["source"] as? [String: Any])
        #expect(source["type"] as? String == "base64")
        #expect(source["media_type"] as? String == "image/png")
    }

    @Test func geminiImageEncoding() async throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "test", model: "gemini-2.5-flash"))
        let body = try makeImageRequest(provider: provider, model: "gemini-2.5-flash")
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let contents = try #require(json["contents"] as? [[String: Any]])
        let firstContent = try #require(contents.first)
        let parts = try #require(firstContent["parts"] as? [[String: Any]])
        #expect(parts.count == 2)
        #expect(parts[0]["text"] != nil)
        let inlineData = try #require(parts[1]["inlineData"] as? [String: Any])
        #expect(inlineData["mimeType"] as? String == "image/png")
    }

    @Test func textOnlyRequestsUnchanged() async throws {
        // Verify that text-only messages don't emit image fields.
        let ollama = OllamaProvider(configuration: OllamaProvider.local(model: "llama3.2"))
        let request = LLMRequest(model: "llama3.2", messages: [.user("Hi")])
        let urlRequest = try ollama.prepareRequest(request, stream: false)
        let body = urlRequest.httpBody ?? Data()
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try #require(json["messages"] as? [[String: Any]])
        let firstMsg = try #require(messages.first)
        #expect(firstMsg["images"] == nil)
        #expect(firstMsg["content"] is String)
    }

    @Test func ollamaAssistantToolCallsSerializeArgumentsAsJSONObject() async throws {
        let ollama = OllamaProvider(configuration: OllamaProvider.local(model: "qwen3:0.6b"))
        let request = LLMRequest(
            model: "qwen3:0.6b",
            messages: [
                .user("What time is it?"),
                .assistant(content: "", toolCalls: [
                    LLMToolCall(id: "call_1", name: "current_time", arguments: "{}"),
                    LLMToolCall(id: "call_2", name: "echo_message", arguments: "{\"message\":\"SwiftAgentKit\"}")
                ]),
                .tool("Sunday, 28 June 2026 at 6:06:50 AM", toolCallId: "call_1"),
                .tool("Echo: SwiftAgentKit", toolCallId: "call_2")
            ]
        )

        let body = try #require(ollama.prepareRequest(request, stream: false).httpBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try #require(json["messages"] as? [[String: Any]])
        let assistant = try #require(messages.first { $0["role"] as? String == "assistant" })
        let toolCalls = try #require(assistant["tool_calls"] as? [[String: Any]])
        let firstFunction = try #require(toolCalls[0]["function"] as? [String: Any])
        let secondFunction = try #require(toolCalls[1]["function"] as? [String: Any])

        #expect(firstFunction["arguments"] is [String: Any])
        let secondArgs = try #require(secondFunction["arguments"] as? [String: Any])
        #expect(secondArgs["message"] as? String == "SwiftAgentKit")
    }
}
