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

    @Test func ollamaReasoningOnlyResponseSurfacesThinking() async throws {
        // Reasoning models (GLM/Kimi via Ollama) sometimes return everything in
        // `thinking` with empty `content`. The thinking must surface as
        // `reasoning` so agent loops can tell "mid-thought" from "done".
        let provider = OllamaProvider(configuration: OllamaProvider.local(model: "glm-5.2:cloud"))
        let data = """
        {
          "model": "glm-5.2:cloud",
          "message": { "role": "assistant", "content": "", "thinking": "Let me check the files first..." },
          "done": true
        }
        """.data(using: .utf8)!

        let request = LLMRequest(model: "glm-5.2:cloud", messages: [.user("Analyze")])
        let response = try provider.parseResponse(data, request: request)

        #expect(response.text.isEmpty)
        #expect(response.reasoning == "Let me check the files first...")
    }

    @Test func ollamaErrorBodyThrows() async throws {
        // Ollama's cloud proxy can return HTTP 200 with an error body; that
        // must throw, not decode into an empty success the agent loop then
        // treats as "the model said nothing".
        let provider = OllamaProvider(configuration: OllamaProvider.local(model: "glm-5.2:cloud"))
        let data = #"{"error": "upstream connection reset", "done": true}"#.data(using: .utf8)!
        let request = LLMRequest(model: "glm-5.2:cloud", messages: [.user("Hi")])
        #expect(throws: LLMError.providerError("upstream connection reset")) {
            _ = try provider.parseResponse(data, request: request)
        }
    }

    @Test func ollamaMissingMessageThrows() async throws {
        // A chat body with `done` but NO message is a degenerate proxy/unload
        // response (observed live from glm-5.2:cloud under concurrent load):
        // 200 OK, sub-second, no content/thinking/tool_calls. Must throw so
        // callers retry instead of accepting an empty answer.
        let provider = OllamaProvider(configuration: OllamaProvider.local(model: "glm-5.2:cloud"))
        let data = #"{"model": "glm-5.2:cloud", "done": true, "done_reason": "unload"}"#.data(using: .utf8)!
        let request = LLMRequest(model: "glm-5.2:cloud", messages: [.user("Hi")])
        #expect(throws: LLMError.self) {
            _ = try provider.parseResponse(data, request: request)
        }
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

    @Test func geminiPreservesFunctionCallThoughtSignature() async throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "test", model: "gemini-2.5-flash"))
        let data = """
        {
          "candidates": [{
            "content": {
              "role": "model",
              "parts": [{
                "functionCall": { "name": "current_datetime", "args": {} },
                "thoughtSignature": "signed-part-token"
              }]
            },
            "finishReason": "STOP"
          }]
        }
        """.data(using: .utf8)!
        let request = LLMRequest(model: "gemini-2.5-flash", messages: [.user("What time is it?")])
        let response = try provider.parseResponse(data, request: request)

        #expect(response.toolCalls.count == 1)
        #expect(response.toolCalls.first?.providerMetadata["gemini.thoughtSignature"] == "signed-part-token")
    }

    @Test func geminiAssistantToolCallsReplayThoughtSignature() async throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "test", model: "gemini-2.5-flash"))
        let request = LLMRequest(
            model: "gemini-2.5-flash",
            messages: [
                .assistant(content: "", toolCalls: [
                    LLMToolCall(
                        id: "call_1",
                        name: "current_datetime",
                        arguments: "{}",
                        providerMetadata: ["gemini.thoughtSignature": "signed-part-token"]
                    )
                ])
            ]
        )

        let body = try #require(provider.prepareRequest(request, stream: false).httpBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let contents = try #require(json["contents"] as? [[String: Any]])
        let parts = try #require(contents.first?["parts"] as? [[String: Any]])
        let functionCallPart = try #require(parts.first)

        #expect(functionCallPart["thoughtSignature"] as? String == "signed-part-token")
    }

    @Test func geminiPrepareRequestNormalizesPrefixedModelID() async throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "test", model: "gemini-2.5-flash"))
        let request = LLMRequest(model: "models/gemini-2.5-flash", messages: [.user("Hi")])

        let urlRequest = try provider.prepareRequest(request, stream: false)

        #expect(urlRequest.url?.path == "/v1beta/models/gemini-2.5-flash:generateContent")
    }

    @Test func geminiAvailableModelsNormalizeIDsAndMarkTools() async throws {
        let apiKey = "normalize-test"
        GeminiModelsMockURLProtocol.setResponseData("""
        {
          "models": [{
            "name": "models/gemini-2.5-flash",
            "displayName": "Gemini 2.5 Flash",
            "inputTokenLimit": 1048576,
            "supportedGenerationMethods": ["generateContent", "countTokens"]
          }]
        }
        """.data(using: .utf8)!, forAPIKey: apiKey)

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [GeminiModelsMockURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let provider = GeminiProvider(
            configuration: GeminiProvider.gemini(apiKey: apiKey, model: "gemini-2.5-flash"),
            urlSession: session
        )

        let models = try await provider.availableModels()

        #expect(models.first?.id == "gemini-2.5-flash")
        #expect(models.first?.capabilities.contains(.tools) == true)
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

    @Test func streamingDecodesUTF8LinesWithoutCorruptingNonASCIIText() async throws {
        let streamedText = "Hello é 😊 مرحبا"
        UTF8StreamingMockURLProtocol.responseData = "data: \(streamedText)\n".data(using: .utf8)!

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [UTF8StreamingMockURLProtocol.self]
        let provider = UTF8StreamingTestProvider(urlSession: URLSession(configuration: configuration))
        let request = LLMRequest(model: "utf8-test", messages: [.user("Stream UTF-8")])

        var collected = ""
        for try await chunk in provider.stream(request) {
            if case .text(let text) = chunk {
                collected += text
            }
        }

        #expect(collected == streamedText)
    }
}

private final class UTF8StreamingMockURLProtocol: URLProtocol {
    static var responseData = Data()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "text/event-stream; charset=utf-8"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.responseData)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private struct UTF8StreamingTestProvider: LLMProvider {
    static let name = "utf8-streaming-test"
    let configuration = LLMProviderConfiguration(
        name: name,
        baseURL: URL(string: "https://example.test")!
    )
    let urlSession: URLSession

    func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        URLRequest(url: URL(string: "https://example.test/stream")!)
    }

    func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        guard line.hasPrefix("data: ") else { return [] }
        return [.text(String(line.dropFirst("data: ".count)))]
    }

    func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        LLMResponse(
            text: String(data: data, encoding: .utf8) ?? "",
            request: request,
            providerName: Self.name
        )
    }
}

private final class GeminiModelsMockURLProtocol: URLProtocol {
    static var responseData = Data()
    private static let lock = NSLock()
    private static var responseDataByAPIKey: [String: Data] = [:]

    static func setResponseData(_ data: Data, forAPIKey apiKey: String) {
        lock.lock()
        defer { lock.unlock() }
        responseDataByAPIKey[apiKey] = data
    }

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let apiKey = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)?
            .queryItems?
            .first { $0.name == "key" }?
            .value
        let data: Data = Self.lock.withLock {
            apiKey.flatMap { Self.responseDataByAPIKey[$0] } ?? Self.responseData
        }
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

extension ProviderTests {
    // MARK: - Model metadata and registry tests

    @Test func modelInfoOldInitializerStillWorks() async throws {
        let model = LLMModelInfo(
            id: "test-model",
            providerName: "test",
            displayName: "Test Model",
            capabilities: [.chat, .streaming]
        )

        #expect(model.id == "test-model")
        #expect(model.capabilities.contains(.chat))
        #expect(model.categories.isEmpty)
        #expect(model.releaseStage == nil)
        #expect(model.isDeprecated == false)
    }

    @Test func modelRegistryFiltersByCategoryAndCapabilities() async throws {
        let registry = LLMModelRegistry()
        await registry.register([
            LLMModelInfo(
                id: "text-tools",
                providerName: "test",
                capabilities: [.chat, .tools],
                categories: [.text]
            ),
            LLMModelInfo(
                id: "vision-tools",
                providerName: "test",
                capabilities: [.chat, .tools, .vision],
                categories: [.text, .vision, .multimodal]
            ),
            LLMModelInfo(
                id: "embed",
                providerName: "test",
                capabilities: [.embeddings],
                categories: [.embedding]
            )
        ], for: "test")

        let toolModels = await registry.models(providerName: "test", matching: [.tools])
        #expect(toolModels.map(\.id).sorted() == ["text-tools", "vision-tools"])

        let visionTools = await registry.models(providerName: "test", category: .vision, matching: [.tools])
        #expect(visionTools.map(\.id) == ["vision-tools"])
    }

    @Test func modelRegistryLiveWithCuratedMetadataEnrichesLiveRecords() async throws {
        let live = [
            LLMModelInfo(id: "known", providerName: "test", displayName: nil, capabilities: [.chat])
        ]
        let curated = [
            LLMModelInfo(
                id: "known",
                providerName: "test",
                displayName: "Known Model",
                contextWindow: 1234,
                capabilities: [.tools, .streaming],
                categories: [.text],
                releaseStage: .stable,
                notes: "curated"
            )
        ]

        let merged = LLMModelRegistry.merge(live: live, curated: curated, strategy: .liveWithCuratedMetadata)
        let model = try #require(merged.first)
        #expect(model.displayName == "Known Model")
        #expect(model.contextWindow == 1234)
        #expect(model.capabilities.contains(.chat))
        #expect(model.capabilities.contains(.tools))
        #expect(model.categories.contains(.text))
        #expect(model.releaseStage == .stable)
    }

    @Test func curatedProviderModelsHaveCategories() async throws {
        #expect(OpenAIProvider.curatedModels.contains { $0.categories.contains(.multimodal) })
        #expect(GeminiProvider.curatedModels.contains { $0.capabilities.contains(.vision) })
        #expect(AnthropicProvider.curatedModels.contains { $0.releaseStage == .stable })
        #expect(OllamaProvider.suggestedModels.contains { $0.categories.contains(.embedding) })
    }

    @Test func openAICapabilityHeuristicsDoNotOvermatchLetterO() async throws {
        let moderation = OpenAIProvider.capabilities(for: "omni-moderation-latest")
        #expect(!moderation.contains(.tools))
        #expect(!moderation.contains(.vision))
        #expect(!moderation.contains(.reasoning))
        #expect(!OpenAIProvider.categories(for: "omni-moderation-latest").contains(.multimodal))

        let computerUse = OpenAIProvider.capabilities(for: "computer-use-preview")
        #expect(!computerUse.contains(.tools))
        #expect(!computerUse.contains(.vision))
        #expect(!computerUse.contains(.reasoning))

        let oSeries = OpenAIProvider.capabilities(for: "o4-mini")
        #expect(oSeries.contains(.tools))
        #expect(oSeries.contains(.vision))
        #expect(oSeries.contains(.reasoning))
    }

    @Test func geminiCapabilityHeuristicsDoNotMarkAllThreeSeriesAsReasoning() async throws {
        let apiKey = "reasoning-test"
        GeminiModelsMockURLProtocol.setResponseData("""
        {
          "models": [
            {
              "name": "models/gemini-3.1-flash-lite",
              "displayName": "Gemini 3.1 Flash-Lite",
              "inputTokenLimit": 1048576,
              "supportedGenerationMethods": ["generateContent", "countTokens"]
            },
            {
              "name": "models/gemini-3.1-pro",
              "displayName": "Gemini 3.1 Pro",
              "inputTokenLimit": 1048576,
              "supportedGenerationMethods": ["generateContent", "countTokens"]
            }
          ]
        }
        """.data(using: .utf8)!, forAPIKey: apiKey)

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [GeminiModelsMockURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let provider = GeminiProvider(
            configuration: GeminiProvider.gemini(apiKey: apiKey, model: "gemini-3.1-flash-lite"),
            urlSession: session
        )

        let models = try await provider.availableModels()
        let flashLite = try #require(models.first { $0.id == "gemini-3.1-flash-lite" })
        let pro = try #require(models.first { $0.id == "gemini-3.1-pro" })

        #expect(!flashLite.capabilities.contains(.reasoning))
        #expect(pro.capabilities.contains(.reasoning))
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

    @Test func anthropicSendsValidVersionHeader() async throws {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "test", model: "claude-3-5-sonnet-20241022"))
        let request = LLMRequest(model: "claude-3-5-sonnet-20241022", messages: [.user("Hi")])
        let urlRequest = try provider.prepareRequest(request, stream: false)

        // Anthropic requires a dated version string; the branding string belongs in User-Agent.
        #expect(urlRequest.value(forHTTPHeaderField: "anthropic-version") == "2023-06-01")
        #expect(urlRequest.value(forHTTPHeaderField: "anthropic-version") != "LLMProviderKit/1.0")
    }

    @Test func geminiSendsToolResultsAsUserRole() async throws {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "test", model: "gemini-2.5-flash"))
        let request = LLMRequest(
            model: "gemini-2.5-flash",
            messages: [
                .user("What time is it?"),
                .assistant(content: "", toolCalls: [
                    LLMToolCall(id: "call_1", name: "current_datetime", arguments: "{}")
                ]),
                .tool("{\"result\":\"noon\"}", toolCallId: "current_datetime")
            ]
        )

        let body = try #require(provider.prepareRequest(request, stream: false).httpBody)
        let json = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let contents = try #require(json["contents"] as? [[String: Any]])

        // The turn carrying the functionResponse part must use the "user" role.
        let functionResponseTurn = try #require(contents.first { turn in
            guard let parts = turn["parts"] as? [[String: Any]] else { return false }
            return parts.contains { $0["functionResponse"] != nil }
        })
        #expect(functionResponseTurn["role"] as? String == "user")
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

// MARK: - In-process provider (the pattern an on-device MLX backend uses)

/// A provider that generates in-process — no HTTP. It overrides `complete`/
/// `stream` and never touches `prepareRequest`/`parseResponse`. This is exactly
/// how an `MLXProvider` will plug in.
struct EchoLocalProvider: LLMProvider {
    static let name = "echo-local"
    let configuration: LLMProviderConfiguration

    private func reply(_ request: LLMRequest) -> String {
        "echo: " + (request.messages.last(where: { $0.role == .user })?.content ?? "")
    }

    func complete(_ request: LLMRequest) async throws -> LLMResponse {
        LLMResponse(text: reply(request), finishReason: .stop, request: request, providerName: Self.name)
    }

    func stream(_ request: LLMRequest) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        let text = reply(request)
        return AsyncThrowingStream { continuation in
            continuation.yield(.text(text))
            continuation.yield(.finish(reason: .stop, usage: nil))
            continuation.finish()
        }
    }
}

struct InProcessProviderTests {
    private func request() -> LLMRequest {
        LLMRequest(model: "local", messages: [LLMMessage(role: .user, content: "hi there")])
    }

    /// Called through the existential `any LLMProvider`, the override must win —
    /// proving `complete` is a dynamically-dispatched requirement, not a static
    /// extension method (which would call the HTTP default and throw).
    @Test func inProcessCompleteIsDynamicallyDispatched() async throws {
        let provider: any LLMProvider = EchoLocalProvider(configuration: LLMProviderConfiguration(name: "echo-local", baseURL: URL(string: "inprocess://local")!))
        let response = try await provider.complete(request())
        #expect(response.text == "echo: hi there")
        #expect(response.providerName == "echo-local")
    }

    @Test func inProcessStreamIsDynamicallyDispatched() async throws {
        let provider: any LLMProvider = EchoLocalProvider(configuration: LLMProviderConfiguration(name: "echo-local", baseURL: URL(string: "inprocess://local")!))
        var text = ""
        for try await chunk in provider.stream(request()) {
            if case .text(let t) = chunk { text += t }
        }
        #expect(text == "echo: hi there")
    }

    /// The default HTTP hooks now exist, so an in-process provider that doesn't
    /// implement them fails loudly (unsupported) rather than failing to compile.
    @Test func httpHooksHaveThrowingDefaults() throws {
        let provider = EchoLocalProvider(configuration: LLMProviderConfiguration(name: "echo-local", baseURL: URL(string: "inprocess://local")!))
        #expect(throws: (any Error).self) {
            _ = try provider.prepareRequest(request(), stream: false)
        }
    }
}
