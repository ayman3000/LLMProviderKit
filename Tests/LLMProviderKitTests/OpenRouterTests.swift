import Testing
import Foundation
import LLMProviderKit
@testable import LLMProviderKitOpenRouter

struct OpenRouterTests {
    static let fixture = """
    {"data":[
      {"id":"anthropic/claude-sonnet-4","name":"Anthropic: Claude Sonnet 4","description":"Best coding model.",
       "context_length":1000000,
       "architecture":{"input_modalities":["image","text","file"],"output_modalities":["text"]},
       "pricing":{"prompt":"0.000003","completion":"0.000015"},
       "top_provider":{"context_length":200000,"max_completion_tokens":64000},
       "supported_parameters":["max_tokens","reasoning","tools","tool_choice"]},
      {"id":"meta-llama/llama-3.3-70b-instruct:free","name":"Meta: Llama 3.3 70B (free)",
       "context_length":131072,
       "architecture":{"input_modalities":["text"],"output_modalities":["text"]},
       "pricing":{"prompt":"0","completion":"0"},
       "supported_parameters":["max_tokens","temperature"]},
      {"id":"openai/gpt-4o-audio-preview","name":"OpenAI: GPT-4o Audio",
       "architecture":{"input_modalities":["text","audio"],"output_modalities":["text","audio"]},
       "pricing":{"prompt":"-1","completion":"-1"},
       "top_provider":{"context_length":128000},
       "supported_parameters":["tools"],"expiration_date":"2026-12-01"}
    ]}
    """.data(using: .utf8)!

    @Test func mapsCapabilitiesContextAndPricing() throws {
        let models = try OpenRouterCatalog.models(from: Self.fixture)
        #expect(models.count == 3)
        let sonnet = models[0]
        #expect(sonnet.id == "anthropic/claude-sonnet-4")
        #expect(sonnet.displayName == "Anthropic: Claude Sonnet 4")
        #expect(sonnet.contextWindow == 1_000_000)
        #expect(sonnet.capabilities.isSuperset(of: [.chat, .vision, .imageInput, .tools, .reasoning, .streaming]))
        #expect(sonnet.categories.contains(.vision))
        #expect(sonnet.pricing?.inputPerMillionTokens == 3.0)
        #expect(sonnet.pricing?.outputPerMillionTokens == 15.0)
        #expect(sonnet.providerName == "openrouter")
    }

    @Test func freeModelIsFreeAndNotVisionNotTools() throws {
        let llama = try OpenRouterCatalog.models(from: Self.fixture)[1]
        #expect(llama.pricing?.isFree == true)
        #expect(!llama.capabilities.contains(.vision))
        #expect(!llama.capabilities.contains(.tools))
        #expect(llama.contextWindow == 131_072)
    }

    @Test func variablePricingIsOmittedAndExpiringIsDeprecated() throws {
        let audio = try OpenRouterCatalog.models(from: Self.fixture)[2]
        #expect(audio.pricing == nil)
        #expect(audio.contextWindow == 128_000)   // falls back to top_provider
        #expect(audio.capabilities.contains(.audioInput))
        #expect(audio.isDeprecated)
    }

    @Test func vendorPrefix() {
        #expect(OpenRouterCatalog.vendor(of: "google/gemini-2.5-pro") == "google")
        #expect(OpenRouterCatalog.vendor(of: "noslash") == "noslash")
    }

    @Test func attributionHeadersAndBearer() throws {
        let provider = OpenRouterProvider(
            configuration: OpenRouterProvider.openRouter(apiKey: "sk-or-test", model: "anthropic/claude-sonnet-4"),
            appName: "Naseem", appURL: URL(string: "https://example.com/naseem"))
        let req = LLMRequest(model: "anthropic/claude-sonnet-4", messages: [.user("Hi")])
        let url = try provider.prepareRequest(req, stream: true)
        #expect(url.url?.absoluteString == "https://openrouter.ai/api/v1/chat/completions")
        #expect(url.value(forHTTPHeaderField: "Authorization") == "Bearer sk-or-test")
        #expect(url.value(forHTTPHeaderField: "X-Title") == "Naseem")
        #expect(url.value(forHTTPHeaderField: "HTTP-Referer") == "https://example.com/naseem")
    }

    @Test func streamingReasoningDeltaPassesThrough() throws {
        let provider = OpenRouterProvider(configuration: OpenRouterProvider.openRouter(apiKey: "k"))
        let req = LLMRequest(model: "x", messages: [.user("Hi")])
        let chunks = try provider.parseStreamLine(
            #"data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"x","choices":[{"index":0,"delta":{"reasoning":"hmm"},"finish_reason":null}]}"#,
            request: req)
        #expect(chunks.count == 1)
        if case .reasoning(let r) = chunks[0] { #expect(r == "hmm") } else { Issue.record("expected reasoning chunk") }
    }
}

/// Live decode of the real public catalog (no key). Gated: set OPENROUTER_LIVE=1.
struct OpenRouterLiveTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["OPENROUTER_LIVE"] == "1"))
    func realCatalogDecodesWithRichMetadata() async throws {
        let provider = OpenRouterProvider(configuration: OpenRouterProvider.openRouter(apiKey: ""))
        let models = try await provider.availableModels()
        #expect(models.count > 100)
        #expect(models.contains { $0.capabilities.contains(.vision) })
        #expect(models.contains { $0.capabilities.contains(.tools) })
        #expect(models.contains { $0.pricing?.isFree == true })
        #expect(models.contains { ($0.contextWindow ?? 0) >= 100_000 })
    }
}
