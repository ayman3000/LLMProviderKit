import Foundation
import LLMProviderKit
import LLMProviderKitOpenAI

/// OpenRouter (https://openrouter.ai): one API key in front of hundreds of
/// models from every major vendor, served over the OpenAI chat-completions
/// wire format. This provider reuses `OpenAIProvider` for requests, streaming
/// and response parsing (including the `reasoning` delta OpenRouter streams
/// for thinking models) and adds what OpenRouter does differently:
///
/// - Attribution headers (`HTTP-Referer`, `X-Title`) so the calling app is
///   credited on openrouter.ai — optional, recommended, never required.
/// - A rich, public `GET /models` catalog: context length, input modalities,
///   supported parameters (tools / reasoning) and per-token list prices. Those
///   become real `LLMModelInfo` capabilities, `contextWindow` and `pricing`,
///   so downstream apps never guess a model's abilities from its name.
public struct OpenRouterProvider: LLMProvider {
    public static let name: String = "openrouter"
    public static let defaultBaseURL = URL(string: "https://openrouter.ai/api/v1")!

    public let configuration: LLMProviderConfiguration
    /// Shown on openrouter.ai's app leaderboard. Optional.
    public let appName: String?
    /// The app's site, sent as `HTTP-Referer`. Optional.
    public let appURL: URL?

    private let inner: OpenAIProvider

    public init(configuration: LLMProviderConfiguration, appName: String? = nil, appURL: URL? = nil) {
        self.configuration = configuration
        self.appName = appName
        self.appURL = appURL
        self.inner = OpenAIProvider(configuration: configuration)
    }

    /// Convenience configuration pointed at the public OpenRouter endpoint.
    public static func openRouter(apiKey: String, model: String? = nil) -> LLMProviderConfiguration {
        LLMProviderConfiguration(name: name, baseURL: defaultBaseURL, apiKey: apiKey, defaultModel: model)
    }

    // MARK: - Wire format (delegated to OpenAI)

    public func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        var urlRequest = try inner.prepareRequest(request, stream: stream)
        applyAttribution(to: &urlRequest)
        return urlRequest
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        try inner.parseStreamLine(line, request: request)
    }

    public func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        try inner.parseResponse(data, request: request)
    }

    public func resolvedModel(for request: LLMRequest) async throws -> String {
        try await inner.resolvedModel(for: request)
    }

    // MARK: - Catalog

    /// `GET /models` is public (no key needed) and carries the metadata that
    /// matters for an agent host. Mapping rules live in `OpenRouterCatalog`
    /// so they are unit-testable from a JSON fixture.
    public func availableModels() async throws -> [LLMModelInfo] {
        let url = configuration.baseURL.appendingPathComponent("models")
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "GET"
        if let apiKey = configuration.apiKey, !apiKey.isEmpty {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        applyAttribution(to: &urlRequest)
        let (data, response) = try await urlSession.data(for: urlRequest)
        try Self.verifyHTTPResponse(response, data: data)
        return try OpenRouterCatalog.models(from: data)
    }

    // MARK: - Helpers

    func applyAttribution(to urlRequest: inout URLRequest) {
        if let appURL { urlRequest.setValue(appURL.absoluteString, forHTTPHeaderField: "HTTP-Referer") }
        if let appName { urlRequest.setValue(appName, forHTTPHeaderField: "X-Title") }
    }
}

/// Pure mapping of OpenRouter's `/models` payload into `LLMModelInfo`.
public enum OpenRouterCatalog {

    public static func models(from data: Data) throws -> [LLMModelInfo] {
        let decoded = try JSONDecoder().decode(ModelsResponse.self, from: data)
        return decoded.data.map(info(for:))
    }

    static func info(for m: Model) -> LLMModelInfo {
        let inputs = Set(m.architecture?.inputModalities ?? [])
        let outputs = Set(m.architecture?.outputModalities ?? ["text"])
        let params = Set(m.supportedParameters ?? [])

        var caps: Set<LLMModelCapability> = [.streaming]
        if outputs.contains("text") { caps.formUnion([.chat, .textGeneration]) }
        if inputs.contains("image") { caps.formUnion([.vision, .imageInput]) }
        if inputs.contains("audio") { caps.insert(.audioInput) }
        if outputs.contains("image") { caps.insert(.imageGeneration) }
        if outputs.contains("audio") { caps.insert(.audioGeneration) }
        if params.contains("tools") { caps.insert(.tools) }
        if params.contains("reasoning") || params.contains("include_reasoning") { caps.insert(.reasoning) }
        if params.contains("response_format") || params.contains("structured_outputs") { caps.insert(.structuredOutput) }

        var categories: Set<LLMModelCategory> = []
        if outputs.contains("text") { categories.insert(.text) }
        if inputs.contains("image") { categories.insert(.vision) }
        if outputs.contains("image") { categories.insert(.image) }
        if inputs.contains("audio") || outputs.contains("audio") { categories.insert(.audio) }
        if inputs.count > 1 { categories.insert(.multimodal) }

        // OpenRouter prices are USD per token as decimal strings; -1 means
        // "varies" (per-request). Only report a price when both sides parse.
        var pricing: LLMModelPricing? = nil
        if let p = m.pricing, let inp = Double(p.prompt ?? ""), let out = Double(p.completion ?? ""),
           inp >= 0, out >= 0 {
            pricing = LLMModelPricing(inputPerMillionTokens: inp * 1_000_000,
                                      outputPerMillionTokens: out * 1_000_000)
        }

        // Context: the model's own limit, else the top provider's.
        let context = m.contextLength ?? m.topProvider?.contextLength

        var stage: LLMModelReleaseStage? = nil
        if let exp = m.expirationDate, !exp.isEmpty { stage = .deprecated }

        return LLMModelInfo(
            id: m.id,
            providerName: OpenRouterProvider.name,
            displayName: m.name,
            contextWindow: context,
            capabilities: caps,
            categories: categories,
            releaseStage: stage,
            isDeprecated: stage == .deprecated,
            notes: m.description.map { String($0.prefix(240)) },
            pricing: pricing
        )
    }

    /// The vendor segment of an OpenRouter id (`anthropic/claude-sonnet-4` →
    /// `anthropic`), for grouping long catalogs in a picker.
    public static func vendor(of modelID: String) -> String {
        let raw = modelID.split(separator: "/", maxSplits: 1).first.map(String.init) ?? modelID
        // `~anthropic/claude-latest` is OpenRouter's alias form (always the newest
        // model in a family) — same vendor, so the tilde is dropped.
        return raw.hasPrefix("~") ? String(raw.dropFirst()) : raw
    }

    // MARK: - Wire types

    struct ModelsResponse: Decodable { let data: [Model] }

    struct Model: Decodable {
        let id: String
        let name: String?
        let description: String?
        let contextLength: Int?
        let architecture: Architecture?
        let pricing: Pricing?
        let topProvider: TopProvider?
        let supportedParameters: [String]?
        let expirationDate: String?

        enum CodingKeys: String, CodingKey {
            case id, name, description, architecture, pricing
            case contextLength = "context_length"
            case topProvider = "top_provider"
            case supportedParameters = "supported_parameters"
            case expirationDate = "expiration_date"
        }
    }
    struct Architecture: Decodable {
        let inputModalities: [String]?
        let outputModalities: [String]?
        enum CodingKeys: String, CodingKey {
            case inputModalities = "input_modalities"
            case outputModalities = "output_modalities"
        }
    }
    struct Pricing: Decodable { let prompt: String?; let completion: String? }
    struct TopProvider: Decodable {
        let contextLength: Int?
        let maxCompletionTokens: Int?
        enum CodingKeys: String, CodingKey {
            case contextLength = "context_length"
            case maxCompletionTokens = "max_completion_tokens"
        }
    }
}
