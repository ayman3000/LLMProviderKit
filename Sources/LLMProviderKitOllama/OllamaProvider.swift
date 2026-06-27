import Foundation
import LLMProviderKit

/// Provider for local [Ollama](https://ollama.com) servers.
///
/// Ollama’s chat API mirrors OpenAI’s shape but uses `model`, `messages`,
/// `stream`, and `options`.
public struct OllamaProvider: LLMProvider {
    public static let name: String = "ollama"

    public let configuration: LLMProviderConfiguration

    public init(configuration: LLMProviderConfiguration) {
        self.configuration = configuration
    }

    public func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        let url = configuration.baseURL
            .appendingPathComponent("api")
            .appendingPathComponent("chat")

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = OllamaChatRequest(
            model: request.model,
            messages: request.messages.map { msg in
                OllamaMessage(
                    role: msg.role,
                    content: msg.content,
                    images: msg.images.isEmpty ? nil : msg.images.map(\.base64)
                )
            },
            stream: stream,
            options: OllamaOptions(
                temperature: request.temperature,
                topP: request.topP,
                numPredict: request.maxTokens
            )
        )

        urlRequest.httpBody = try JSONEncoder().encode(body)
        return urlRequest
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        // Ollama streams full JSON objects, one per line.
        guard let data = line.data(using: .utf8) else {
            throw LLMError.streamingError("Invalid UTF-8 in stream line")
        }

        let decoded = try JSONDecoder().decode(OllamaChatResponse.self, from: data)
        var chunks: [LLMStreamChunk] = []

        if let text = decoded.message?.content, !text.isEmpty {
            chunks.append(.text(text))
        }

        if decoded.done {
            let usage = LLMUsage(
                promptTokens: decoded.promptEvalCount,
                completionTokens: decoded.evalCount,
                totalTokens: (decoded.promptEvalCount ?? 0) + (decoded.evalCount ?? 0)
            )
            chunks.append(.finish(reason: .stop, usage: usage))
        }

        return chunks
    }

    public func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        let decoded = try JSONDecoder().decode(OllamaChatResponse.self, from: data)
        let text = decoded.message?.content ?? ""
        let usage = LLMUsage(
            promptTokens: decoded.promptEvalCount,
            completionTokens: decoded.evalCount,
            totalTokens: (decoded.promptEvalCount ?? 0) + (decoded.evalCount ?? 0)
        )
        return LLMResponse(
            text: text,
            finishReason: .stop,
            usage: usage,
            request: request,
            providerName: Self.name,
            rawData: data
        )
    }

    public func availableModels() async throws -> [LLMModelInfo] {
        let url = configuration.baseURL
            .appendingPathComponent("api")
            .appendingPathComponent("tags")

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "GET"

        let (data, response) = try await urlSession.data(for: urlRequest)
        try Self.verifyHTTPResponse(response, data: data)

        let decoded = try JSONDecoder().decode(OllamaTagResponse.self, from: data)
        return decoded.models.map { model in
            LLMModelInfo(
                id: model.name,
                providerName: Self.name,
                displayName: model.name,
                contextWindow: model.details?.contextLength,
                capabilities: Self.capabilities(for: model)
            )
        }
    }

    private static func capabilities(for model: OllamaTagResponse.Model) -> Set<LLMModelCapability> {
        var caps: Set<LLMModelCapability> = [.chat, .streaming]
        if let capabilities = model.capabilities {
            for capability in capabilities {
                switch capability {
                case "vision": caps.insert(.vision)
                case "tools": caps.insert(.tools)
                case "completion": break // already covered by chat
                case "thinking": caps.insert(.reasoning)
                default: break
                }
            }
        }
        return caps
    }
}

// MARK: - Ollama chat API types

private struct OllamaChatRequest: Encodable {
    let model: String
    let messages: [OllamaMessage]
    let stream: Bool
    let options: OllamaOptions?
}

private struct OllamaMessage: Encodable {
    let role: LLMMessageRole
    let content: String
    let images: [String]?

    init(role: LLMMessageRole, content: String, images: [String]? = nil) {
        self.role = role
        self.content = content
        self.images = images
    }

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case images
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        // Only emit images when non-nil (keeps text-only requests unchanged).
        if let images {
            try container.encode(images, forKey: .images)
        }
    }
}

private struct OllamaOptions: Encodable {
    let temperature: Double?
    let topP: Double?
    let numPredict: Int?

    enum CodingKeys: String, CodingKey {
        case temperature
        case topP = "top_p"
        case numPredict = "num_predict"
    }
}

private struct OllamaChatResponse: Decodable {
    struct Message: Decodable {
        let role: String?
        let content: String?
    }

    let model: String?
    let createdAt: String?
    let message: Message?
    let done: Bool
    let totalDuration: Int64?
    let loadDuration: Int64?
    let promptEvalCount: Int?
    let promptEvalDuration: Int64?
    let evalCount: Int?
    let evalDuration: Int64?

    enum CodingKeys: String, CodingKey {
        case model
        case createdAt = "created_at"
        case message
        case done
        case totalDuration = "total_duration"
        case loadDuration = "load_duration"
        case promptEvalCount = "prompt_eval_count"
        case promptEvalDuration = "prompt_eval_duration"
        case evalCount = "eval_count"
        case evalDuration = "eval_duration"
    }
}

// MARK: - Ollama model-tag API types

private struct OllamaTagResponse: Decodable {
    struct Model: Decodable {
        let name: String
        let details: Details?
        let capabilities: [String]?

        struct Details: Decodable {
            let contextLength: Int?

            enum CodingKeys: String, CodingKey {
                case contextLength = "context_length"
            }
        }
    }

    let models: [Model]
}

// MARK: - Model constants

/// Well-known Ollama model names. These are convenience constants only;
/// pass any string to `LLMRequest.model` for models not listed here.
public enum OllamaModel {
    public static let llama3_2 = "llama3.2"
    public static let llama3_1 = "llama3.1"
    public static let gemma2 = "gemma2"
    public static let mistral = "mistral"
    public static let qwen2_5 = "qwen2.5"
    public static let phi4 = "phi4"
    public static let nomicEmbedText = "nomic-embed-text"
}

// MARK: - Configuration presets

extension OllamaProvider {
    /// Convenience configuration for a local Ollama server.
    ///
    /// - Parameters:
    ///   - model: Default model to use when none is specified in the request.
    ///            Pass `nil` to let the provider pick the first available model
    ///            from the server at request time.
    ///   - baseURL: Ollama server URL. Defaults to `http://localhost:11434`.
    public static func local(model: String? = nil, baseURL: URL = URL(string: "http://localhost:11434")!) -> LLMProviderConfiguration {
        LLMProviderConfiguration(
            name: name,
            baseURL: baseURL,
            defaultModel: model
        )
    }
}

// MARK: - Default model resolution

extension OllamaProvider {
    /// Resolve the model ID to use for a request.
    ///
    /// Resolution order:
    /// 1. Model explicitly set in the `LLMRequest`.
    /// 2. Model configured as `defaultModel` in `LLMProviderConfiguration`.
    /// 3. First model returned by Ollama's local `/api/tags` endpoint.
    public func resolvedModel(for request: LLMRequest) async throws -> String {
        if !request.model.isEmpty { return request.model }
        if let defaultModel = configuration.defaultModel, !defaultModel.isEmpty { return defaultModel }
        let models = try await availableModels()
        guard let first = models.first else {
            throw LLMError.invalidRequest("No models found on the Ollama server at \(configuration.baseURL).")
        }
        return first.id
    }
}
