import Foundation
import LLMProviderKit

/// Provider for OpenAI-compatible APIs.
///
/// Works with OpenAI (`https://api.openai.com/v1`) and any service that exposes
/// the same `/chat/completions` shape (e.g. Groq, xAI, DeepSeek, OpenRouter).
public struct OpenAIProvider: LLMProvider {
    public static let name: String = "openai"

    public let configuration: LLMProviderConfiguration

    public init(configuration: LLMProviderConfiguration) {
        self.configuration = configuration
    }

    public func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        let url = configuration.baseURL
            .appendingPathComponent("chat")
            .appendingPathComponent("completions")

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let apiKey = configuration.apiKey {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        let body = OpenAIChatRequest(
            model: request.model,
            messages: request.messages.map { msg in
                OpenAIMessage(role: msg.role, content: msg.content, images: msg.images)
            },
            temperature: request.temperature,
            topP: request.topP,
            maxTokens: request.maxTokens,
            stream: stream
        )

        urlRequest.httpBody = try JSONEncoder().encode(body)
        return urlRequest
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        // OpenAI streams SSE lines prefixed with `data: `.
        let prefix = "data: "
        guard line.hasPrefix(prefix) else { return [] }

        let payload = String(line.dropFirst(prefix.count))
        if payload == "[DONE]" {
            return [.finish(reason: .stop, usage: nil)]
        }

        guard let data = payload.data(using: .utf8) else {
            throw LLMError.streamingError("Invalid UTF-8 in OpenAI stream line")
        }

        let decoded = try JSONDecoder().decode(OpenAIStreamChunk.self, from: data)
        var chunks: [LLMStreamChunk] = []

        if let delta = decoded.choices.first?.delta?.content, !delta.isEmpty {
            chunks.append(.text(delta))
        }

        if let reason = decoded.choices.first?.finishReason {
            let mapped: LLMFinishReason = switch reason {
            case "stop": .stop
            case "length": .length
            case "content_filter": .contentFilter
            default: .unknown
            }
            chunks.append(.finish(reason: mapped, usage: nil))
        }

        return chunks
    }

    public func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        let decoded = try JSONDecoder().decode(OpenAIChatResponse.self, from: data)
        let text = decoded.choices.first?.message?.content ?? ""
        let finishReason = decoded.choices.first?.finishReason.map { reason -> LLMFinishReason in
            switch reason {
            case "stop": return .stop
            case "length": return .length
            case "content_filter": return .contentFilter
            default: return .unknown
            }
        }
        let usage = decoded.usage.map {
            LLMUsage(
                promptTokens: $0.promptTokens,
                completionTokens: $0.completionTokens,
                totalTokens: $0.totalTokens
            )
        }
        return LLMResponse(
            text: text,
            finishReason: finishReason,
            usage: usage,
            request: request,
            providerName: Self.name,
            rawData: data
        )
    }

    public func availableModels() async throws -> [LLMModelInfo] {
        let url = configuration.baseURL.appendingPathComponent("models")

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "GET"

        if let apiKey = configuration.apiKey {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await urlSession.data(for: urlRequest)
        try Self.verifyHTTPResponse(response, data: data)

        let decoded = try JSONDecoder().decode(OpenAIModelsResponse.self, from: data)
        return decoded.data.map { model in
            LLMModelInfo(
                id: model.id,
                providerName: Self.name,
                displayName: model.id,
                contextWindow: nil,
                capabilities: [.chat, .streaming]
            )
        }
    }
}

// MARK: - OpenAI chat API types

private struct OpenAIChatRequest: Encodable {
    let model: String
    let messages: [OpenAIMessage]
    let temperature: Double?
    let topP: Double?
    let maxTokens: Int?
    let stream: Bool

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case temperature
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case stream
    }
}

private struct OpenAIMessage: Encodable {
    let role: LLMMessageRole
    let content: OpenAIContent
    let images: [LLMImage]

    init(role: LLMMessageRole, content: String, images: [LLMImage]) {
        self.role = role
        self.images = images
        // When there are images, content becomes an array of parts.
        // Otherwise, keep it as a plain string for backward compatibility.
        if images.isEmpty {
            self.content = .text(content)
        } else {
            var parts: [OpenAIContentPart] = [.text(content)]
            for img in images {
                parts.append(.imageURL("data:\(img.mimeType);base64,\(img.base64)"))
            }
            self.content = .parts(parts)
        }
    }

    enum CodingKeys: String, CodingKey {
        case role
        case content
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        switch content {
        case .text(let str):
            try container.encode(str, forKey: .content)
        case .parts(let parts):
            try container.encode(parts, forKey: .content)
        }
    }
}

private enum OpenAIContent {
    case text(String)
    case parts([OpenAIContentPart])
}

private enum OpenAIContentPart: Encodable {
    case text(String)
    case imageURL(String) // data:<mime>;base64,<b64>

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)
        case .imageURL(let url):
            try container.encode("image_url", forKey: .type)
            try container.encode(OpenAIImageURL(url: url), forKey: .imageURL)
        }
    }

    enum CodingKeys: String, CodingKey {
        case type
        case text
        case imageURL = "image_url"
    }
}

private struct OpenAIImageURL: Encodable {
    let url: String
}

private struct OpenAIChatResponse: Decodable {
    struct Choice: Decodable {
        struct Message: Decodable {
            let role: String?
            let content: String?
        }
        let message: Message?
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case message
            case finishReason = "finish_reason"
        }
    }

    struct Usage: Decodable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }

    let id: String?
    let choices: [Choice]
    let usage: Usage?
}

private struct OpenAIStreamChunk: Decodable {
    struct Choice: Decodable {
        struct Delta: Decodable {
            let role: String?
            let content: String?
        }
        let delta: Delta?
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case delta
            case finishReason = "finish_reason"
        }
    }

    let id: String?
    let choices: [Choice]
}

// MARK: - OpenAI models API types

private struct OpenAIModelsResponse: Decodable {
    struct Model: Decodable {
        let id: String
    }
    let data: [Model]
}

// MARK: - Model constants

/// Well-known OpenAI model names. These are convenience constants only;
/// pass any string to `LLMRequest.model` for models not listed here.
public enum OpenAIModel {
    public static let gpt4o = "gpt-4o"
    public static let gpt4oMini = "gpt-4o-mini"
    public static let gpt4Turbo = "gpt-4-turbo"
    public static let gpt4 = "gpt-4"
    public static let gpt35Turbo = "gpt-3.5-turbo"
    public static let o1 = "o1"
    public static let o1Mini = "o1-mini"
    public static let o3Mini = "o3-mini"
    public static let dalle3 = "dall-e-3"
    public static let whisper1 = "whisper-1"
}

// MARK: - Configuration presets

extension OpenAIProvider {
    /// Convenience configuration for the official OpenAI API.
    public static func openAI(apiKey: String, model: String = OpenAIModel.gpt4oMini) -> LLMProviderConfiguration {
        LLMProviderConfiguration(
            name: name,
            baseURL: URL(string: "https://api.openai.com/v1")!,
            apiKey: apiKey,
            defaultModel: model
        )
    }
}
