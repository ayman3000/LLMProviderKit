import Foundation
import LLMProviderKit

/// Provider for Google Gemini (`https://generativelanguage.googleapis.com/v1beta`).
///
/// Gemini exposes `generateContent` for non-streaming and
/// `streamGenerateContent` for streaming. The content shape is a flat list of
/// "parts" that we map from/to `LLMMessage`.
public struct GeminiProvider: LLMProvider {
    public static let name: String = "gemini"

    public let configuration: LLMProviderConfiguration

    public init(configuration: LLMProviderConfiguration) {
        self.configuration = configuration
    }

    public func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        let action = stream ? "streamGenerateContent" : "generateContent"
        let path = "models/\(request.model):\(action)"

        guard var components = URLComponents(url: configuration.baseURL, resolvingAgainstBaseURL: true) else {
            throw LLMError.invalidRequest("Invalid Gemini base URL")
        }
        components.path = (components.path as NSString).appendingPathComponent(path)

        var queryItems: [URLQueryItem] = []
        if let apiKey = configuration.apiKey {
            queryItems.append(URLQueryItem(name: "key", value: apiKey))
        }
        // Gemini's streaming endpoint returns newline-delimited JSON by default.
        // Request SSE format (data: {...}\n\n) so our line-based parser works cleanly.
        if stream {
            queryItems.append(URLQueryItem(name: "alt", value: "sse"))
        }
        if !queryItems.isEmpty {
            components.queryItems = queryItems
        }

        guard let url = components.url else {
            throw LLMError.invalidRequest("Could not build Gemini URL")
        }

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let contents = request.messages.map { message -> GeminiContent in
            var parts: [GeminiPart] = [.text(message.content)]
            for img in message.images {
                parts.append(.inlineData(mimeType: img.mimeType, data: img.base64))
            }
            return GeminiContent(role: Self.geminiRole(for: message.role), parts: parts)
        }

        let body = GeminiRequest(
            contents: contents,
            generationConfig: GeminiGenerationConfig(
                temperature: request.temperature,
                topP: request.topP,
                maxOutputTokens: request.maxTokens
            )
        )

        urlRequest.httpBody = try JSONEncoder().encode(body)
        return urlRequest
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        // Gemini SSE streams use "data: {...}" format.
        var jsonLine = line
        if jsonLine.hasPrefix("data: ") {
            jsonLine = String(jsonLine.dropFirst(6))
        } else if jsonLine.hasPrefix("data:") {
            jsonLine = String(jsonLine.dropFirst(5))
        }

        guard let data = jsonLine.data(using: .utf8), !data.isEmpty else {
            return []
        }

        let decoded = try JSONDecoder().decode(GeminiResponse.self, from: data)
        let text = decoded.candidates?
            .first?
            .content?
            .parts?
            .compactMap(\.text)
            .joined() ?? ""

        var chunks: [LLMStreamChunk] = []
        if !text.isEmpty {
            chunks.append(.text(text))
        }

        let finishReason = decoded.candidates?.first?.finishReason
        if finishReason != nil || decoded.usageMetadata != nil {
            let usage = decoded.usageMetadata.map { meta -> LLMUsage in
                LLMUsage(
                    promptTokens: meta.promptTokenCount,
                    completionTokens: meta.candidatesTokenCount,
                    totalTokens: meta.totalTokenCount
                )
            }
            let mappedReason: LLMFinishReason? = finishReason.map {
                switch $0 {
                case "STOP": return .stop
                case "MAX_TOKENS": return .length
                case "SAFETY": return .contentFilter
                default: return .unknown
                }
            }
            chunks.append(.finish(reason: mappedReason, usage: usage))
        }

        return chunks
    }

    public func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        let decoded = try JSONDecoder().decode(GeminiResponse.self, from: data)
        let text = decoded.candidates?
            .first?
            .content?
            .parts?
            .compactMap(\.text)
            .joined() ?? ""

        let usage = decoded.usageMetadata.map { meta in
            LLMUsage(
                promptTokens: meta.promptTokenCount,
                completionTokens: meta.candidatesTokenCount,
                totalTokens: meta.totalTokenCount
            )
        }

        let finishReason = decoded.candidates?.first?.finishReason.map { reason -> LLMFinishReason in
            switch reason {
            case "STOP": return .stop
            case "MAX_TOKENS": return .length
            case "SAFETY": return .contentFilter
            default: return .unknown
            }
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
        guard var components = URLComponents(url: configuration.baseURL, resolvingAgainstBaseURL: true) else {
            throw LLMError.invalidRequest("Invalid Gemini base URL")
        }
        components.path = (components.path as NSString).appendingPathComponent("models")

        if let apiKey = configuration.apiKey {
            var query = components.queryItems ?? []
            query.append(URLQueryItem(name: "key", value: apiKey))
            components.queryItems = query
        }

        guard let url = components.url else {
            throw LLMError.invalidRequest("Could not build Gemini URL")
        }

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "GET"

        let (data, response) = try await urlSession.data(for: urlRequest)
        try Self.verifyHTTPResponse(response, data: data)

        let decoded = try JSONDecoder().decode(GeminiModelsResponse.self, from: data)
        return decoded.models.map { model in
            LLMModelInfo(
                id: model.name,
                providerName: Self.name,
                displayName: model.displayName ?? model.name,
                contextWindow: model.inputTokenLimit,
                capabilities: Self.capabilities(for: model)
            )
        }
    }

    private static func geminiRole(for role: LLMMessageRole) -> String {
        switch role {
        case .system: return "user" // Gemini has no system role; treat as user.
        case .user: return "user"
        case .assistant: return "model"
        case .tool: return "model" // Approximation.
        }
    }

    private static func capabilities(for model: GeminiModelsResponse.Model) -> Set<LLMModelCapability> {
        var caps: Set<LLMModelCapability> = [.chat, .streaming]
        if let supported = model.supportedGenerationMethods {
            for method in supported {
                switch method {
                case "generateContent": caps.insert(.chat)
                case "countTokens": break
                case "embedContent": break
                default: break
                }
            }
        }
        return caps
    }
}

// MARK: - Gemini API types

private struct GeminiRequest: Encodable {
    let contents: [GeminiContent]
    let generationConfig: GeminiGenerationConfig?
}

private struct GeminiContent: Encodable {
    let role: String?
    let parts: [GeminiPart]
}

private enum GeminiPart: Encodable {
    case text(String)
    case inlineData(mimeType: String, data: String)

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let value):
            try container.encode(value, forKey: .text)
        case .inlineData(let mimeType, let data):
            try container.encode(GeminiInlineData(mimeType: mimeType, data: data), forKey: .inlineData)
        }
    }

    enum CodingKeys: String, CodingKey {
        case text
        case inlineData = "inlineData"
    }
}

private struct GeminiInlineData: Encodable {
    let mimeType: String
    let data: String
}

private struct GeminiGenerationConfig: Encodable {
    let temperature: Double?
    let topP: Double?
    let maxOutputTokens: Int?

    enum CodingKeys: String, CodingKey {
        case temperature
        case topP = "top_p"
        case maxOutputTokens = "max_output_tokens"
    }
}

private struct GeminiResponse: Decodable {
    struct Candidate: Decodable {
        struct Content: Decodable {
            struct Part: Decodable {
                let text: String?
            }
            let role: String?
            let parts: [Part]?
        }
        let content: Content?
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case content
            case finishReason
        }
    }

    struct UsageMetadata: Decodable {
        let promptTokenCount: Int?
        let candidatesTokenCount: Int?
        let totalTokenCount: Int?

        enum CodingKeys: String, CodingKey {
            case promptTokenCount = "promptTokenCount"
            case candidatesTokenCount = "candidatesTokenCount"
            case totalTokenCount = "totalTokenCount"
        }
    }

    let candidates: [Candidate]?
    let usageMetadata: UsageMetadata?
}

// MARK: - Gemini models API types

private struct GeminiModelsResponse: Decodable {
    struct Model: Decodable {
        let name: String
        let displayName: String?
        let inputTokenLimit: Int?
        let supportedGenerationMethods: [String]?

        enum CodingKeys: String, CodingKey {
            case name
            case displayName
            case inputTokenLimit
            case supportedGenerationMethods
        }
    }

    let models: [Model]
}

// MARK: - Model constants

/// Well-known Gemini model names. These are convenience constants only;
/// pass any string to `LLMRequest.model` for models not listed here.
public enum GeminiModel {
    public static let flash = "gemini-2.5-flash"
    public static let flashLite = "gemini-2.5-flash-lite"
    public static let pro = "gemini-2.5-pro"
    public static let flash15 = "gemini-1.5-flash"
    public static let pro15 = "gemini-1.5-pro"
}

// MARK: - Configuration presets

extension GeminiProvider {
    /// Convenience configuration for the official Gemini API.
    public static func gemini(apiKey: String, model: String = GeminiModel.flash) -> LLMProviderConfiguration {
        LLMProviderConfiguration(
            name: name,
            baseURL: URL(string: "https://generativelanguage.googleapis.com/v1beta")!,
            apiKey: apiKey,
            defaultModel: model
        )
    }
}
