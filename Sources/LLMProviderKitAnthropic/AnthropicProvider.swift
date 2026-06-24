import Foundation
import LLMProviderKit

/// Provider for the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages).
///
/// Anthropic uses SSE streaming with event types (`message_start`, `content_block_delta`,
/// `message_delta`, etc.). This provider maps them into `LLMStreamChunk`.
public struct AnthropicProvider: LLMProvider {
    public static let name: String = "anthropic"

    public let configuration: LLMProviderConfiguration

    public init(configuration: LLMProviderConfiguration) {
        self.configuration = configuration
    }

    public func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        let url = configuration.baseURL
            .appendingPathComponent("messages")

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("application/json", forHTTPHeaderField: "Accept")

        if let apiKey = configuration.apiKey {
            urlRequest.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        }
        urlRequest.setValue("LLMProviderKit/1.0", forHTTPHeaderField: "anthropic-version")

        // Anthropic requires max_tokens; supply a reasonable default if none given.
        let maxTokens = request.maxTokens ?? 4096

        let body = AnthropicRequest(
            model: request.model,
            messages: request.messages.map { msg in
                AnthropicMessage(
                    role: Self.anthropicRole(for: msg.role),
                    content: msg.content
                )
            },
            system: request.messages.first(where: { $0.role == .system })?.content,
            maxTokens: maxTokens,
            temperature: request.temperature,
            topP: request.topP,
            stream: stream
        )

        urlRequest.httpBody = try JSONEncoder().encode(body)
        return urlRequest
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        // SSE lines: `event: type` and `data: {json}`.
        if line.isEmpty || line.hasPrefix(":") { return [] }

        if line.hasPrefix("event: ") {
            return []
        }

        let prefix = "data: "
        guard line.hasPrefix(prefix) else { return [] }

        let payload = String(line.dropFirst(prefix.count))
        guard let data = payload.data(using: .utf8) else {
            throw LLMError.streamingError("Invalid UTF-8 in Anthropic stream line")
        }

        // Anthropic error events are also embedded in SSE data.
        let event = try JSONDecoder().decode(AnthropicStreamEvent.self, from: data)

        switch event.type {
        case "content_block_delta":
            guard let text = event.delta?.text, !text.isEmpty else { return [] }
            return [.text(text)]
        case "message_delta":
            let usage = event.usage.map { u in
                LLMUsage(
                    promptTokens: u.inputTokens,
                    completionTokens: u.outputTokens,
                    totalTokens: (u.inputTokens ?? 0) + (u.outputTokens ?? 0)
                )
            }
            let reason = event.delta?.stopReason.map { r -> LLMFinishReason in
                switch r {
                case "end_turn": return .stop
                case "max_tokens": return .length
                case "stop_sequence": return .stop
                default: return .unknown
                }
            }
            return [.finish(reason: reason, usage: usage)]
        case "error":
            let errorText = event.error?.message ?? "Unknown Anthropic stream error"
            return [.error(LLMError.providerError(errorText))]
        default:
            return []
        }
    }

    public func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        let decoded = try JSONDecoder().decode(AnthropicResponse.self, from: data)
        let text = decoded.content
            .compactMap(\.text)
            .joined()

        let usage = decoded.usage.map { u in
            LLMUsage(
                promptTokens: u.inputTokens,
                completionTokens: u.outputTokens,
                totalTokens: (u.inputTokens ?? 0) + (u.outputTokens ?? 0)
            )
        }

        let finishReason = decoded.stopReason.map { reason -> LLMFinishReason in
            switch reason {
            case "end_turn": return .stop
            case "max_tokens": return .length
            case "stop_sequence": return .stop
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

    /// Anthropic does not expose a public models endpoint, so this returns a
    /// curated static list of current models. Developers can override it by
    /// registering their own list with `LLMModelRegistry`.
    public func availableModels() async throws -> [LLMModelInfo] {
        Self.curatedModels
    }

    private static func anthropicRole(for role: LLMMessageRole) -> String {
        switch role {
        case .system: return "user" // System messages go in the `system` field, not here.
        case .user: return "user"
        case .assistant: return "assistant"
        case .tool: return "user"
        }
    }
}

// MARK: - Anthropic API types

private struct AnthropicRequest: Encodable {
    let model: String
    let messages: [AnthropicMessage]
    let system: String?
    let maxTokens: Int
    let temperature: Double?
    let topP: Double?
    let stream: Bool

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case system
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case stream
    }
}

private struct AnthropicMessage: Encodable {
    let role: String
    let content: String
}

private struct AnthropicResponse: Decodable {
    struct ContentBlock: Decodable {
        let type: String
        let text: String?
    }

    struct Usage: Decodable {
        let inputTokens: Int?
        let outputTokens: Int?

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
        }
    }

    let id: String?
    let type: String?
    let role: String?
    let content: [ContentBlock]
    let stopReason: String?
    let usage: Usage?

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case role
        case content
        case stopReason = "stop_reason"
        case usage
    }
}

private struct AnthropicStreamEvent: Decodable {
    struct Delta: Decodable {
        let type: String?
        let text: String?
        let stopReason: String?

        enum CodingKeys: String, CodingKey {
            case type
            case text
            case stopReason = "stop_reason"
        }
    }

    struct Usage: Decodable {
        let inputTokens: Int?
        let outputTokens: Int?

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
        }
    }

    struct ErrorDetail: Decodable {
        let type: String
        let message: String
    }

    let type: String
    let delta: Delta?
    let usage: Usage?
    let error: ErrorDetail?

    enum CodingKeys: String, CodingKey {
        case type
        case delta
        case usage
        case error
    }
}

// MARK: - Model constants

/// Well-known Anthropic model names. These are convenience constants only;
/// pass any string to `LLMRequest.model` for models not listed here.
public enum AnthropicModel {
    public static let sonnet35 = "claude-3-5-sonnet-20241022"
    public static let haiku35 = "claude-3-5-haiku-20241022"
    public static let opus3 = "claude-3-opus-20240229"
    public static let sonnet4 = "claude-4-sonnet-20250514"
    public static let opus4 = "claude-4-opus-20250514"
    public static let sonnet = sonnet35
}

// MARK: - Configuration presets

extension AnthropicProvider {
    /// Curated static model list for Anthropic.
    public static let curatedModels: [LLMModelInfo] = [
        LLMModelInfo(
            id: AnthropicModel.opus4,
            providerName: name,
            displayName: "Claude 4 Opus",
            contextWindow: 200_000,
            capabilities: [.chat, .streaming, .reasoning, .tools, .vision]
        ),
        LLMModelInfo(
            id: AnthropicModel.sonnet4,
            providerName: name,
            displayName: "Claude 4 Sonnet",
            contextWindow: 200_000,
            capabilities: [.chat, .streaming, .reasoning, .tools, .vision]
        ),
        LLMModelInfo(
            id: AnthropicModel.sonnet35,
            providerName: name,
            displayName: "Claude 3.5 Sonnet",
            contextWindow: 200_000,
            capabilities: [.chat, .streaming, .tools, .vision]
        ),
        LLMModelInfo(
            id: AnthropicModel.haiku35,
            providerName: name,
            displayName: "Claude 3.5 Haiku",
            contextWindow: 200_000,
            capabilities: [.chat, .streaming, .tools]
        ),
        LLMModelInfo(
            id: AnthropicModel.opus3,
            providerName: name,
            displayName: "Claude 3 Opus",
            contextWindow: 200_000,
            capabilities: [.chat, .streaming, .reasoning, .tools, .vision]
        )
    ]

    /// Convenience configuration for the official Anthropic API.
    public static func anthropic(apiKey: String, model: String = AnthropicModel.sonnet35) -> LLMProviderConfiguration {
        LLMProviderConfiguration(
            name: name,
            baseURL: URL(string: "https://api.anthropic.com/v1")!,
            apiKey: apiKey,
            defaultModel: model
        )
    }
}
