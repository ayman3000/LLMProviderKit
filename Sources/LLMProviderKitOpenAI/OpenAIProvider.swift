import Foundation
import LLMProviderKit

/// Provider for OpenAI-compatible APIs.
///
/// Works with OpenAI (`https://api.openai.com/v1`) and any service that exposes
/// the same `/chat/completions` shape (e.g. Groq, xAI, DeepSeek, OpenRouter).
/// Supports native tool calling (function calling).
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

        // Build body as a dictionary to handle [String: Any] tool parameters
        var bodyDict: [String: Any] = [
            "model": request.model,
            "messages": request.messages.map { msg -> [String: Any] in
                var msgDict: [String: Any] = [
                    "role": msg.role.rawValue,
                    "content": msg.content
                ]
                if !msg.images.isEmpty {
                    var parts: [[String: Any]] = [["type": "text", "text": msg.content]]
                    for img in msg.images {
                        parts.append([
                            "type": "image_url",
                            "image_url": ["url": "data:\(img.mimeType);base64,\(img.base64)"]
                        ])
                    }
                    msgDict["content"] = parts
                }
                if let toolCallId = msg.toolCallId {
                    msgDict["tool_call_id"] = toolCallId
                }
                // Include tool_calls for assistant messages
                if let toolCalls = msg.toolCalls, !toolCalls.isEmpty {
                    msgDict["tool_calls"] = toolCalls.map { tc -> [String: Any] in
                        [
                            "id": tc.id,
                            "type": "function",
                            "function": [
                                "name": tc.name,
                                "arguments": tc.arguments
                            ]
                        ]
                    }
                }
                return msgDict
            },
            "stream": stream
        ]

        if let temp = request.temperature { bodyDict["temperature"] = temp }
        if let topP = request.topP { bodyDict["top_p"] = topP }
        if let maxTokens = request.maxTokens { bodyDict["max_tokens"] = maxTokens }

        // Add tools if any
        if !request.tools.isEmpty {
            bodyDict["tools"] = request.tools.map { tool -> [String: Any] in
                [
                    "type": "function",
                    "function": [
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    ]
                ]
            }
            bodyDict["tool_choice"] = "auto"
        }

        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: bodyDict, options: [])
        return urlRequest
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
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

        if let toolCalls = decoded.choices.first?.delta?.toolCalls, !toolCalls.isEmpty {
            for tc in toolCalls {
                chunks.append(.toolCall(LLMToolCall(
                    id: tc.id ?? UUID().uuidString,
                    name: tc.function?.name ?? "",
                    arguments: tc.function?.arguments ?? "{}"
                )))
            }
        }

        if let reason = decoded.choices.first?.finishReason {
            let mapped: LLMFinishReason = switch reason {
            case "stop": .stop
            case "length": .length
            case "content_filter": .contentFilter
            case "tool_calls": .toolCalls
            default: .unknown
            }
            chunks.append(.finish(reason: mapped, usage: nil))
        }

        return chunks
    }

    public func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        let decoded = try JSONDecoder().decode(OpenAIChatResponse.self, from: data)
        let text = decoded.choices.first?.message?.content ?? ""

        // Parse native tool calls
        let toolCalls: [LLMToolCall] = decoded.choices.first?.message?.toolCalls?.map { tc in
            LLMToolCall(
                id: tc.id ?? UUID().uuidString,
                name: tc.function?.name ?? "",
                arguments: tc.function?.arguments ?? "{}"
            )
        } ?? []

        let finishReason = decoded.choices.first?.finishReason.map { reason -> LLMFinishReason in
            switch reason {
            case "stop": return .stop
            case "length": return .length
            case "content_filter": return .contentFilter
            case "tool_calls": return .toolCalls
            default: return .unknown
            }
        } ?? (toolCalls.isEmpty ? .stop : .toolCalls)

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
            toolCalls: toolCalls,
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

// MARK: - OpenAI chat API response types

private struct OpenAIChatResponse: Decodable {
    struct Choice: Decodable {
        struct Message: Decodable {
            let role: String?
            let content: String?
            let toolCalls: [OpenAIToolCall]?

            enum CodingKeys: String, CodingKey {
                case role, content
                case toolCalls = "tool_calls"
            }
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

private struct OpenAIToolCall: Decodable {
    let id: String?
    let function: OpenAIToolCallFunction?
}

private struct OpenAIToolCallFunction: Decodable {
    let name: String?
    let arguments: String?
}

private struct OpenAIStreamChunk: Decodable {
    struct Choice: Decodable {
        struct Delta: Decodable {
            let role: String?
            let content: String?
            let toolCalls: [OpenAIToolCall]?

            enum CodingKeys: String, CodingKey {
                case role, content
                case toolCalls = "tool_calls"
            }
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
    public static func openAI(apiKey: String, model: String = OpenAIModel.gpt4oMini) -> LLMProviderConfiguration {
        LLMProviderConfiguration(
            name: name,
            baseURL: URL(string: "https://api.openai.com/v1")!,
            apiKey: apiKey,
            defaultModel: model
        )
    }
}