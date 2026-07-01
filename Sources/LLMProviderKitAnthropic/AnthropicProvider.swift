import Foundation
import LLMProviderKit

/// Provider for the [Anthropic Messages API](https://docs.anthropic.com/en/api/messages).
///
/// Anthropic uses SSE streaming with event types (`message_start`, `content_block_delta`,
/// `message_delta`, etc.). This provider maps them into `LLMStreamChunk`.
///
/// Supports native tool calling via `tools` (with `input_schema`) in the request
/// and `tool_use` content blocks in the response.
public struct AnthropicProvider: LLMProvider {
    public static let name: String = "anthropic"

    public let configuration: LLMProviderConfiguration

    public init(configuration: LLMProviderConfiguration) {
        self.configuration = configuration
    }

    public func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        let url = configuration.baseURL.appendingPathComponent("messages")

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("application/json", forHTTPHeaderField: "Accept")

        if let apiKey = configuration.apiKey {
            urlRequest.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        }
        urlRequest.setValue("LLMProviderKit/1.0", forHTTPHeaderField: "anthropic-version")

        let maxTokens = request.maxTokens ?? 4096

        // Build body as a dictionary (handles [String: Any] tool parameters natively)
        var bodyDict: [String: Any] = [
            "model": request.model,
            "max_tokens": maxTokens,
            "stream": stream
        ]

        // Separate system messages from conversation messages
        var messages: [[String: Any]] = []
        var systemText: String?

        for msg in request.messages {
            if msg.role == .system {
                systemText = msg.content
                continue
            }

            let role = Self.anthropicRole(for: msg.role)
            var msgDict: [String: Any] = ["role": role]

            // Tool result messages → content as tool_result blocks
            if msg.role == .tool {
                var blocks: [[String: Any]] = []
                if let toolCallId = msg.toolCallId {
                    blocks.append([
                        "type": "tool_result",
                        "tool_use_id": toolCallId,
                        "content": msg.content
                    ])
                } else {
                    blocks.append(["type": "text", "text": msg.content])
                }
                msgDict["content"] = blocks
            }
            // Assistant messages with tool calls → content as text + tool_use blocks
            else if msg.role == .assistant, let toolCalls = msg.toolCalls, !toolCalls.isEmpty {
                var blocks: [[String: Any]] = []
                if !msg.content.isEmpty {
                    blocks.append(["type": "text", "text": msg.content])
                }
                for tc in toolCalls {
                    var input: [String: Any] = [:]
                    if let decoded = tc.decodedArguments() {
                        input = decoded
                    }
                    blocks.append([
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": input
                    ])
                }
                msgDict["content"] = blocks
            }
            // User messages with images → content as text + image blocks
            else if !msg.images.isEmpty {
                var blocks: [[String: Any]] = [["type": "text", "text": msg.content]]
                for img in msg.images {
                    blocks.append([
                        "type": "image",
                        "source": [
                            "type": "base64",
                            "media_type": img.mimeType,
                            "data": img.base64
                        ]
                    ])
                }
                msgDict["content"] = blocks
            }
            // Plain text messages
            else {
                msgDict["content"] = msg.content
            }

            messages.append(msgDict)
        }

        bodyDict["messages"] = messages

        if let sys = systemText {
            bodyDict["system"] = sys
        }

        if let temp = request.temperature { bodyDict["temperature"] = temp }
        if let topP = request.topP { bodyDict["top_p"] = topP }

        // Tools (Anthropic format: name, description, input_schema)
        if !request.tools.isEmpty {
            bodyDict["tools"] = request.tools.map { tool -> [String: Any] in
                [
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters
                ]
            }
            bodyDict["tool_choice"] = ["type": "auto"]
        }

        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: bodyDict, options: [])
        return urlRequest
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        if line.isEmpty || line.hasPrefix(":") { return [] }
        if line.hasPrefix("event: ") { return [] }

        let prefix = "data: "
        guard line.hasPrefix(prefix) else { return [] }

        let payload = String(line.dropFirst(prefix.count))
        guard let data = payload.data(using: .utf8) else {
            throw LLMError.streamingError("Invalid UTF-8 in Anthropic stream line")
        }

        let event = try JSONDecoder().decode(AnthropicStreamEvent.self, from: data)

        switch event.type {
        case "content_block_start":
            // Tool use blocks arrive as content_block_start with type == "tool_use"
            if let block = event.contentBlock, block.type == "tool_use" {
                let toolCall = LLMToolCall(
                    id: block.id ?? UUID().uuidString,
                    name: block.name ?? "",
                    arguments: "{}"
                )
                return [.toolCall(toolCall)]
            }
            return []
        case "content_block_delta":
            // Text deltas
            if let text = event.delta?.text, !text.isEmpty {
                return [.text(text)]
            }
            // Tool input deltas (partial_json) — cannot accumulate across stateless calls,
            // so we skip emitting incomplete argument fragments.
            return []
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
                case "tool_use": return .toolCalls
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

        // Extract text and tool calls from content blocks
        var text = ""
        var toolCalls: [LLMToolCall] = []

        for block in decoded.content {
            if let blockText = block.text {
                text += blockText
            }
            if block.type == "tool_use" {
                let inputData = try? JSONSerialization.data(withJSONObject: block.input ?? [:], options: [])
                let inputString = inputData.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                toolCalls.append(LLMToolCall(
                    id: block.id ?? UUID().uuidString,
                    name: block.name ?? "",
                    arguments: inputString
                ))
            }
        }

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
            case "tool_use": return .toolCalls
            default: return .unknown
            }
        } ?? (toolCalls.isEmpty ? .stop : .toolCalls)

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
        Self.curatedModels
    }

    private static func anthropicRole(for role: LLMMessageRole) -> String {
        switch role {
        case .system: return "user"
        case .user: return "user"
        case .assistant: return "assistant"
        case .tool: return "user" // Anthropic expects tool results as user messages
        }
    }
}

// MARK: - Anthropic API response types

private struct AnthropicResponse: Decodable {
    struct ContentBlock: Decodable {
        let type: String
        let text: String?
        let id: String?
        let name: String?
        let input: [String: Any]?

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.type = try container.decode(String.self, forKey: .type)
            self.text = try container.decodeIfPresent(String.self, forKey: .text)
            self.id = try container.decodeIfPresent(String.self, forKey: .id)
            self.name = try container.decodeIfPresent(String.self, forKey: .name)
            if let decoded = try container.decodeIfPresent(JSONValue.self, forKey: .input) {
                if case .object(let object) = decoded {
                    self.input = object.mapValues { $0.anyValue }
                } else {
                    self.input = nil
                }
            } else {
                self.input = nil
            }
        }

        enum CodingKeys: String, CodingKey {
            case type, text, id, name, input
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

    let id: String?
    let type: String?
    let role: String?
    let content: [ContentBlock]
    let stopReason: String?
    let usage: Usage?

    enum CodingKeys: String, CodingKey {
        case id, type, role, content
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
            case type, text
            case stopReason = "stop_reason"
        }
    }

    struct ContentBlock: Decodable {
        let type: String?
        let id: String?
        let name: String?
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
    let contentBlock: ContentBlock?
    let usage: Usage?
    let error: ErrorDetail?

    enum CodingKeys: String, CodingKey {
        case type, delta, usage, error
        case contentBlock = "content_block"
    }
}

private enum JSONValue: Decodable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .null
        }
    }

    var anyValue: Any {
        switch self {
        case .string(let value): return value
        case .number(let value): return value
        case .bool(let value): return value
        case .object(let value): return value.mapValues { $0.anyValue }
        case .array(let value): return value.map { $0.anyValue }
        case .null: return NSNull()
        }
    }
}

// MARK: - Model constants

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

    public static func anthropic(apiKey: String, model: String = AnthropicModel.sonnet35) -> LLMProviderConfiguration {
        LLMProviderConfiguration(
            name: name,
            baseURL: URL(string: "https://api.anthropic.com/v1")!,
            apiKey: apiKey,
            defaultModel: model
        )
    }
}