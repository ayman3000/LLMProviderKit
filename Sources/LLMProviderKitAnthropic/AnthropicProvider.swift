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
    public let urlSession: URLSession

    public init(configuration: LLMProviderConfiguration, urlSession: URLSession = .shared) {
        self.configuration = configuration
        self.urlSession = urlSession
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
        // Anthropic requires a dated API version string; an arbitrary value is rejected
        // with HTTP 400 "invalid anthropic-version".
        urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        urlRequest.setValue("LLMProviderKit/1.0", forHTTPHeaderField: "User-Agent")

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
                if let toolCallId = msg.toolCallId {
                    var toolResultContent: [[String: Any]] = [
                        ["type": "text", "text": msg.content]
                    ]
                    for img in msg.images {
                        toolResultContent.append([
                            "type": "image",
                            "source": [
                                "type": "base64",
                                "media_type": img.mimeType,
                                "data": img.base64
                            ]
                        ])
                    }
                    let toolResultBlock: [String: Any] = [
                        "type": "tool_result",
                        "tool_use_id": toolCallId,
                        "content": toolResultContent
                    ]
                    msgDict["content"] = [toolResultBlock]
                } else {
                    msgDict["content"] = [["type": "text", "text": msg.content]]
                }
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

        // Prompt caching. Anthropic caches nothing unless asked: a breakpoint on
        // the last block of the final message caches the whole prefix
        // (tools → system → messages) up to it, and the next call's lookup
        // walks back from its own breakpoint through earlier block boundaries,
        // so an append-only conversation is a hit turn after turn. Below the
        // provider's minimum prefix size the marker is simply ignored.
        // Observed 2026-09-06: without this, cache_read_input_tokens was
        // never reported for any Naseem call.
        bodyDict["messages"] = Self.markingLastBlockForCache(messages)

        if let sys = systemText {
            bodyDict["system"] = [["type": "text", "text": sys, "cache_control": Self.ephemeral]]
        }

        if let temp = request.temperature { bodyDict["temperature"] = temp }
        if let topP = request.topP { bodyDict["top_p"] = topP }

        // Tools (Anthropic format: name, description, input_schema). The last
        // tool carries a breakpoint so the (large, static) tool block caches
        // even when system/messages change.
        if !request.tools.isEmpty {
            var tools = request.tools.map { tool -> [String: Any] in
                [
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters
                ]
            }
            tools[tools.count - 1]["cache_control"] = Self.ephemeral
            bodyDict["tools"] = tools
            bodyDict["tool_choice"] = ["type": "auto"]
        }

        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: bodyDict, options: [])
        return urlRequest
    }

    static let ephemeral: [String: String] = ["type": "ephemeral"]

    /// Put a cache breakpoint on the last content block of the last message.
    /// String content becomes a single text block so it can carry the marker.
    static func markingLastBlockForCache(_ messages: [[String: Any]]) -> [[String: Any]] {
        guard var last = messages.last else { return messages }
        if let text = last["content"] as? String {
            guard !text.isEmpty else { return messages }
            last["content"] = [["type": "text", "text": text, "cache_control": ephemeral]]
        } else if var blocks = last["content"] as? [[String: Any]], !blocks.isEmpty {
            blocks[blocks.count - 1]["cache_control"] = ephemeral
            last["content"] = blocks
        } else {
            return messages
        }
        var out = messages
        out[out.count - 1] = last
        return out
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
            // Extended-thinking deltas (`thinking_delta`) carry `thinking`, not `text`.
            if let thinking = event.delta?.thinking, !thinking.isEmpty {
                return [.reasoning(thinking)]
            }
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
                    totalTokens: (u.inputTokens ?? 0) + (u.outputTokens ?? 0),
                    cachedTokens: u.cacheReadInputTokens
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
                totalTokens: (u.inputTokens ?? 0) + (u.outputTokens ?? 0),
                cachedTokens: u.cacheReadInputTokens
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

    /// Fetch the live model list from `GET /v1/models`, enriching each record
    /// with curated metadata (context window, capabilities, release stage).
    /// Falls back to ``curatedModels`` when the endpoint is unreachable, so
    /// offline apps keep working as before.
    public func availableModels() async throws -> [LLMModelInfo] {
        var components = URLComponents(
            url: configuration.baseURL.appendingPathComponent("models"),
            resolvingAgainstBaseURL: false
        )
        // The API paginates; 1000 is the documented maximum page size and far
        // exceeds the current model count, so one request suffices.
        components?.queryItems = [URLQueryItem(name: "limit", value: "1000")]
        guard let url = components?.url else { return Self.curatedModels }

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "GET"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Accept")
        if let apiKey = configuration.apiKey {
            urlRequest.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        }
        urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

        let live: AnthropicModelsResponse
        do {
            let (data, response) = try await urlSession.data(for: urlRequest)
            try Self.verifyHTTPResponse(response, data: data)
            live = try JSONDecoder().decode(AnthropicModelsResponse.self, from: data)
        } catch {
            return Self.curatedModels
        }

        let curatedByID = Dictionary(uniqueKeysWithValues: Self.curatedModels.map { ($0.id, $0) })
        return live.data.map { model in
            LLMModelInfo(
                id: model.id,
                providerName: Self.name,
                displayName: model.displayName,
                contextWindow: nil,
                capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput],
                categories: [.text, .vision, .multimodal]
            ).enriched(with: curatedByID[model.id])
        }
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

private struct AnthropicModelsResponse: Decodable {
    struct Model: Decodable {
        let id: String
        let displayName: String?

        enum CodingKeys: String, CodingKey {
            case id
            case displayName = "display_name"
        }
    }

    let data: [Model]
}

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
        let cacheReadInputTokens: Int?

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
            case cacheReadInputTokens = "cache_read_input_tokens"
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
        let thinking: String?
        let stopReason: String?

        enum CodingKeys: String, CodingKey {
            case type, text, thinking
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
        let cacheReadInputTokens: Int?

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
            case cacheReadInputTokens = "cache_read_input_tokens"
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
    // Current models
    public static let fable5 = "claude-fable-5"
    public static let opus48 = "claude-opus-4-8"
    public static let opus47 = "claude-opus-4-7"
    public static let opus46 = "claude-opus-4-6"
    public static let sonnet46 = "claude-sonnet-4-6"
    public static let haiku45 = "claude-haiku-4-5-20251001"
    // Legacy
    public static let opus41 = "claude-opus-4-1-20250805"
    public static let sonnet4 = "claude-sonnet-4-20250514"
    public static let opus4 = "claude-opus-4-20250514"
    public static let sonnet35 = "claude-3-5-sonnet-20241022"
    public static let haiku35 = "claude-3-5-haiku-20241022"
    public static let opus3 = "claude-3-opus-20240229"
    public static let sonnet = sonnet46
}

// MARK: - Configuration presets

extension AnthropicProvider {
    public static let curatedModels: [LLMModelInfo] = [
        LLMModelInfo(
            id: AnthropicModel.fable5,
            providerName: name,
            displayName: "Claude Fable 5",
            contextWindow: 1_000_000,
            capabilities: [.chat, .textGeneration, .streaming, .reasoning, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable,
            notes: "Most capable Claude model in the curated list."
        ),
        LLMModelInfo(
            id: AnthropicModel.opus48,
            providerName: name,
            displayName: "Claude Opus 4.8",
            contextWindow: 1_000_000,
            capabilities: [.chat, .textGeneration, .streaming, .reasoning, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable
        ),
        LLMModelInfo(
            id: AnthropicModel.opus47,
            providerName: name,
            displayName: "Claude Opus 4.7",
            contextWindow: 1_000_000,
            capabilities: [.chat, .textGeneration, .streaming, .reasoning, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable
        ),
        LLMModelInfo(
            id: AnthropicModel.opus46,
            providerName: name,
            displayName: "Claude Opus 4.6",
            contextWindow: 1_000_000,
            capabilities: [.chat, .textGeneration, .streaming, .reasoning, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable
        ),
        LLMModelInfo(
            id: AnthropicModel.sonnet46,
            providerName: name,
            displayName: "Claude Sonnet 4.6",
            contextWindow: 1_000_000,
            capabilities: [.chat, .textGeneration, .streaming, .reasoning, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable,
            notes: "Default balance of intelligence, latency, and cost."
        ),
        LLMModelInfo(
            id: AnthropicModel.haiku45,
            providerName: name,
            displayName: "Claude Haiku 4.5",
            contextWindow: 200_000,
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable
        ),
        LLMModelInfo(
            id: AnthropicModel.sonnet4,
            providerName: name,
            displayName: "Claude Sonnet 4 (legacy)",
            contextWindow: 200_000,
            capabilities: [.chat, .textGeneration, .streaming, .reasoning, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .legacy
        ),
        LLMModelInfo(
            id: AnthropicModel.sonnet35,
            providerName: name,
            displayName: "Claude 3.5 Sonnet (legacy)",
            contextWindow: 200_000,
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .legacy
        ),
    ]

    public static func anthropic(apiKey: String, model: String = AnthropicModel.sonnet46) -> LLMProviderConfiguration {
        LLMProviderConfiguration(
            name: name,
            baseURL: URL(string: "https://api.anthropic.com/v1")!,
            apiKey: apiKey,
            defaultModel: model
        )
    }
}