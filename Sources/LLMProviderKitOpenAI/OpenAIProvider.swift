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

        let curatedByID = Dictionary(uniqueKeysWithValues: Self.curatedModels.map { ($0.id, $0) })
        let decoded = try JSONDecoder().decode(OpenAIModelsResponse.self, from: data)
        return decoded.data.map { model in
            LLMModelInfo(
                id: model.id,
                providerName: Self.name,
                displayName: model.id,
                contextWindow: nil,
                capabilities: Self.capabilities(for: model.id),
                categories: Self.categories(for: model.id)
            ).enriched(with: curatedByID[model.id])
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
    // GPT-5.5 family (current flagship)
    public static let gpt55 = "gpt-5.5"
    // GPT-5.4 family
    public static let gpt54 = "gpt-5.4"
    public static let gpt54Mini = "gpt-5.4-mini"
    // GPT-4o family (still available)
    public static let gpt4o = "gpt-4o"
    public static let gpt4oMini = "gpt-4o-mini"
}

// MARK: - Configuration presets

extension OpenAIProvider {
    public static let curatedModels: [LLMModelInfo] = [
        LLMModelInfo(
            id: OpenAIModel.gpt55,
            providerName: name,
            displayName: "GPT-5.5",
            contextWindow: 1_000_000,
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .reasoning, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable,
            notes: "Flagship model for complex reasoning and coding."
        ),
        LLMModelInfo(
            id: OpenAIModel.gpt54,
            providerName: name,
            displayName: "GPT-5.4",
            contextWindow: 1_000_000,
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .reasoning, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable,
            notes: "Frontier model with lower cost than the flagship tier."
        ),
        LLMModelInfo(
            id: OpenAIModel.gpt54Mini,
            providerName: name,
            displayName: "GPT-5.4 mini",
            contextWindow: 400_000,
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .reasoning, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable,
            notes: "Fast, cost-efficient model suited to app agents and subagents."
        ),
        LLMModelInfo(
            id: OpenAIModel.gpt4o,
            providerName: name,
            displayName: "GPT-4o",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .legacy
        ),
        LLMModelInfo(
            id: OpenAIModel.gpt4oMini,
            providerName: name,
            displayName: "GPT-4o mini",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .legacy
        ),
    ]

    static func capabilities(for modelID: String) -> Set<LLMModelCapability> {
        let lowercased = modelID.lowercased()
        if lowercased.contains("embedding") || lowercased.contains("embed") {
            return [.embeddings]
        }
        if lowercased.contains("tts") || lowercased.contains("speech") {
            return [.textToSpeech, .audioGeneration]
        }
        if lowercased.contains("whisper") || lowercased.contains("transcribe") {
            return [.speechToText, .audioInput]
        }
        if lowercased.contains("image") || lowercased.contains("dall") {
            return [.imageGeneration]
        }

        // The OpenAI models endpoint can include non-chat products such as
        // moderation, realtime, and computer-use models. Do not advertise
        // chat/tool/vision support for those just because their IDs happen to
        // contain a broad substring like "o".
        if isKnownNonChatModel(lowercased) {
            return []
        }

        var capabilities: Set<LLMModelCapability> = [.chat, .textGeneration, .streaming]
        if lowercased.contains("gpt") || isOSeriesReasoningModel(lowercased) {
            capabilities.formUnion([.tools, .vision, .imageInput, .structuredOutput])
        }
        if lowercased.contains("gpt-5") || isOSeriesReasoningModel(lowercased) {
            capabilities.insert(.reasoning)
        }
        return capabilities
    }

    private static func isKnownNonChatModel(_ modelID: String) -> Bool {
        modelID.contains("moderation") ||
            modelID.contains("realtime") ||
            modelID.contains("computer-use") ||
            modelID.contains("search-preview") ||
            modelID.contains("codex")
    }

    private static func isOSeriesReasoningModel(_ modelID: String) -> Bool {
        // Match o-series model families as prefixes/tokens: o3, o3-mini,
        // o4-mini, o1-preview, etc. Avoid `contains("o")`, which matches
        // unrelated IDs like omni-moderation or computer-use-preview.
        let tokens = modelID.split { !$0.isLetter && !$0.isNumber }.map(String.init)
        guard let first = tokens.first else { return false }
        return first.range(of: #"^o[1-9][a-z]*$"#, options: .regularExpression) != nil
    }

    static func categories(for modelID: String) -> Set<LLMModelCategory> {
        let capabilities = capabilities(for: modelID)
        var categories: Set<LLMModelCategory> = []
        if !capabilities.intersection([.chat, .textGeneration, .tools, .reasoning]).isEmpty { categories.insert(.text) }
        if !capabilities.intersection([.vision, .imageInput]).isEmpty { categories.formUnion([.vision, .multimodal]) }
        if capabilities.contains(.imageGeneration) { categories.insert(.image) }
        if !capabilities.intersection([.audioInput, .audioGeneration, .speechToText, .textToSpeech]).isEmpty { categories.insert(.audio) }
        if capabilities.contains(.embeddings) { categories.insert(.embedding) }
        return categories
    }

    public static func openAI(apiKey: String, model: String = OpenAIModel.gpt54Mini) -> LLMProviderConfiguration {
        LLMProviderConfiguration(
            name: name,
            baseURL: URL(string: "https://api.openai.com/v1")!,
            apiKey: apiKey,
            defaultModel: model
        )
    }
}