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

        // Build the request body as a dictionary (handles [String: Any] parameters natively)
        var bodyDict: [String: Any] = [
            "model": request.model,
            "messages": request.messages.map { msg -> [String: Any] in
                var msgDict: [String: Any] = [
                    "role": msg.role.rawValue,
                    "content": msg.content
                ]
                if !msg.images.isEmpty {
                    msgDict["images"] = msg.images.map { $0.base64 }
                }
                // Include tool_call_id for tool-result messages
                if let toolCallId = msg.toolCallId {
                    msgDict["tool_call_id"] = toolCallId
                }
                // Include tool_calls for assistant messages that requested tools
                if let toolCalls = msg.toolCalls, !toolCalls.isEmpty {
                    msgDict["tool_calls"] = toolCalls.map { tc -> [String: Any] in
                        [
                            "id": tc.id,
                            "type": "function",
                            "function": [
                                "name": tc.name,
                                // Ollama expects assistant loop-closure arguments as a JSON object/array,
                                // not a JSON-encoded string. Sending the string form triggers HTTP 400:
                                // "Value looks like object, but can't find closing '}' symbol".
                                "arguments": Self.jsonValueOrString(from: tc.arguments)
                            ]
                        ]
                    }
                }
                return msgDict
            },
            "stream": stream
        ]

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
        }

        // Add options if any
        var options: [String: Any] = [:]
        if let temp = request.temperature { options["temperature"] = temp }
        if let topP = request.topP { options["top_p"] = topP }
        if let maxTokens = request.maxTokens { options["num_predict"] = maxTokens }
        if !options.isEmpty {
            bodyDict["options"] = options
        }

        let bodyData = try JSONSerialization.data(withJSONObject: bodyDict, options: [])
        urlRequest.httpBody = bodyData
        Self.debugLogRequest(urlRequest, body: bodyData)
        return urlRequest
    }

    /// Ollama returns tool-call arguments to us as a provider-agnostic JSON string,
    /// but expects assistant `tool_calls.function.arguments` to be sent back as
    /// native JSON when closing the tool loop.
    private static func jsonValueOrString(from jsonString: String) -> Any {
        guard let data = jsonString.data(using: .utf8),
              let value = try? JSONSerialization.jsonObject(with: data),
              JSONSerialization.isValidJSONObject(value)
        else {
            return jsonString
        }
        return value
    }

    private static func debugLogRequest(_ request: URLRequest, body: Data) {
        let env = ProcessInfo.processInfo.environment
        guard env["LLM_PROVIDERKIT_DEBUG_HTTP"] == "1" || env["LLM_PROVIDERKIT_DEBUG_HTTP"] == "true" else {
            return
        }

        let method = request.httpMethod ?? "<method>"
        let url = request.url?.absoluteString ?? "<url>"
        let bodyText = String(data: body, encoding: .utf8) ?? "<non-UTF8 body: \(body.count) bytes>"
        print("""
        \n========== LLMProviderKit HTTP Request ==========
        Provider: \(Self.name)
        \(method) \(url)
        Body bytes: \(body.count)
        Body:
        \(bodyText)
        =========================================\n
        """)
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

        // Parse tool calls from stream chunks (same format as non-streaming).
        // Ollama streams incremental JSON objects; when tool_calls are present
        // we surface them via the finish chunk's reason so consumers know to
        // collect them from the final non-streaming response, matching the
        // pattern used by the OpenAI provider's streaming path.
        if let toolCalls = decoded.message?.toolCalls, !toolCalls.isEmpty, decoded.done != true {
            // Tool calls appeared mid-stream; mark the finish reason accordingly.
            // The full tool call details are carried in the final accumulated response.
            chunks.append(.finish(reason: .toolCalls, usage: nil))
        }

        if decoded.done == true {
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

        // Ollama (notably its cloud proxy) can return HTTP 200 with an error
        // body, or a degenerate `done` body with NO message at all (observed
        // live: sub-second responses under concurrent load, no content /
        // thinking / tool_calls). Both must throw — silently decoding them as
        // an empty success makes agent loops treat a server failure as "the
        // model said nothing".
        if let apiError = decoded.error, !apiError.isEmpty {
            throw LLMError.providerError(apiError)
        }
        guard let message = decoded.message else {
            throw LLMError.invalidResponse(
                "Ollama returned a body with no message (done_reason: \(decoded.doneReason ?? "nil")) — transient server/proxy failure, retry")
        }

        // `done_reason:"load"` is a model-LOAD acknowledgement, not a
        // completion: under concurrent load the Ollama cloud proxy returns
        // these (empty content, no tool calls, ~µs duration) instead of
        // generating. Captured live from glm-5.2:cloud. Throw so callers retry
        // — decoding it as an empty answer is the root cause of sub-agents
        // "returning no answer" under parallel delegation.
        if decoded.doneReason == "load",
           (message.content ?? "").isEmpty,
           (message.toolCalls ?? []).isEmpty {
            throw LLMError.invalidResponse(
                "Ollama returned a model-load response (done_reason: \"load\") with no generation — transient, retry")
        }

        let text = message.content ?? ""

        // Parse native tool calls from the response
        let toolCalls: [LLMToolCall] = message.toolCalls?.map { tc in
            LLMToolCall(
                id: tc.id ?? UUID().uuidString,
                name: tc.function?.name ?? "",
                arguments: tc.function?.arguments ?? "{}"
            )
        } ?? []

        let usage = LLMUsage(
            promptTokens: decoded.promptEvalCount,
            completionTokens: decoded.evalCount,
            totalTokens: (decoded.promptEvalCount ?? 0) + (decoded.evalCount ?? 0)
        )

        let finishReason: LLMFinishReason = toolCalls.isEmpty ? .stop : .toolCalls

        let thinking = message.thinking
        return LLMResponse(
            text: text,
            reasoning: (thinking?.isEmpty == false) ? thinking : nil,
            finishReason: finishReason,
            usage: usage,
            toolCalls: toolCalls,
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
                capabilities: Self.capabilities(for: model),
                categories: Self.categories(for: model)
            )
        }
    }

    private static func capabilities(for model: OllamaTagResponse.Model) -> Set<LLMModelCapability> {
        let lowercased = model.name.lowercased()
        if lowercased.contains("embed") || lowercased.contains("nomic") || lowercased.contains("mxbai") {
            return [.embeddings]
        }

        var caps: Set<LLMModelCapability> = [.chat, .textGeneration, .streaming]
        if let capabilities = model.capabilities {
            for capability in capabilities {
                switch capability {
                case "vision": caps.formUnion([.vision, .imageInput])
                case "tools": caps.insert(.tools)
                case "completion": break // already covered by chat/textGeneration
                case "thinking": caps.insert(.reasoning)
                default: break
                }
            }
        }
        if lowercased.contains("llava") || lowercased.contains("bakllava") || lowercased.contains("vision") {
            caps.formUnion([.vision, .imageInput])
        }
        if lowercased.contains("qwen") || lowercased.contains("llama") || lowercased.contains("mistral") {
            caps.insert(.tools)
        }
        return caps
    }

    private static func categories(for model: OllamaTagResponse.Model) -> Set<LLMModelCategory> {
        let caps = capabilities(for: model)
        var categories: Set<LLMModelCategory> = []
        if !caps.intersection([.chat, .textGeneration, .tools, .reasoning]).isEmpty { categories.insert(.text) }
        if !caps.intersection([.vision, .imageInput]).isEmpty { categories.formUnion([.vision, .multimodal]) }
        if caps.contains(.embeddings) { categories.insert(.embedding) }
        return categories
    }
}

// MARK: - Ollama chat API types (response only; request is built as a dict)

private struct OllamaChatResponse: Decodable {
    struct Message: Decodable {
        let role: String?
        let content: String?
        let toolCalls: [OllamaToolCall]?
        // Reasoning models (GLM/Kimi/DeepSeek) return separated reasoning here.
        // Surfaced as `LLMResponse.reasoning` — never merged into the answer.
        let thinking: String?

        enum CodingKeys: String, CodingKey {
            case role
            case content
            case toolCalls = "tool_calls"
            case thinking
        }
    }

    let model: String?
    let createdAt: String?
    let message: Message?
    let done: Bool?
    let doneReason: String?
    /// Ollama (and its cloud proxy) can return HTTP 200 with an error body.
    let error: String?
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
        case error
        case done
        case doneReason = "done_reason"
        case totalDuration = "total_duration"
        case loadDuration = "load_duration"
        case promptEvalCount = "prompt_eval_count"
        case promptEvalDuration = "prompt_eval_duration"
        case evalCount = "eval_count"
        case evalDuration = "eval_duration"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.model = try container.decodeIfPresent(String.self, forKey: .model)
        self.createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt)
        self.message = try container.decodeIfPresent(Message.self, forKey: .message)
        self.error = try container.decodeIfPresent(String.self, forKey: .error)
        self.done = try container.decodeIfPresent(Bool.self, forKey: .done)
        self.totalDuration = try container.decodeIfPresent(Int64.self, forKey: .totalDuration)
        self.loadDuration = try container.decodeIfPresent(Int64.self, forKey: .loadDuration)
        self.promptEvalCount = try container.decodeIfPresent(Int.self, forKey: .promptEvalCount)
        self.promptEvalDuration = try container.decodeIfPresent(Int64.self, forKey: .promptEvalDuration)
        self.evalCount = try container.decodeIfPresent(Int.self, forKey: .evalCount)
        self.evalDuration = try container.decodeIfPresent(Int64.self, forKey: .evalDuration)
        self.doneReason = try container.decodeIfPresent(String.self, forKey: .doneReason)
    }
}

private struct OllamaToolCall: Decodable {
    let id: String?
    let function: OllamaToolCallFunction?
}

private struct OllamaToolCallFunction: Decodable {
    let name: String?
    // Ollama returns arguments as either a JSON string or a JSON object.
    // We handle both by decoding into an `AnyCodableJSON` wrapper.
    let arguments: String?

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)

        // Try string first (some models return a JSON string)
        if let str = try? container.decode(String.self, forKey: .arguments) {
            self.arguments = str
            return
        }

        // Otherwise it's a JSON object — serialize it to a string
        if let dict = try? container.decode([String: AnyCodableJSON].self, forKey: .arguments) {
            let data = try JSONEncoder().encode(dict)
            self.arguments = String(data: data, encoding: .utf8) ?? "{}"
            return
        }

        self.arguments = "{}"
    }

    enum CodingKeys: String, CodingKey {
        case name, arguments, index
    }
}

/// Helper to decode arbitrary JSON values (dict, array, primitives).
private struct AnyCodableJSON: Decodable {
    let value: Any

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let v = try? container.decode(Bool.self) { self.value = v }
        else if let v = try? container.decode(Int.self) { self.value = v }
        else if let v = try? container.decode(Double.self) { self.value = v }
        else if let v = try? container.decode(String.self) { self.value = v }
        else if let v = try? container.decode([AnyCodableJSON].self) { self.value = v.map(\.value) }
        else if let v = try? container.decode([String: AnyCodableJSON].self) { self.value = v.mapValues(\.value) }
        else { self.value = NSNull() }
    }
}

extension AnyCodableJSON: Encodable {
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case is NSNull: try container.encodeNil()
        case let v as Bool: try container.encode(v)
        case let v as Int: try container.encode(v)
        case let v as Double: try container.encode(v)
        case let v as String: try container.encode(v)
        case let v as [Any]: try container.encode(v.map { AnyCodableJSON($0) })
        case let v as [String: Any]: try container.encode(v.mapValues { AnyCodableJSON($0) })
        default: try container.encodeNil()
        }
    }

    init(_ v: Any) { self.value = v }
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
    public static let suggestedModels: [LLMModelInfo] = [
        LLMModelInfo(
            id: OllamaModel.llama3_2,
            providerName: name,
            displayName: "Llama 3.2",
            capabilities: [.chat, .textGeneration, .streaming, .tools],
            categories: [.text],
            releaseStage: .stable
        ),
        LLMModelInfo(
            id: OllamaModel.llama3_1,
            providerName: name,
            displayName: "Llama 3.1",
            capabilities: [.chat, .textGeneration, .streaming, .tools],
            categories: [.text],
            releaseStage: .stable
        ),
        LLMModelInfo(
            id: OllamaModel.qwen2_5,
            providerName: name,
            displayName: "Qwen 2.5",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .reasoning],
            categories: [.text],
            releaseStage: .stable
        ),
        LLMModelInfo(
            id: OllamaModel.nomicEmbedText,
            providerName: name,
            displayName: "Nomic Embed Text",
            capabilities: [.embeddings],
            categories: [.embedding],
            releaseStage: .stable
        ),
    ]

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
