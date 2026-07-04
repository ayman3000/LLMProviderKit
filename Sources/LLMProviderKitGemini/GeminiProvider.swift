import Foundation
import LLMProviderKit

/// Provider for Google Gemini (`https://generativelanguage.googleapis.com/v1beta`).
///
/// Gemini exposes `generateContent` for non-streaming and
/// `streamGenerateContent` for streaming. The content shape is a flat list of
/// "parts" that we map from/to `LLMMessage`.
///
/// Supports native tool calling via `functionDeclarations` (tools in the request)
/// and `functionCall` parts (tool calls in the response).
public struct GeminiProvider: LLMProvider {
    public static let name: String = "gemini"
    private static let thoughtSignatureMetadataKey = "gemini.thoughtSignature"

    public let configuration: LLMProviderConfiguration
    public let urlSession: URLSession

    public init(configuration: LLMProviderConfiguration, urlSession: URLSession = .shared) {
        self.configuration = configuration
        self.urlSession = urlSession
    }

    public func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        let action = stream ? "streamGenerateContent" : "generateContent"
        let model = Self.normalizedModelID(request.model)
        let path = "models/\(model):\(action)"

        guard var components = URLComponents(url: configuration.baseURL, resolvingAgainstBaseURL: true) else {
            throw LLMError.invalidRequest("Invalid Gemini base URL")
        }
        components.path = (components.path as NSString).appendingPathComponent(path)

        var queryItems: [URLQueryItem] = []
        if let apiKey = configuration.apiKey {
            queryItems.append(URLQueryItem(name: "key", value: apiKey))
        }
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

        // Build body as a dictionary (handles [String: Any] tool parameters natively)
        var bodyDict: [String: Any] = [:]

        // Build contents with parts (text, images, function calls, function responses)
        var contents: [[String: Any]] = []
        var systemInstruction: String?

        for message in request.messages {
            if message.role == .system {
                // Gemini has no system role in contents; use systemInstruction
                systemInstruction = message.content
                continue
            }

            let geminiRole = Self.geminiRole(for: message.role)
            var parts: [[String: Any]] = []

            // Text content
            if !message.content.isEmpty {
                parts.append(["text": message.content])
            }

            // Images
            for img in message.images {
                parts.append([
                    "inlineData": [
                        "mimeType": img.mimeType,
                        "data": img.base64
                    ]
                ])
            }

            // Tool calls (assistant messages with function calls)
            if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
                for tc in toolCalls {
                    var args: [String: Any] = [:]
                    if let decoded = tc.decodedArguments() {
                        args = decoded
                    }
                    var functionCallPart: [String: Any] = [
                        "functionCall": [
                            "name": tc.name,
                            "args": args
                        ]
                    ]
                    if let thoughtSignature = tc.providerMetadata[Self.thoughtSignatureMetadataKey] {
                        functionCallPart["thoughtSignature"] = thoughtSignature
                    }
                    parts.append(functionCallPart)
                }
            }

            // Tool results (tool messages → function response)
            if message.role == .tool {
                // Gemini expects functionResponse in parts; use the toolCallId
                // (which carries the function name from the originating call)
                // as the name so the model can correlate the response.
                let responseName = message.toolCallId ?? "tool_result"
                // Try to parse the content as JSON; if it fails, wrap as a dict.
                var responseObj: [String: Any] = [:]
                if let data = message.content.data(using: .utf8),
                   let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    responseObj = parsed
                } else {
                    responseObj = ["result": message.content]
                }
                parts.append([
                    "functionResponse": [
                        "name": responseName,
                        "response": responseObj
                    ]
                ])
            }

            if !parts.isEmpty {
                contents.append([
                    "role": geminiRole,
                    "parts": parts
                ])
            }
        }

        bodyDict["contents"] = contents

        // System instruction
        if let sys = systemInstruction {
            bodyDict["systemInstruction"] = ["parts": [["text": sys]]]
        }

        // Generation config
        var genConfig: [String: Any] = [:]
        if let temp = request.temperature { genConfig["temperature"] = temp }
        if let topP = request.topP { genConfig["topP"] = topP }
        if let maxTokens = request.maxTokens { genConfig["maxOutputTokens"] = maxTokens }
        if !genConfig.isEmpty {
            bodyDict["generationConfig"] = genConfig
        }

        // Tools (function declarations)
        if !request.tools.isEmpty {
            let functionDeclarations = request.tools.map { tool -> [String: Any] in
                [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                ]
            }
            bodyDict["tools"] = [["functionDeclarations": functionDeclarations]]
        }

        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: bodyDict, options: [])
        return urlRequest
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
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
        var chunks: [LLMStreamChunk] = []

        // Text and tool calls from parts
        let parts = decoded.candidates?.first?.content?.parts ?? []
        var text = ""
        for part in parts {
            if let partText = part.text {
                text += partText
            }
            if let funcCall = part.functionCall {
                let argsData = try? JSONSerialization.data(withJSONObject: funcCall.args ?? [:], options: [])
                let argsString = argsData.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                let toolCall = LLMToolCall(
                    id: funcCall.name ?? UUID().uuidString,
                    name: funcCall.name ?? "",
                    arguments: argsString,
                    providerMetadata: Self.providerMetadata(for: part)
                )
                chunks.append(.toolCall(toolCall))
            }
        }

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
            // Map STOP→.stop unless we already emitted tool calls (→ .toolCalls)
            let mappedReason: LLMFinishReason? = finishReason.map {
                switch $0 {
                case "STOP": return chunks.contains(where: { chunk in
                    if case .toolCall = chunk { return true }
                    return false
                }) ? .toolCalls : .stop
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

        // Extract text and tool calls from parts
        var text = ""
        var toolCalls: [LLMToolCall] = []

        if let parts = decoded.candidates?.first?.content?.parts {
            for part in parts {
                if let partText = part.text {
                    text += partText
                }
                // Check for function call
                if let funcCall = part.functionCall {
                    let argsData = try? JSONSerialization.data(withJSONObject: funcCall.args ?? [:], options: [])
                    let argsString = argsData.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                    toolCalls.append(LLMToolCall(
                        id: funcCall.name ?? UUID().uuidString,
                        name: funcCall.name ?? "",
                        arguments: argsString,
                        providerMetadata: Self.providerMetadata(for: part)
                    ))
                }
            }
        }

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

        let curatedByID = Dictionary(uniqueKeysWithValues: Self.curatedModels.map { ($0.id, $0) })
        let decoded = try JSONDecoder().decode(GeminiModelsResponse.self, from: data)
        return decoded.models.map { model in
            // Gemini API returns names like "models/gemini-3.5-flash"
            // Strip the "models/" prefix to get the clean model ID
            let cleanId = model.name.replacingOccurrences(of: "models/", with: "")
            return LLMModelInfo(
                id: cleanId,
                providerName: Self.name,
                displayName: model.displayName ?? cleanId,
                contextWindow: model.inputTokenLimit,
                capabilities: Self.capabilities(for: model, cleanId: cleanId),
                categories: Self.categories(for: model, cleanId: cleanId)
            ).enriched(with: curatedByID[cleanId])
        }
    }

    private static func normalizedModelID(_ model: String) -> String {
        model.hasPrefix("models/") ? String(model.dropFirst("models/".count)) : model
    }

    private static func providerMetadata(for part: GeminiResponse.Candidate.Content.Part) -> [String: String] {
        guard let thoughtSignature = part.thoughtSignature else {
            return [:]
        }
        return [thoughtSignatureMetadataKey: thoughtSignature]
    }

    private static func geminiRole(for role: LLMMessageRole) -> String {
        switch role {
        case .system: return "user"
        case .user: return "user"
        case .assistant: return "model"
        case .tool: return "model" // Gemini uses "model" role for function responses
        }
    }

    private static func capabilities(for model: GeminiModelsResponse.Model, cleanId: String) -> Set<LLMModelCapability> {
        let lowercased = cleanId.lowercased()
        if lowercased.contains("embed") {
            return [.embeddings]
        }

        var caps: Set<LLMModelCapability> = [.chat, .textGeneration, .streaming]
        if let supported = model.supportedGenerationMethods {
            for method in supported {
                switch method {
                case "generateContent":
                    caps.formUnion([.chat, .textGeneration, .tools])
                case "countTokens": break
                case "embedContent": caps.insert(.embeddings)
                default: break
                }
            }
        }
        if lowercased.contains("gemini") {
            caps.formUnion([.vision, .imageInput, .structuredOutput])
        }
        if lowercased.contains("pro") || lowercased.contains("3.") {
            caps.insert(.reasoning)
        }
        return caps
    }

    private static func categories(for model: GeminiModelsResponse.Model, cleanId: String) -> Set<LLMModelCategory> {
        let caps = capabilities(for: model, cleanId: cleanId)
        var categories: Set<LLMModelCategory> = []
        if !caps.intersection([.chat, .textGeneration, .tools, .reasoning]).isEmpty { categories.insert(.text) }
        if !caps.intersection([.vision, .imageInput]).isEmpty { categories.formUnion([.vision, .multimodal]) }
        if caps.contains(.embeddings) { categories.insert(.embedding) }
        return categories
    }
}

// MARK: - Gemini API types

private struct GeminiResponse: Decodable {
    struct Candidate: Decodable {
        struct Content: Decodable {
            struct Part: Decodable {
                let text: String?
                let functionCall: GeminiFunctionCall?
                let thoughtSignature: String?
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
    }

    let candidates: [Candidate]?
    let usageMetadata: UsageMetadata?
}

private struct GeminiFunctionCall: Decodable {
    let name: String?
    let args: [String: Any]?

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)
        if let decoded = try container.decodeIfPresent(JSONValue.self, forKey: .args) {
            if case .object(let object) = decoded {
                self.args = object.mapValues { $0.anyValue }
            } else {
                self.args = nil
            }
        } else {
            self.args = nil
        }
    }

    enum CodingKeys: String, CodingKey {
        case name, args
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

// MARK: - Gemini models API types

private struct GeminiModelsResponse: Decodable {
    struct Model: Decodable {
        let name: String
        let displayName: String?
        let inputTokenLimit: Int?
        let supportedGenerationMethods: [String]?
    }

    let models: [Model]
}

// MARK: - Model constants

public enum GeminiModel {
    // Gemini 3.x family (current)
    public static let flash35 = "gemini-3.5-flash"
    public static let flashLite31 = "gemini-3.1-flash-lite"
    public static let pro31 = "gemini-3.1-pro"
    public static let flash30 = "gemini-3-flash"
    // Gemini 2.5 family (still available)
    public static let flash = "gemini-2.5-flash"
    public static let flashLite = "gemini-2.5-flash-lite"
    public static let pro = "gemini-2.5-pro"
}

// MARK: - Configuration presets

extension GeminiProvider {
    public static let curatedModels: [LLMModelInfo] = [
        LLMModelInfo(
            id: GeminiModel.flash35,
            providerName: name,
            displayName: "Gemini 3.5 Flash",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .reasoning, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable,
            notes: "Stable Gemini 3.x model for agentic and coding workloads."
        ),
        LLMModelInfo(
            id: GeminiModel.flashLite31,
            providerName: name,
            displayName: "Gemini 3.1 Flash-Lite",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable,
            notes: "Low-cost Gemini 3.x option."
        ),
        LLMModelInfo(
            id: GeminiModel.pro31,
            providerName: name,
            displayName: "Gemini 3.1 Pro",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .reasoning, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .preview,
            notes: "Advanced Gemini 3.x option."
        ),
        LLMModelInfo(
            id: GeminiModel.flash30,
            providerName: name,
            displayName: "Gemini 3 Flash",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .preview
        ),
        LLMModelInfo(
            id: GeminiModel.flash,
            providerName: name,
            displayName: "Gemini 2.5 Flash",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable
        ),
        LLMModelInfo(
            id: GeminiModel.flashLite,
            providerName: name,
            displayName: "Gemini 2.5 Flash-Lite",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable
        ),
        LLMModelInfo(
            id: GeminiModel.pro,
            providerName: name,
            displayName: "Gemini 2.5 Pro",
            capabilities: [.chat, .textGeneration, .streaming, .tools, .vision, .imageInput, .reasoning, .structuredOutput],
            categories: [.text, .vision, .multimodal],
            releaseStage: .stable
        ),
    ]

    public static func gemini(apiKey: String, model: String = GeminiModel.flash35) -> LLMProviderConfiguration {
        LLMProviderConfiguration(
            name: name,
            baseURL: URL(string: "https://generativelanguage.googleapis.com/v1beta")!,
            apiKey: apiKey,
            defaultModel: model
        )
    }
}
