import Foundation

/// A role in a chat conversation.
///
/// Providers map this to their own enum names internally.
public enum LLMMessageRole: String, Sendable, Codable {
    case system
    case user
    case assistant
    /// Some providers (e.g. Gemini, Anthropic) support an explicit `tool` role.
    case tool
}

/// A single message in a chat conversation.
public struct LLMMessage: Sendable, Equatable {
    public let role: LLMMessageRole
    public let content: String
    public var images: [LLMImage]
    /// Correlates a tool-result message with the original tool call.
    /// Only set when `role == .tool`.
    public var toolCallId: String?
    /// Tool calls made by the assistant. Only set when `role == .assistant`.
    /// Providers serialize these into the provider-specific format so the
    /// model can correlate subsequent tool results with the original calls.
    public var toolCalls: [LLMToolCall]?

    public init(role: LLMMessageRole, content: String, images: [LLMImage] = [], toolCallId: String? = nil, toolCalls: [LLMToolCall]? = nil) {
        self.role = role
        self.content = content
        self.images = images
        self.toolCallId = toolCallId
        self.toolCalls = toolCalls
    }

    public static func system(_ content: String) -> Self {
        LLMMessage(role: .system, content: content)
    }

    public static func user(_ content: String) -> Self {
        LLMMessage(role: .user, content: content)
    }

    public static func user(_ content: String, images: [LLMImage]) -> Self {
        LLMMessage(role: .user, content: content, images: images)
    }

    public static func assistant(_ content: String) -> Self {
        LLMMessage(role: .assistant, content: content)
    }

    /// Create an assistant message with tool calls.
    public static func assistant(content: String = "", toolCalls: [LLMToolCall]) -> Self {
        LLMMessage(role: .assistant, content: content, toolCalls: toolCalls)
    }

    /// Create a tool-result message with the tool call ID for correlation.
    public static func tool(_ content: String, toolCallId: String) -> Self {
        LLMMessage(role: .tool, content: content, toolCallId: toolCallId)
    }
}

/// An image payload attached to a message for vision-capable models.
///
/// Stores raw bytes and a MIME type. Providers encode this as base64
/// in their provider-specific format. Does **not** import UIKit/AppKit.
public struct LLMImage: Sendable, Equatable, Codable {
    /// Raw image bytes (PNG, JPEG, etc.).
    public let data: Data

    /// MIME type, e.g. `"image/png"`, `"image/jpeg"`.
    public let mimeType: String

    public init(data: Data, mimeType: String = "image/png") {
        self.data = data
        self.mimeType = mimeType
    }

    /// Base64-encoded string of the raw bytes (no data: prefix).
    public var base64: String {
        data.base64EncodedString()
    }
}

/// A chat completion request, independent of any provider.
public struct LLMRequest: Sendable {
    public var model: String
    public var messages: [LLMMessage]
    public var temperature: Double?
    public var maxTokens: Int?
    public var topP: Double?
    public var tools: [LLMToolDefinition]
    public var id: UUID

    public init(
        model: String,
        messages: [LLMMessage],
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        topP: Double? = nil,
        tools: [LLMToolDefinition] = [],
        id: UUID = UUID()
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.topP = topP
        self.tools = tools
        self.id = id
    }
}

/// A single chunk of a streaming response.
public enum LLMStreamChunk: Sendable {
    case text(String)
    case toolCall(LLMToolCall)
    case finish(reason: LLMFinishReason?, usage: LLMUsage?)
    case error(Error)
}

/// Why a response finished.
public enum LLMFinishReason: String, Sendable {
    case stop
    case length
    case contentFilter = "content_filter"
    case toolCalls = "tool_calls"
    case unknown
}

/// Token usage returned by the provider.
public struct LLMUsage: Sendable, Equatable {
    public let promptTokens: Int?
    public let completionTokens: Int?
    public let totalTokens: Int?

    public init(promptTokens: Int? = nil, completionTokens: Int? = nil, totalTokens: Int? = nil) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
    }
}

/// A complete chat response assembled from one or more stream chunks.
public struct LLMResponse: Sendable {
    public let text: String
    public let finishReason: LLMFinishReason?
    public let usage: LLMUsage?
    public let toolCalls: [LLMToolCall]
    public let request: LLMRequest
    public let providerName: String
    public let rawData: Data?

    public init(
        text: String,
        finishReason: LLMFinishReason? = nil,
        usage: LLMUsage? = nil,
        toolCalls: [LLMToolCall] = [],
        request: LLMRequest,
        providerName: String,
        rawData: Data? = nil
    ) {
        self.text = text
        self.finishReason = finishReason
        self.usage = usage
        self.toolCalls = toolCalls
        self.request = request
        self.providerName = providerName
        self.rawData = rawData
    }
}

// MARK: - Tool Calling

/// A tool definition sent to the LLM so the model can call it.
public struct LLMToolDefinition: @unchecked Sendable {
    public let name: String
    public let description: String
    public let parameters: [String: Any]

    public init(name: String, description: String, parameters: [String: Any]) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    public static func == (lhs: LLMToolDefinition, rhs: LLMToolDefinition) -> Bool {
        lhs.name == rhs.name && lhs.description == rhs.description
    }
}

/// A tool call returned by the LLM in its response.
public struct LLMToolCall: Sendable, Equatable, Codable {
    public let id: String
    public let name: String
    public let arguments: String

    public init(id: String = UUID().uuidString, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    public func decodedArguments() -> [String: Any]? {
        guard let data = arguments.data(using: .utf8) else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }
}

// MARK: - Internal helpers

extension LLMMessage: Codable {
    enum CodingKeys: String, CodingKey {
        case role
        case content
        case images
        case toolCallId
        case toolCalls
    }
}

extension LLMFinishReason: Codable {}
extension LLMUsage: Codable {}