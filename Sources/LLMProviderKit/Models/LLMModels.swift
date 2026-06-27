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

    public init(role: LLMMessageRole, content: String, images: [LLMImage] = []) {
        self.role = role
        self.content = content
        self.images = images
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
    /// Model identifier as understood by the target provider.
    /// Examples: `"llama3.2"` for Ollama, `"gpt-4o"` for OpenAI,
    /// `"gemini-2.0-flash"` for Gemini.
    public var model: String

    /// Conversation history plus the new message to answer.
    public var messages: [LLMMessage]

    /// Sampling temperature. `nil` lets the provider use its default.
    public var temperature: Double?

    /// Maximum tokens to generate. `nil` lets the provider decide.
    public var maxTokens: Int?

    /// Penalize repeated tokens. `nil` uses the provider default.
    public var topP: Double?

    /// Optional unique identifier for correlating requests and streams.
    public var id: UUID

    public init(
        model: String,
        messages: [LLMMessage],
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        topP: Double? = nil,
        id: UUID = UUID()
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.topP = topP
        self.id = id
    }
}

/// A single chunk of a streaming response.
///
/// For non-streaming calls the stream emits exactly one `.text` value followed by
/// a `.finish` value.
public enum LLMStreamChunk: Sendable {
    case text(String)
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
    /// The full generated text.
    public let text: String

    /// Provider-supplied finish reason, if any.
    public let finishReason: LLMFinishReason?

    /// Provider-supplied usage, if any.
    public let usage: LLMUsage?

    /// The original request.
    public let request: LLMRequest

    /// Provider that produced this response.
    public let providerName: String

    /// Raw response data from the provider for advanced inspection.
    public let rawData: Data?

    public init(
        text: String,
        finishReason: LLMFinishReason? = nil,
        usage: LLMUsage? = nil,
        request: LLMRequest,
        providerName: String,
        rawData: Data? = nil
    ) {
        self.text = text
        self.finishReason = finishReason
        self.usage = usage
        self.request = request
        self.providerName = providerName
        self.rawData = rawData
    }
}

// MARK: - Internal helpers

extension LLMMessage: Codable {
    enum CodingKeys: String, CodingKey {
        case role
        case content
        case images
    }
}

extension LLMFinishReason: Codable {}
extension LLMUsage: Codable {}
