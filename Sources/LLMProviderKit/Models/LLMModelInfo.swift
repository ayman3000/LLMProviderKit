import Foundation

/// Describes a capability a model may advertise.
public struct LLMModelCapability: RawRepresentable, Sendable, Hashable, Codable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// General text chat.
    public static let chat = LLMModelCapability(rawValue: "chat")
    /// Reasoning/thinking modes.
    public static let reasoning = LLMModelCapability(rawValue: "reasoning")
    /// Image understanding.
    public static let vision = LLMModelCapability(rawValue: "vision")
    /// Tool/function calling.
    public static let tools = LLMModelCapability(rawValue: "tools")
    /// Streaming completions.
    public static let streaming = LLMModelCapability(rawValue: "streaming")
}

/// Metadata for a single model from a provider.
public struct LLMModelInfo: Sendable, Identifiable {
    /// Provider-specific model identifier, e.g. `gpt-4o` or `llama3.2`.
    public let id: String

    /// Provider name this model belongs to, e.g. `openai`.
    public let providerName: String

    /// Human-readable name. Falls back to `id` when `nil`.
    public let displayName: String?

    /// Maximum context length, if known.
    public let contextWindow: Int?

    /// Provider-advertised capabilities.
    public let capabilities: Set<LLMModelCapability>

    public init(
        id: String,
        providerName: String,
        displayName: String? = nil,
        contextWindow: Int? = nil,
        capabilities: Set<LLMModelCapability> = []
    ) {
        self.id = id
        self.providerName = providerName
        self.displayName = displayName
        self.contextWindow = contextWindow
        self.capabilities = capabilities
    }
}

// MARK: - Codable conformance

extension LLMModelInfo: Codable {
    enum CodingKeys: String, CodingKey {
        case id
        case providerName
        case displayName
        case contextWindow
        case capabilities
    }
}
