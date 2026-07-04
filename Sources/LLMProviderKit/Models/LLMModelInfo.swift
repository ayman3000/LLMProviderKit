import Foundation

/// Describes a capability a model may advertise.
///
/// This is a string-backed type instead of an enum so LLMProviderKit can add new
/// capabilities without breaking downstream exhaustive `switch` statements.
public struct LLMModelCapability: RawRepresentable, Sendable, Hashable, Codable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// General text chat.
    public static let chat = LLMModelCapability(rawValue: "chat")
    /// Text generation/completion.
    public static let textGeneration = LLMModelCapability(rawValue: "text_generation")
    /// Reasoning/thinking modes.
    public static let reasoning = LLMModelCapability(rawValue: "reasoning")
    /// Image understanding.
    public static let vision = LLMModelCapability(rawValue: "vision")
    /// Image input in multimodal prompts.
    public static let imageInput = LLMModelCapability(rawValue: "image_input")
    /// Image generation.
    public static let imageGeneration = LLMModelCapability(rawValue: "image_generation")
    /// Audio input.
    public static let audioInput = LLMModelCapability(rawValue: "audio_input")
    /// Audio generation.
    public static let audioGeneration = LLMModelCapability(rawValue: "audio_generation")
    /// Speech-to-text transcription.
    public static let speechToText = LLMModelCapability(rawValue: "speech_to_text")
    /// Text-to-speech synthesis.
    public static let textToSpeech = LLMModelCapability(rawValue: "text_to_speech")
    /// Embedding vector generation.
    public static let embeddings = LLMModelCapability(rawValue: "embeddings")
    /// Tool/function calling.
    public static let tools = LLMModelCapability(rawValue: "tools")
    /// Streaming completions.
    public static let streaming = LLMModelCapability(rawValue: "streaming")
    /// Structured output / constrained JSON output.
    public static let structuredOutput = LLMModelCapability(rawValue: "structured_output")
}

/// Broad model category intended for UI filtering.
///
/// Use `LLMModelCapability` for precise feature checks and this type for broad
/// picker sections such as Text, Vision, Image, Audio, and Embeddings.
public struct LLMModelCategory: RawRepresentable, Sendable, Hashable, Codable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public static let text = LLMModelCategory(rawValue: "text")
    public static let vision = LLMModelCategory(rawValue: "vision")
    public static let image = LLMModelCategory(rawValue: "image")
    public static let audio = LLMModelCategory(rawValue: "audio")
    public static let embedding = LLMModelCategory(rawValue: "embedding")
    public static let multimodal = LLMModelCategory(rawValue: "multimodal")
}

/// Release state for a model, when known.
public struct LLMModelReleaseStage: RawRepresentable, Sendable, Hashable, Codable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public static let stable = LLMModelReleaseStage(rawValue: "stable")
    public static let preview = LLMModelReleaseStage(rawValue: "preview")
    public static let beta = LLMModelReleaseStage(rawValue: "beta")
    public static let legacy = LLMModelReleaseStage(rawValue: "legacy")
    public static let deprecated = LLMModelReleaseStage(rawValue: "deprecated")
}

/// Metadata for a single model from a provider.
public struct LLMModelInfo: Sendable, Identifiable, Hashable, Codable {
    /// Provider-specific model identifier, e.g. `gpt-4o` or `llama3.2`.
    public let id: String

    /// Provider name this model belongs to, e.g. `openai`.
    public let providerName: String

    /// Human-readable name. Falls back to `id` when `nil`.
    public let displayName: String?

    /// Maximum context length, if known.
    public let contextWindow: Int?

    /// Provider-advertised or curated capabilities.
    public let capabilities: Set<LLMModelCapability>

    /// Broad UI-oriented categories.
    public let categories: Set<LLMModelCategory>

    /// Release stage when known.
    public let releaseStage: LLMModelReleaseStage?

    /// True when the model should not be selected for new apps.
    public let isDeprecated: Bool

    /// Optional human-readable notes for picker/tooltips/docs.
    public let notes: String?

    public init(
        id: String,
        providerName: String,
        displayName: String? = nil,
        contextWindow: Int? = nil,
        capabilities: Set<LLMModelCapability> = [],
        categories: Set<LLMModelCategory> = [],
        releaseStage: LLMModelReleaseStage? = nil,
        isDeprecated: Bool = false,
        notes: String? = nil
    ) {
        self.id = id
        self.providerName = providerName
        self.displayName = displayName
        self.contextWindow = contextWindow
        self.capabilities = capabilities
        self.categories = categories
        self.releaseStage = releaseStage
        self.isDeprecated = isDeprecated
        self.notes = notes
    }
}

extension LLMModelInfo {
    /// Returns this model enriched with metadata from a curated model record.
    /// Live values keep precedence for providerName/id, while missing metadata is
    /// filled from curated records.
    public func enriched(with curated: LLMModelInfo?) -> LLMModelInfo {
        guard let curated else { return self }
        return LLMModelInfo(
            id: id,
            providerName: providerName,
            displayName: displayName ?? curated.displayName,
            contextWindow: contextWindow ?? curated.contextWindow,
            capabilities: capabilities.isEmpty ? curated.capabilities : capabilities.union(curated.capabilities),
            categories: categories.isEmpty ? curated.categories : categories.union(curated.categories),
            releaseStage: releaseStage ?? curated.releaseStage,
            isDeprecated: isDeprecated || curated.isDeprecated,
            notes: notes ?? curated.notes
        )
    }
}
