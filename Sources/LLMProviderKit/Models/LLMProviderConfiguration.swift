import Foundation

/// Conformances represent the configuration needed to talk to a provider.
///
/// Each provider exposes a human-readable name and a base URL, plus helpers to
/// build requests and parse responses in its own format.
public struct LLMProviderConfiguration: Sendable {
    /// Provider identifier, e.g. `ollama`, `openai`, `gemini`.
    public let name: String

    /// Base URL for all API calls. Usually includes the major API version path.
    public let baseURL: URL

    /// Optional default authorization/token.
    public let apiKey: String?

    /// Default model identifier for this configuration.
    public let defaultModel: String?

    public init(
        name: String,
        baseURL: URL,
        apiKey: String? = nil,
        defaultModel: String? = nil
    ) {
        self.name = name
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.defaultModel = defaultModel
    }
}
