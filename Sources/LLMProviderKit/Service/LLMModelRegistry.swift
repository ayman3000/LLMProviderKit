import Foundation

/// Central, offline-friendly registry for model metadata.
///
/// `LLMModelRegistry` is **opt-in**. Apps that only need to pass a model string
/// directly can ignore it entirely. Use it when you want to:
///
/// - Show a picker of available models.
/// - Support offline usage by pre-seeding a curated list.
/// - Cache a provider's online model list locally.
/// - Merge developer-provided overrides with provider-discovered models.
public actor LLMModelRegistry {
    /// Strategy used when refreshing a provider's model list.
    public enum MergeStrategy: Sendable {
        /// Replace any previously registered models for this provider.
        case replace
        /// Keep existing models and add only new IDs.
        case append
        /// Merge, preferring newly fetched display names and metadata.
        case merge
    }

    private var storage: [String: [String: LLMModelInfo]] = [:]

    public init() {}

    /// Register a list of models for a provider. Useful for offline support or
    /// developer-curated lists.
    public func register(_ models: [LLMModelInfo], for providerName: String) {
        let keyed = models.reduce(into: [:]) { dict, model in
            dict[model.id] = model
        }
        storage[providerName] = keyed
    }

    /// Fetch the provider's live model list and store it.
    ///
    /// - Parameters:
    ///   - provider: The provider to query.
    ///   - strategy: How to combine fetched models with existing entries.
    public func refresh(
        from provider: any LLMProvider,
        strategy: MergeStrategy = .replace
    ) async throws {
        let providerName = type(of: provider).name
        let fetched = try await provider.availableModels()

        switch strategy {
        case .replace:
            register(fetched, for: providerName)
        case .append:
            let existing = storage[providerName] ?? [:]
            let merged = fetched.reduce(into: existing) { dict, model in
                if dict[model.id] == nil {
                    dict[model.id] = model
                }
            }
            storage[providerName] = merged
        case .merge:
            let existing = storage[providerName] ?? [:]
            let merged = fetched.reduce(into: existing) { dict, model in
                dict[model.id] = model
            }
            storage[providerName] = merged
        }
    }

    /// Return all registered models for a provider, sorted by ID.
    public func models(for providerName: String) -> [LLMModelInfo] {
        storage[providerName]?.values.sorted { $0.id < $1.id } ?? []
    }

    /// Return all models across every registered provider.
    public func allModels() -> [LLMModelInfo] {
        storage.values.flatMap { $0.values }.sorted { $0.id < $1.id }
    }

    /// Return a single model by provider and ID, if registered.
    public func model(providerName: String, id: String) -> LLMModelInfo? {
        storage[providerName]?[id]
    }

    /// Return the default model ID for a provider.
    ///
    /// Resolution order:
    /// 1. Provider configuration's `defaultModel`.
    /// 2. First registered model in the registry.
    /// 3. Query the provider live via `availableModels()` (Ollama, OpenAI, Gemini, Anthropic).
    public func defaultModelID(
        for providerName: String,
        configuration: LLMProviderConfiguration? = nil,
        provider: (any LLMProvider)? = nil
    ) async throws -> String? {
        if let configured = configuration?.defaultModel, !configured.isEmpty { return configured }
        if let registered = models(for: providerName).first?.id { return registered }
        if let provider = provider {
            let live = try await provider.availableModels()
            return live.first?.id
        }
        return nil
    }

    /// Remove all registered models for a provider.
    public func clear(providerName: String) {
        storage[providerName] = nil
    }

    /// Remove all registered models.
    public func clearAll() {
        storage.removeAll()
    }
}
