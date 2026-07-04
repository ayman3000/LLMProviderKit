import Foundation

/// Central, offline-friendly registry/catalog for model metadata.
///
/// `LLMModelRegistry` is **opt-in**. Apps that only need to pass a model string
/// directly can ignore it entirely. Use it when you want to:
///
/// - Show provider/model pickers.
/// - Filter models by category (`text`, `vision`, `audio`, `image`, `embedding`).
/// - Filter models by capability (`tools`, `vision`, `reasoning`, etc.).
/// - Support offline usage by pre-seeding curated lists.
/// - Merge provider-discovered live models with curated metadata.
public actor LLMModelRegistry {
    /// Strategy used when refreshing or merging a provider's model list.
    public enum MergeStrategy: Sendable {
        /// Replace any previously registered models for this provider.
        case replace
        /// Keep existing models and add only new IDs.
        case append
        /// Merge by ID, preferring newly fetched model values.
        case merge
        /// Use live IDs when available, but fill missing metadata from curated records.
        case liveWithCuratedMetadata
        /// Ignore live models and use curated records only.
        case curatedOnly
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
        merge(fetched, for: providerName, strategy: strategy)
    }

    /// Merge a provided list into the registry. This is useful when you already
    /// fetched models outside the registry or want to combine curated/live lists.
    public func merge(
        _ models: [LLMModelInfo],
        for providerName: String,
        strategy: MergeStrategy = .merge
    ) {
        let existing = storage[providerName] ?? [:]
        let incoming = models.reduce(into: [:]) { dict, model in
            dict[model.id] = model
        }

        switch strategy {
        case .replace, .curatedOnly:
            storage[providerName] = incoming
        case .append:
            storage[providerName] = incoming.reduce(into: existing) { dict, pair in
                if dict[pair.key] == nil {
                    dict[pair.key] = pair.value
                }
            }
        case .merge:
            storage[providerName] = incoming.reduce(into: existing) { dict, pair in
                dict[pair.key] = pair.value
            }
        case .liveWithCuratedMetadata:
            storage[providerName] = incoming.reduce(into: existing) { dict, pair in
                dict[pair.key] = pair.value.enriched(with: existing[pair.key])
            }
        }
    }

    /// Refresh from a provider and enrich live records with an explicit curated list.
    public func refresh(
        from provider: any LLMProvider,
        curatedModels: [LLMModelInfo],
        strategy: MergeStrategy = .liveWithCuratedMetadata
    ) async throws {
        let providerName = type(of: provider).name
        let live = try await provider.availableModels()
        let merged = Self.merge(live: live, curated: curatedModels, strategy: strategy)
        register(merged, for: providerName)
    }

    /// Return all registered models for a provider, sorted by ID.
    public func models(for providerName: String) -> [LLMModelInfo] {
        storage[providerName]?.values.sorted { $0.id < $1.id } ?? []
    }

    /// Return all models across every registered provider.
    public func allModels() -> [LLMModelInfo] {
        storage.values.flatMap { $0.values }.sorted { lhs, rhs in
            if lhs.providerName == rhs.providerName { return lhs.id < rhs.id }
            return lhs.providerName < rhs.providerName
        }
    }

    /// Filter registered models by provider, category, and capabilities.
    public func models(
        providerName: String? = nil,
        category: LLMModelCategory? = nil,
        matching capabilities: Set<LLMModelCapability> = []
    ) -> [LLMModelInfo] {
        let candidates = providerName.map { models(for: $0) } ?? allModels()
        return candidates.filter { model in
            let categoryMatches = category.map { model.categories.contains($0) } ?? true
            let capabilitiesMatch = capabilities.isSubset(of: model.capabilities)
            return categoryMatches && capabilitiesMatch
        }
    }

    /// Return a single model by provider and ID, if registered.
    public func model(providerName: String, id: String) -> LLMModelInfo? {
        storage[providerName]?[id]
    }

    /// Return the default model ID for a provider.
    ///
    /// Resolution order:
    /// 1. Provider configuration's `defaultModel`.
    /// 2. First registered non-deprecated model in the registry.
    /// 3. First registered model in the registry.
    /// 4. Query the provider live via `availableModels()`.
    public func defaultModelID(
        for providerName: String,
        configuration: LLMProviderConfiguration? = nil,
        provider: (any LLMProvider)? = nil
    ) async throws -> String? {
        if let configured = configuration?.defaultModel, !configured.isEmpty { return configured }
        let registered = models(for: providerName)
        if let preferred = registered.first(where: { !$0.isDeprecated })?.id { return preferred }
        if let fallback = registered.first?.id { return fallback }
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

    public static func merge(
        live: [LLMModelInfo],
        curated: [LLMModelInfo],
        strategy: MergeStrategy
    ) -> [LLMModelInfo] {
        let curatedByID = Dictionary(uniqueKeysWithValues: curated.map { ($0.id, $0) })
        switch strategy {
        case .replace, .liveWithCuratedMetadata:
            return live.map { $0.enriched(with: curatedByID[$0.id]) }
        case .append:
            let liveIDs = Set(live.map(\.id))
            return live + curated.filter { !liveIDs.contains($0.id) }
        case .merge:
            var merged = Dictionary(uniqueKeysWithValues: curated.map { ($0.id, $0) })
            for model in live { merged[model.id] = model.enriched(with: merged[model.id]) }
            return merged.values.sorted { $0.id < $1.id }
        case .curatedOnly:
            return curated
        }
    }
}

/// More discoverable alias for new apps. Kept as a typealias so the existing
/// `LLMModelRegistry` API remains source-compatible.
public typealias LLMModelCatalog = LLMModelRegistry
