import Foundation

/// Every declared effort vocabulary, in one place, overridable from a file.
///
/// These sets rot: a vendor ships a model, or changes what an existing one
/// accepts, and a level that used to work starts answering HTTP 400. Compiled
/// into the binary, the fix for that is a release. Here it is also a file, so
/// a wrong row can be corrected on a machine that already has the app.
///
/// The built-in rows are the floor and always present; an override file is
/// consulted first. That ordering is deliberate — a bad or stale override can
/// only change what is offered, never remove the app's ability to run.
///
/// **A vocabulary belongs to a model served by an endpoint, not to a model.**
/// The same model answers differently through different gateways: GLM-5.3
/// publishes low/high/max on ollama.com and accepts low/medium/high/max
/// natively; `ultra` is accepted on OpenAI's ChatGPT-subscription surface and
/// absent from the public API documentation for the same model. Hence the key
/// is (provider, model) and never a model alone.
///
/// When a wire rejects a level, correct the row — never add a condition to a
/// dispatch path.
public final class EffortCatalog: @unchecked Sendable {

    /// One rule: what a model accepts, plus the vendor mappings that the
    /// ladder's arithmetic would get wrong.
    public struct Rule: Sendable, Equatable, Codable {
        public var supported: [LLMReasoningEffort]
        public var overrides: [LLMReasoningEffort: LLMReasoningEffort]

        public init(supported: [LLMReasoningEffort],
                    overrides: [LLMReasoningEffort: LLMReasoningEffort] = [:]) {
            self.supported = supported
            self.overrides = overrides
        }

        var vocabulary: LLMEffortVocabulary {
            LLMEffortVocabulary(supported: supported, overrides: overrides)
        }

        // Written by hand because Swift encodes an enum-keyed dictionary as a
        // flat array of alternating keys and values, which is not a file a
        // person can edit. `overrides` is a plain JSON object here, exactly as
        // documented, and an unrecognised level name fails the decode rather
        // than being dropped — a silently ignored typo is worse than an error.
        private enum CodingKeys: String, CodingKey { case supported, overrides }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            supported = try c.decode([LLMReasoningEffort].self, forKey: .supported)
            let raw = try c.decodeIfPresent([String: String].self, forKey: .overrides) ?? [:]
            var mapped: [LLMReasoningEffort: LLMReasoningEffort] = [:]
            for (from, to) in raw {
                guard let key = LLMReasoningEffort(rawValue: from),
                      let value = LLMReasoningEffort(rawValue: to) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .overrides, in: c,
                        debugDescription: "\(from) → \(to) names a level that does not exist")
                }
                mapped[key] = value
            }
            overrides = mapped
        }

        public func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(supported, forKey: .supported)
            try c.encode(Dictionary(uniqueKeysWithValues:
                overrides.map { ($0.key.rawValue, $0.value.rawValue) }), forKey: .overrides)
        }
    }

    /// A model id, or a prefix when it ends in `*`. Exact wins over prefix, and
    /// a longer prefix wins over a shorter one, so a specific row can always
    /// carve an exception out of a general one.
    public struct Key: Hashable, Sendable {
        public let provider: String
        public let pattern: String
        public init(provider: String, pattern: String) {
            self.provider = provider.lowercased()
            self.pattern = pattern.lowercased()
        }
    }

    public static let shared = EffortCatalog()

    private let lock = NSLock()
    private var builtIn: [Key: Rule] = [:]
    private var overrides: [Key: Rule] = [:]

    private init() { builtIn = Self.declaredRules }

    // MARK: - Lookup

    /// The vocabulary for a model on a provider, or nil when nothing declares
    /// one — in which case no level is sent and the model keeps its own
    /// default. Failing closed is the safe direction: several wires answer an
    /// unsupported level with 400.
    public func vocabulary(provider: String, model: String) -> LLMEffortVocabulary? {
        lock.lock(); defer { lock.unlock() }
        // Both spellings, full id first. On some providers the vendor prefix IS
        // the signal — an OpenRouter id is `vendor/model`, and a row keyed on
        // `anthropic*` must still match `anthropic/claude-sonnet-4`. On others
        // the id arrives tagged (`glm-5.2:cloud`) and only the slug matches.
        for id in [model.lowercased(), Self.slug(model)] {
            if let rule = match(id, provider: provider, in: overrides) { return rule.vocabulary }
            if let rule = match(id, provider: provider, in: builtIn) { return rule.vocabulary }
        }
        return nil
    }

    private func match(_ id: String, provider: String, in table: [Key: Rule]) -> Rule? {
        let provider = provider.lowercased()
        if let exact = table[Key(provider: provider, pattern: id)] { return exact }
        return table
            .filter { $0.key.provider == provider
                      && $0.key.pattern.hasSuffix("*")
                      && id.hasPrefix(String($0.key.pattern.dropLast())) }
            // Longest prefix wins, so `glm-5.3*` beats `glm*`.
            .max { $0.key.pattern.count < $1.key.pattern.count }?.value
    }

    /// Strip a `vendor/` prefix and any `:tag` — the shapes model ids arrive in.
    public static func slug(_ model: String) -> String {
        let bare = model.lowercased().split(separator: "/").last.map(String.init)
            ?? model.lowercased()
        return bare.split(separator: ":").first.map(String.init) ?? bare
    }

    // MARK: - Overrides

    /// Replace the override layer. The built-in rows are untouched, so a file
    /// that is wrong, stale, or deleted degrades to shipped behaviour.
    public func applyOverrides(_ rules: [Key: Rule]) {
        lock.lock(); overrides = rules; lock.unlock()
    }

    /// Load overrides from JSON:
    ///
    /// ```json
    /// { "ollama": { "glm-5.4*": { "supported": ["low","high","max"],
    ///                             "overrides": {"medium":"high"} } } }
    /// ```
    ///
    /// A malformed file throws and changes nothing — the app keeps running on
    /// its built-in rows rather than losing the feature to a typo.
    @discardableResult
    public func loadOverrides(from url: URL) throws -> Int {
        let wire = try JSONDecoder().decode([String: [String: Rule]].self,
                                            from: Data(contentsOf: url))
        var rules: [Key: Rule] = [:]
        for (provider, models) in wire {
            for (pattern, rule) in models {
                rules[Key(provider: provider, pattern: pattern)] = rule
            }
        }
        applyOverrides(rules)
        return rules.count
    }

    /// Test hook.
    public func _resetOverrides() { applyOverrides([:]) }

    // MARK: - Declared rows

    private static func rules(_ provider: String,
                              _ rows: [(String, [LLMReasoningEffort], [LLMReasoningEffort: LLMReasoningEffort])])
    -> [Key: Rule] {
        var out: [Key: Rule] = [:]
        for (pattern, supported, overrides) in rows {
            out[Key(provider: provider, pattern: pattern)] =
                Rule(supported: supported, overrides: overrides)
        }
        return out
    }

    /// Everything we have evidence for. Sources are recorded in Naseem's
    /// docs/research/reasoning-effort.md with the date each was read.
    private static let declaredRules: [Key: Rule] = {
        var all: [Key: Rule] = [:]

        // Anthropic — five levels, uniform across every model that takes one.
        // `high` is the API default and identical to omitting the field.
        let anthropic: [LLMReasoningEffort] = [.low, .medium, .high, .xhigh, .max]
        all.merge(rules("anthropic", [
            ("claude-fable-5*", anthropic, [:]), ("claude-mythos-5*", anthropic, [:]),
            ("claude-mythos-preview*", anthropic, [:]), ("claude-opus-5*", anthropic, [:]),
            ("claude-opus-4-8*", anthropic, [:]), ("claude-opus-4-7*", anthropic, [:]),
            ("claude-opus-4-6*", anthropic, [:]), ("claude-opus-4-5*", anthropic, [:]),
            ("claude-sonnet-5*", anthropic, [:]), ("claude-sonnet-4-6*", anthropic, [:]),
        ])) { a, _ in a }

        // OpenRouter — widest OpenAI-compatible vocabulary; it normalizes per
        // model itself, so the rows only say which vendors take the parameter.
        let openRouter: [LLMReasoningEffort] = [.off, .minimal, .low, .medium, .high, .xhigh, .max]
        all.merge(rules("openrouter", [
            ("deepseek*", openRouter, [:]), ("anthropic*", openRouter, [:]),
            ("openai*", openRouter, [:]), ("x-ai*", openRouter, [:]),
            ("google*", openRouter, [:]), ("qwen*", openRouter, [:]),
            ("z-ai*", openRouter, [:]), ("moonshotai*", openRouter, [:]),
        ])) { a, _ in a }

        // Ollama, native /api/chat — `think` as a level string. No xhigh.
        let ollamaLocal: [LLMReasoningEffort] = [.low, .medium, .high, .max]
        // Ollama Cloud, OpenAI-compatible — rejects `minimal` with a 400.
        let ollamaCloud: [LLMReasoningEffort] = [.off, .low, .medium, .high, .max]
        let toMax: [LLMReasoningEffort: LLMReasoningEffort] = [.xhigh: .max]
        for provider in ["ollama", "ollamacloud"] {
            let base = provider == "ollama" ? ollamaLocal : ollamaCloud
            all.merge(rules(provider, [
                // GLM-5.2's knob is exactly its minimum thinking level and its
                // top tier; 5.3 widens it to a graded scale.
                ("glm-5.2*", [.high, .max], toMax),
                ("glm-5-2*", [.high, .max], toMax),
                ("glm-5.3*", [.low, .medium, .high, .max], toMax),
                ("glm-5-3*", [.low, .medium, .high, .max], toMax),
                ("glm*", base, toMax),
                ("deepseek-v4*", [.low, .medium, .high, .max], toMax),
                ("deepseek*", base, toMax),
                // Kimi K3 has no medium, and its `high` is both the positional
                // middle and the server default — so medium belongs at high.
                ("k3*", [.low, .high, .max], [.medium: .high, .xhigh: .max]),
                ("kimi-k3*", [.low, .high, .max], [.medium: .high, .xhigh: .max]),
                ("kimi*", [.low, .medium, .high], [:]),
            ])) { a, _ in a }
        }

        // OpenAI's ChatGPT-subscription surface. The endpoint states its own
        // vocabulary in the 400 it returns for anything else:
        //
        //   Invalid value: 'ultra'. Supported values are: 'none', 'minimal',
        //   'low', 'medium', 'high', 'xhigh', and 'max'.
        //   param: reasoning.effort        (observed live on gpt-5.6-sol, 2026-09-15)
        //
        // These rows were first transcribed from OpenAI's own Codex catalog
        // (codex-rs/models-manager/models.json), which lists `ultra` for five
        // models and caps gpt-5.5/5.4 at xhigh. The server honoured neither:
        // `ultra` is refused, and the message names one set for the parameter
        // rather than one per model. The catalog describes what the Codex CLI
        // offers — plan tiers included — not what this endpoint accepts.
        //
        // A server's own rejection outranks a vendor's catalog. Hermes had it
        // right ("ultra is the Codex product tier: no wire accepts it") and was
        // clearly written from the same 400.
        let codex: [LLMReasoningEffort] = [.off, .minimal, .low, .medium, .high, .xhigh, .max]
        all.merge(rules("chatgptcodex", [
            ("gpt-6-astra*", codex, [:]), ("gpt-5.6-sol*", codex, [:]),
            ("gpt-5.6-terra*", codex, [:]), ("gpt-5.6-luna*", codex, [:]),
            ("gpt-daybreak-blue*", codex, [:]), ("gpt-daybreak-red*", codex, [:]),
            ("codex-auto-review*", codex, [:]),
            ("gpt-5.5*", codex, [:]), ("gpt-5.4*", codex, [:]),
        ])) { a, _ in a }

        return all
    }()
}
