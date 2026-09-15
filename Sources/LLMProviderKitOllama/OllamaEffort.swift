import Foundation
import LLMProviderKit

/// Declared effort vocabularies for models served by Ollama.
///
/// Two wires, deliberately separate, because the same model answers differently
/// depending on which one reaches it:
///
/// - **local** — native `/api/chat`, which takes a top-level `think` field
/// - **cloud** — `ollama.com/v1`, an OpenAI-compatible endpoint taking
///   `reasoning_effort`
///
/// Everything here is data. When a wire rejects a level, correct the set it is
/// declared in — never add a condition to the dispatch path.
///
/// Anything not named below returns nil and is sent no level at all. That is
/// the safe direction: a model keeps its own default instead of meeting a 400.
/// The honest long-term gate is `/api/show`, which reports a `thinking`
/// capability per model, but it needs a network round trip and this lookup is
/// synchronous — so until that is cached, only families we have evidence for
/// are declared.
public enum OllamaEffort {

    /// What Ollama's own `/api/chat` documents for `think` as a level string
    /// (docs read 2026-09-15): low, medium, high, max. No `xhigh`, so a request
    /// for it is mapped to the top tier rather than quietly dropped.
    static let localDefault = LLMEffortVocabulary(
        supported: [.low, .medium, .high, .max],
        overrides: [.xhigh: .max])

    /// `ollama.com/v1/chat/completions` accepts none/low/medium/high/max and
    /// **rejects `minimal` with HTTP 400**.
    static let cloudDefault = LLMEffortVocabulary(
        supported: [.off, .low, .medium, .high, .max],
        overrides: [.xhigh: .max])

    /// GLM-5.2's knob is exactly its minimum thinking level and its top tier —
    /// it has no low or medium at all.
    static let glm52 = LLMEffortVocabulary(
        supported: [.high, .max],
        overrides: [.xhigh: .max])

    /// GLM-5.3 widens the same knob to a graded scale. Live-verified with
    /// monotonic reasoning-token scaling on the native endpoint.
    static let glm53 = LLMEffortVocabulary(
        supported: [.low, .medium, .high, .max],
        overrides: [.xhigh: .max])

    /// Kimi K3 has no `medium`, and `high` is both its positional middle and
    /// its server default — so `medium` belongs at `high`, not down at `low`.
    /// Ladder arithmetic alone would get this wrong.
    static let kimiK3 = LLMEffortVocabulary(
        supported: [.low, .high, .max],
        overrides: [.medium: .high, .xhigh: .max])

    /// Everything Kimi before K3.
    static let kimiK2 = LLMEffortVocabulary(supported: [.low, .medium, .high])

    /// DeepSeek V4's OpenAI-compatible knob.
    static let deepSeekV4 = LLMEffortVocabulary(
        supported: [.low, .medium, .high, .max],
        overrides: [.xhigh: .max])

    /// `k3` as a delimited token — `k3`, `k3-256k`, `kimi-k3-cot` — and never a
    /// K2-era name like `kimi-k2.6`.
    private static let kimiK3Pattern = try? NSRegularExpression(
        pattern: "(?:^|[^a-z0-9])k3(?:[^a-z0-9]|$)")

    /// The bare model id: no vendor prefix, no `:tag`, lowercased.
    static func slug(_ model: String) -> String {
        let bare = model.lowercased().split(separator: "/").last.map(String.init) ?? model.lowercased()
        return bare.split(separator: ":").first.map(String.init) ?? bare
    }

    private static func isKimiK3(_ slug: String) -> Bool {
        guard let pattern = kimiK3Pattern else { return false }
        return pattern.firstMatch(in: slug, range: NSRange(slug.startIndex..., in: slug)) != nil
    }

    /// The vocabulary for a model, or nil when we have no evidence it takes a
    /// level. `cloud` selects the OpenAI-compatible wire's defaults for a
    /// thinking model we recognise but have no model-specific set for.
    public static func vocabulary(for model: String, cloud: Bool) -> LLMEffortVocabulary? {
        let slug = slug(model)
        if slug.hasPrefix("glm-5.3") || slug.hasPrefix("glm-5-3") { return glm53 }
        if slug.hasPrefix("glm-5.2") || slug.hasPrefix("glm-5-2") { return glm52 }
        if slug.hasPrefix("deepseek-v4") { return deepSeekV4 }
        if slug.hasPrefix("kimi") || slug.hasPrefix("k3") {
            return isKimiK3(slug) ? kimiK3 : kimiK2
        }
        // A thinking family we know of but have no measured set for: the wire's
        // own documented vocabulary, which the gateway normalizes per model.
        if slug.hasPrefix("glm") || slug.hasPrefix("deepseek") {
            return cloud ? cloudDefault : localDefault
        }
        return nil
    }
}
