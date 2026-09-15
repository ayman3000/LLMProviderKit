import Foundation

/// How much work a model should spend on a response.
///
/// Effort is a **behavioural signal, not a token cap**: it shapes every output
/// token — the answer, the tool calls, and the thinking when a model thinks —
/// so it applies whether or not a model reasons. Lower effort tends to produce
/// fewer and terser tool calls, which is what makes it a cost lever for an
/// agent rather than only a reasoning dial.
///
/// The cases are a **ladder**, ordered weakest to strongest. The ladder is
/// deliberately wider than any single wire accepts: its job is to give
/// *ordering*, which is what `LLMEffortVocabulary` needs in order to clamp a
/// level a wire cannot express onto one it can.
public enum LLMReasoningEffort: String, Sendable, CaseIterable, Codable, Identifiable {
    /// Reasoning off. Some wires publish this as a level; most express it by
    /// omitting the field. Never a clamp target — see `LLMEffortVocabulary`.
    ///
    /// Named `off` rather than `none` on purpose: `clamp(.none)` in optional
    /// position resolves to `Optional.none` — nil — so a case called `none`
    /// would silently mean "no level" at every call site. The wire spelling
    /// stays `none`.
    case off = "none"
    /// Barely any. Rejected outright by several wires.
    case minimal
    /// Most efficient. Real token savings for some capability; suits short,
    /// scoped tasks and sub-agents.
    case low
    /// Balanced. The usual step down from the default when cost matters.
    case medium
    /// High capability. Matches what most providers do when nothing is sent.
    case high
    /// Extended capability for long-horizon agentic and coding work.
    case xhigh
    /// Maximum capability, no constraint on spending.
    case max

    public var id: String { rawValue }

    /// Weakest to strongest. `allCases` already follows declaration order; this
    /// name says the order is load-bearing rather than incidental.
    public static var ladder: [LLMReasoningEffort] { allCases }

    /// Position on the ladder. The whole point of the enum being ordered.
    public var rank: Int { Self.ladder.firstIndex(of: self) ?? 0 }

    /// Shown in pickers. Apps with a string catalog should localize this
    /// themselves — the kit ships no translations.
    public var displayName: String {
        switch self {
        case .off:     "Off"
        case .minimal: "Minimal"
        case .low:     "Low"
        case .medium:  "Medium"
        case .high:    "High"
        case .xhigh:   "Extra high"
        case .max:     "Max"
        }
    }
}

/// The effort levels one wire accepts, and how to reach the ones it does not.
///
/// A vocabulary belongs to a model **served by an endpoint**, not to a model:
/// GLM-5.3 publishes `low/high/max` on ollama.com and accepts
/// `low/medium/high/max` natively on z.ai. Both are true at once, so anything
/// keyed on a model id alone is wrong.
///
/// When a wire rejects a level, the fix is to correct its declared `supported`
/// set — never to add a condition to the dispatch path. That discipline is why
/// this type holds data and one function and nothing else.
public struct LLMEffortVocabulary: Sendable, Equatable {
    /// Levels this wire accepts verbatim.
    public let supported: [LLMReasoningEffort]

    /// Vendor semantics the ladder's arithmetic gets wrong, consulted before
    /// any clamping. Kimi K3 is the canonical example: `high` is both its
    /// positional middle and its server default, so `medium` belongs at `high`
    /// rather than down at `low`.
    public let overrides: [LLMReasoningEffort: LLMReasoningEffort]

    public init(supported: [LLMReasoningEffort],
                overrides: [LLMReasoningEffort: LLMReasoningEffort] = [:]) {
        self.supported = supported
        self.overrides = overrides
    }

    /// Whether this wire takes a level at all.
    public var isEmpty: Bool { supported.isEmpty }

    /// Resolve a requested level onto something this wire accepts.
    ///
    /// 1. Nothing requested → nothing sent. **Never invent an effort.**
    /// 2. An override names a supported level → use it.
    /// 3. Already supported → verbatim.
    /// 4. Otherwise → the nearest **weaker** supported level.
    /// 5. Nothing weaker exists → the weakest supported level.
    ///
    /// Three properties this guarantees, each of which cost somebody a bug:
    ///
    /// - **A clamp never escalates above the request** while anything weaker
    ///   exists. The one exception is rule 5: when a wire's floor is stronger
    ///   than the ask — "off" on a wire that cannot stop reasoning — the floor
    ///   is the closest honest match, and it is still cheaper than omitting the
    ///   field and inheriting a provider default of `high`.
    /// - **`off` is never a clamp target.** Landing there would switch
    ///   thinking off when the caller asked for a little, not for none.
    /// - **Monotonic.** A stronger request never resolves weaker than a weaker
    ///   request would.
    public func clamp(_ requested: LLMReasoningEffort?) -> LLMReasoningEffort? {
        guard let requested, !supported.isEmpty else { return requested }
        if supported.contains(requested) { return requested }
        if let mapped = overrides[requested], supported.contains(mapped) { return mapped }
        // `off` means "do not reason" — a destination only an explicit request
        // may reach, never a degradation.
        let candidates = supported.filter { $0 != .off }
        guard !candidates.isEmpty else { return nil }
        let weaker = candidates.filter { $0.rank < requested.rank }
        return weaker.max(by: { $0.rank < $1.rank })
            ?? candidates.min(by: { $0.rank < $1.rank })
    }
}
