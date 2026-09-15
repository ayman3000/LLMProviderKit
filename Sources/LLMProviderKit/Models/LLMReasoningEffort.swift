import Foundation

/// How much work a model should spend on a response.
///
/// Effort is a **behavioural signal, not a token cap**: it shapes every output
/// token — the answer, the tool calls, and the thinking when a model thinks —
/// so it applies whether or not a model reasons. Lower effort tends to produce
/// fewer and terser tool calls, which is what makes it a cost lever for an
/// agent rather than only a reasoning dial.
///
/// Providers spell this differently and support different subsets, so a level
/// is a *request*, not a guarantee. `nil` on `LLMRequest.reasoningEffort` means
/// "say nothing", which is every provider's existing behaviour.
///
/// - Important: sending a level to a model that does not accept one is an HTTP
///   400 on at least one provider. Gate the control on
///   `LLMModelCapability.reasoningEffort` rather than sending it blind.
public enum LLMReasoningEffort: String, Sendable, CaseIterable, Codable, Identifiable {
    /// Most efficient. Real token savings for some capability; suits short,
    /// scoped tasks and sub-agents.
    case low
    /// Balanced. The usual step down from the default when cost matters.
    case medium
    /// High capability. Matches what providers do when nothing is sent.
    case high
    /// Extended capability for long-horizon agentic and coding work. Not every
    /// model that accepts `max` accepts this one.
    case xhigh
    /// Maximum capability, no constraint on spending. Reserve it for work that
    /// justifies the bill.
    case max

    public var id: String { rawValue }

    /// Shown in pickers.
    public var displayName: String {
        switch self {
        case .low:    "Low"
        case .medium: "Medium"
        case .high:   "High"
        case .xhigh:  "Extra high"
        case .max:    "Max"
        }
    }

    /// Ordered cheapest-first, which is the order a picker should offer.
    public static var ordered: [LLMReasoningEffort] { [.low, .medium, .high, .xhigh, .max] }
}
