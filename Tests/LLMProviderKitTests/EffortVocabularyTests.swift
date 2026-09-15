import Testing
import Foundation
import LLMProviderKit

/// The clamp rules, one test each. Every one of these is a bug somebody else
/// already shipped: levels leaking to wires that answer 400, an unknown level
/// resolving to a weak default, a clamp quietly switching thinking off.
struct EffortVocabularyTests {
    /// Anthropic's set, and the shape most wires have.
    private static let fiveLevel = LLMEffortVocabulary(
        supported: [.low, .medium, .high, .xhigh, .max])
    /// GLM-5.3 natively: no xhigh, with the vendor's own mapping for it.
    private static let glm53 = LLMEffortVocabulary(
        supported: [.low, .medium, .high, .max], overrides: [.xhigh: .max])
    /// Kimi K3: no medium, and `high` is its middle AND its server default, so
    /// ladder arithmetic would wrongly send `medium` down to `low`.
    private static let kimiK3 = LLMEffortVocabulary(
        supported: [.low, .high, .max], overrides: [.medium: .high, .xhigh: .max])

    @Test func unsetStaysUnset() {
        #expect(Self.fiveLevel.clamp(nil) == nil)
        #expect(Self.kimiK3.clamp(nil) == nil)
    }

    @Test func supportedPassesThroughVerbatim() {
        for level in Self.fiveLevel.supported {
            #expect(Self.fiveLevel.clamp(level) == level)
        }
    }

    @Test func anOverrideBeatsTheLadder() {
        // Down to .low would be the arithmetic answer; the vendor says .high.
        #expect(Self.kimiK3.clamp(.medium) == .high)
        #expect(Self.glm53.clamp(.xhigh) == .max)
    }

    @Test func unsupportedFallsToTheNearestWeaker() {
        // No xhigh, no override → high, not max.
        let noOverride = LLMEffortVocabulary(supported: [.low, .medium, .high, .max])
        #expect(noOverride.clamp(.xhigh) == .high)
        #expect(Self.kimiK3.clamp(.minimal) == .low)
    }

    /// The floor is the closest honest match when a wire cannot go lower — and
    /// still cheaper than omitting the field and inheriting a default of high.
    @Test func nothingWeakerFallsToTheFloor() {
        #expect(Self.fiveLevel.clamp(.off) == .low)
        #expect(Self.fiveLevel.clamp(.minimal) == .low)
    }

    /// Clamping onto `none` would switch reasoning off for someone who asked
    /// for a little of it.
    @Test func neverClampsOntoNone() {
        let withNone = LLMEffortVocabulary(supported: [.off, .high, .max])
        #expect(withNone.clamp(.minimal) == .high)
        #expect(withNone.clamp(.low) == .high)
        // Explicitly asking for it still works — it is a target, not a fallback.
        #expect(withNone.clamp(.off) == .off)
    }

    /// A stronger ask must never resolve weaker than a weaker ask would.
    @Test func resolutionIsMonotonic() {
        for vocabulary in [Self.fiveLevel, Self.glm53, Self.kimiK3] {
            let resolved = LLMReasoningEffort.ladder.map { vocabulary.clamp($0)?.rank ?? -1 }
            #expect(resolved == resolved.sorted(), "not monotonic for \(vocabulary.supported)")
        }
    }

    /// A wire nobody has declared must not have its caller's level altered —
    /// and a provider that returns no vocabulary sends nothing at all.
    @Test func anEmptyVocabularyChangesNothing() {
        let unknown = LLMEffortVocabulary(supported: [])
        #expect(unknown.isEmpty)
        #expect(unknown.clamp(.xhigh) == .xhigh)
    }

    /// The ladder's order is load-bearing: clamping is rank arithmetic.
    @Test func theLadderRunsWeakestToStrongest() {
        #expect(LLMReasoningEffort.ladder == [.off, .minimal, .low, .medium, .high, .xhigh, .max])
        #expect(LLMReasoningEffort.ladder.map(\.rank) == Array(0..<7))
    }
}
