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
        #expect(LLMReasoningEffort.ladder == [.off, .minimal, .low, .medium, .high, .xhigh, .max, .ultra])
        #expect(LLMReasoningEffort.ladder.map(\.rank) == Array(0..<8))
    }
}

import LLMProviderKitOllama

/// Ollama's declared families. These are the sets a wrong entry turns into a
/// 400, so each is pinned rather than trusted.
struct OllamaEffortTests {
    @Test func glmVersionsDifferAndBothAreDeclared() {
        // 5.2's knob starts at its minimum thinking level; 5.3 is graded.
        #expect(OllamaEffort.vocabulary(for: "glm-5.2:cloud", cloud: true)?.supported == [.high, .max])
        #expect(OllamaEffort.vocabulary(for: "glm-5.3", cloud: false)?.supported
                == [.low, .medium, .high, .max])
    }

    /// The level the app defaults to must land somewhere valid on the model
    /// actually in use — this is the case the old boolean gate would have sent
    /// straight through.
    @Test func mediumResolvesOnAModelWithoutIt() {
        let glm52 = try! #require(OllamaEffort.vocabulary(for: "glm-5.2:cloud", cloud: true))
        #expect(glm52.supported.contains(.medium) == false)
        #expect(glm52.clamp(.medium) == .high)   // the floor, not a refusal
    }

    @Test func kimiK3IsMatchedAsADelimitedToken() {
        for id in ["k3", "k3-256k", "kimi-k3-cot"] {
            #expect(OllamaEffort.vocabulary(for: id, cloud: true)?.supported == [.low, .high, .max],
                    "\(id) should be K3")
        }
        // K2-era names must not match the K3 token.
        #expect(OllamaEffort.vocabulary(for: "kimi-k2.6", cloud: true)?.supported
                == [.low, .medium, .high])
    }

    /// K3's `high` is its middle AND its server default, so ladder arithmetic
    /// (which would pick `low`) is wrong and the override is right.
    @Test func kimiK3MapsMediumUpwardByVendorRule() {
        let k3 = try! #require(OllamaEffort.vocabulary(for: "k3", cloud: true))
        #expect(k3.clamp(.medium) == .high)
    }

    @Test func minimalNeverReachesOllamaCloud() {
        // The cloud wire answers 400 for `minimal`; it must be clamped away.
        let cloud = try! #require(OllamaEffort.vocabulary(for: "glm-5.3", cloud: true))
        #expect(cloud.supported.contains(.minimal) == false)
        #expect(cloud.clamp(.minimal) == .low)
    }

    @Test func unknownModelsAreSentNoLevel() {
        for id in ["llama3.2", "mistral-small", "something-new"] {
            #expect(OllamaEffort.vocabulary(for: id, cloud: false) == nil, "\(id) must be undeclared")
        }
    }

    @Test func thinkIsATopLevelStringNotAnOption() throws {
        let provider = OllamaProvider(configuration: OllamaProvider.local(
            model: "glm-5.3", baseURL: URL(string: "http://localhost:11434")!))
        let request = LLMRequest(model: "glm-5.3",
                                 messages: [LLMMessage(role: .user, content: "hi")],
                                 reasoningEffort: .low)
        let data = try #require(provider.prepareRequest(request, stream: false).httpBody)
        let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(body["think"] as? String == "low")
        #expect((body["options"] as? [String: Any])?["think"] == nil)
    }

    @Test func noEffortSendsNoThink() throws {
        let provider = OllamaProvider(configuration: OllamaProvider.local(
            model: "glm-5.3", baseURL: URL(string: "http://localhost:11434")!))
        let request = LLMRequest(model: "glm-5.3",
                                 messages: [LLMMessage(role: .user, content: "hi")])
        let data = try #require(provider.prepareRequest(request, stream: false).httpBody)
        let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(body["think"] == nil)
    }
}
