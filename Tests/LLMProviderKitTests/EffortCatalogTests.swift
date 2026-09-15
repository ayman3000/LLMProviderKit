import Testing
import Foundation
import LLMProviderKit

/// The catalog is the one place these sets live, and the override file is the
/// reason a wrong row no longer needs a release. Both halves are pinned here:
/// that the declared rows are right, and that a bad file cannot break the app.
/// Serialized: EffortCatalog.shared is a singleton and these cases mutate its
/// override layer. Run in parallel they clobber each other — which they did,
/// intermittently, until adding unrelated tests changed the timing enough to
/// make it show.
@Suite(.serialized)
struct EffortCatalogTests {
    private func catalog() -> EffortCatalog {
        let c = EffortCatalog.shared
        c._resetOverrides()
        return c
    }

    @Test func providerAndModelTogetherDecide() {
        let c = catalog()
        // The same vendor, two surfaces, two answers — the case a per-model or
        // per-provider table could not express.
        // The endpoint refuses `ultra` — observed live, quoted in the rows.
        #expect(c.vocabulary(provider: "chatgptCodex", model: "gpt-5.6-sol")?
                .supported.contains(.ultra) == false)
        #expect(c.vocabulary(provider: "chatgptCodex", model: "gpt-5.6-sol")?.supported
                == [.off, .minimal, .low, .medium, .high, .xhigh, .max])
        // A model the catalog has never heard of gets nothing at all.
        #expect(c.vocabulary(provider: "chatgptCodex", model: "gpt-4o") == nil)
        #expect(c.vocabulary(provider: "gemini", model: "gemini-3.6-flash") == nil)
    }

    /// OpenRouter ids are `vendor/model` and the vendor is the signal; Ollama
    /// ids arrive tagged. Both spellings have to resolve.
    @Test func theFullIdAndTheSlugBothResolve() {
        let c = catalog()
        #expect(c.vocabulary(provider: "openrouter", model: "anthropic/claude-sonnet-4") != nil)
        #expect(c.vocabulary(provider: "ollamaCloud", model: "glm-5.2:cloud")?.supported
                == [.high, .max])
    }

    /// A specific row must be able to carve an exception out of a general one.
    @Test func theLongestPrefixWins() {
        let c = catalog()
        #expect(c.vocabulary(provider: "ollama", model: "glm-5.2")?.supported == [.high, .max])
        #expect(c.vocabulary(provider: "ollama", model: "glm-5.3")?.supported
                == [.low, .medium, .high, .max])
        // …and anything else GLM falls back to the wire's own set.
        #expect(c.vocabulary(provider: "ollama", model: "glm-4")?.supported
                == [.low, .medium, .high, .max])
    }

    @Test func anOverrideWinsOverTheBuiltInRow() throws {
        let c = catalog()
        #expect(c.vocabulary(provider: "ollama", model: "glm-5.2")?.supported == [.high, .max])
        c.applyOverrides([
            EffortCatalog.Key(provider: "ollama", pattern: "glm-5.2*"):
                EffortCatalog.Rule(supported: [.low, .high], overrides: [.medium: .low])
        ])
        let overridden = try #require(c.vocabulary(provider: "ollama", model: "glm-5.2:cloud"))
        #expect(overridden.supported == [.low, .high])
        #expect(overridden.clamp(.medium) == .low)
        c._resetOverrides()
        #expect(c.vocabulary(provider: "ollama", model: "glm-5.2")?.supported == [.high, .max])
    }

    @Test func aFileOfOverridesLoads() throws {
        let c = catalog()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("effort-\(UUID().uuidString).json")
        try #"""
        {"ollama": {"glm-5.9*": {"supported": ["low","max"], "overrides": {"medium": "low"}}}}
        """#.write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url); c._resetOverrides() }

        #expect(try c.loadOverrides(from: url) == 1)
        let rule = try #require(c.vocabulary(provider: "ollama", model: "glm-5.9"))
        #expect(rule.supported == [.low, .max])
        #expect(rule.clamp(.medium) == .low)
    }

    /// A typo in the file must cost the override, never the app: it throws, and
    /// the built-in rows are still there afterwards.
    @Test func aBadFileChangesNothing() throws {
        let c = catalog()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("effort-bad-\(UUID().uuidString).json")
        try "{ not json at all".write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(throws: (any Error).self) { try c.loadOverrides(from: url) }
        #expect(c.vocabulary(provider: "ollama", model: "glm-5.2")?.supported == [.high, .max])
    }

    /// An unknown level name in a file must not take the whole file down with
    /// it — but it must not be invented into the ladder either.
    @Test func anUnknownLevelInAFileIsRejected() throws {
        let c = catalog()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("effort-unknown-\(UUID().uuidString).json")
        try #"{"ollama": {"glm-9*": {"supported": ["low","hyper"], "overrides": {}}}}"#
            .write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url); c._resetOverrides() }

        #expect(throws: (any Error).self) { try c.loadOverrides(from: url) }
        #expect(c.vocabulary(provider: "ollama", model: "glm-5.2")?.supported == [.high, .max])
    }
}
