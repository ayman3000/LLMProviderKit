import Testing
import Foundation
import LLMProviderKit
@testable import LLMProviderKitGemini

/// Gemini wants a function-call turn right after a user turn or a
/// function-response turn. An agent's sifted history (receipts in place)
/// sends model text → model text → model function call, so consecutive
/// messages of one role go out as ONE content, parts in order.
struct GeminiContentsTests {
    static func contents(_ messages: [LLMMessage]) throws -> [[String: Any]] {
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: "k", model: GeminiModel.flashLite31))
        let request = LLMRequest(model: GeminiModel.flashLite31, messages: messages)
        let body = try #require(try provider.prepareRequest(request, stream: false).httpBody)
        let json = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        return try #require(json["contents"] as? [[String: Any]])
    }

    /// Two receipts, a step with two parallel calls, their two results, and
    /// the progress note after them (a trailing user message).
    static func siftedShape() -> [LLMMessage] {
        var call = LLMMessage(role: .assistant, content: "")
        call.toolCalls = [LLMToolCall(id: "get_time", name: "get_time", arguments: "{}"),
                          LLMToolCall(id: "get_date", name: "get_date", arguments: "{}")]
        var time = LLMMessage(role: .tool, content: "12:00")
        time.toolCallId = "get_time"
        var date = LLMMessage(role: .tool, content: "2026-10-09")
        date.toolCallId = "get_date"
        return [
            LLMMessage(role: .system, content: "sys"),
            LLMMessage(role: .user, content: "do three things"),
            LLMMessage(role: .assistant, content: "[receipt 1]"),
            LLMMessage(role: .assistant, content: "[receipt 2]"),
            call, time, date,
            LLMMessage(role: .user, content: "[Progress check] note"),
        ]
    }

    @Test func noTwoContentsInARowShareARole() throws {
        let roles = try Self.contents(Self.siftedShape()).compactMap { $0["role"] as? String }
        #expect(roles == ["user", "model", "user"])
        for (earlier, later) in zip(roles, roles.dropFirst()) { #expect(earlier != later) }
    }

    @Test func mergedPartsKeepTheirOrder() throws {
        let contents = try Self.contents(Self.siftedShape())
        let model = try #require(contents[1]["parts"] as? [[String: Any]])
        #expect(model.count == 4)
        #expect(model[0]["text"] as? String == "[receipt 1]")
        #expect(model[1]["text"] as? String == "[receipt 2]")
        #expect(model[2]["functionCall"] != nil && model[3]["functionCall"] != nil)
        let user = try #require(contents[2]["parts"] as? [[String: Any]])
        // A tool message goes out as its text part, then its functionResponse
        // part (the adapter's existing tool-message shape), so two results
        // and the note make five parts.
        #expect(user.count == 5)
        #expect(user[0]["text"] as? String == "12:00" && user[1]["functionResponse"] != nil)
        #expect(user[2]["text"] as? String == "2026-10-09" && user[3]["functionResponse"] != nil)
        #expect(user[4]["text"] as? String == "[Progress check] note")
    }

    @Test func alternatingMessagesAreUnchanged() throws {
        let contents = try Self.contents([LLMMessage(role: .user, content: "a"),
                                          LLMMessage(role: .assistant, content: "b"),
                                          LLMMessage(role: .user, content: "c")])
        #expect(contents.count == 3)
    }
}

/// Opt-in, owner-approved, ONE call to the cheapest Gemini model: the merged
/// shape (receipts and a function call in one model turn; function responses
/// and a note in one user turn) is accepted by the live API.
/// GEMINI_LIVE=1 GEMINI_API_KEY=… [GEMINI_LIVE_MODEL=…] swift test --filter GeminiLiveTurnOrderTests
struct GeminiLiveTurnOrderTests {
    static let env = ProcessInfo.processInfo.environment

    @Test(.enabled(if: GeminiLiveTurnOrderTests.env["GEMINI_LIVE"] == "1"))
    func theMergedShapeIsAccepted() async throws {
        let key = try #require(Self.env["GEMINI_API_KEY"])
        let model = Self.env["GEMINI_LIVE_MODEL"] ?? GeminiModel.flashLite31
        let provider = GeminiProvider(configuration: GeminiProvider.gemini(apiKey: key, model: model))
        var messages = GeminiContentsTests.siftedShape()
        // A replayed call that Gemini did not make itself has no thought
        // signature, and Gemini 3 validates them. Google documents this
        // placeholder for such injected calls; confirm it in the current
        // Gemini docs before the run. ("gemini.thoughtSignature" is the
        // provider's private metadata key, GeminiProvider.swift:14.)
        for index in messages.indices where messages[index].toolCalls != nil {
            messages[index].toolCalls = messages[index].toolCalls?.map { call in
                var call = call
                call.providerMetadata["gemini.thoughtSignature"] = "context_engineering_is_the_way_to_go"
                return call
            }
        }
        let tools = ["get_time", "get_date"].map {
            LLMToolDefinition(name: $0, description: "Returns the current value.",
                              parameters: ["type": "object", "properties": [String: Any]()])
        }
        let request = LLMRequest(model: model, messages: messages, maxTokens: 64, tools: tools)
        let response = try await provider.complete(request)
        print("LIVE gemini turn order (\(model)): \(response.text.prefix(80))")
    }
}
