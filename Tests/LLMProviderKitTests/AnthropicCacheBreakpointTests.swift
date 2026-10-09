import Testing
import Foundation
@testable import LLMProviderKit
@testable import LLMProviderKitAnthropic

/// Anthropic looks a cached prefix up from each breakpoint, walking back
/// about 20 blocks. One step with many parallel tool calls adds more than
/// that, so the single moving breakpoint on the last message could miss the
/// previous request's entry. A fourth breakpoint sits exactly where the
/// previous request ended: the last block of the message right before the
/// newest assistant turn.
struct AnthropicCacheBreakpointTests {
    static let tool = LLMToolDefinition(name: "read_file", description: "Read.",
                                        parameters: ["type": "object", "properties": [:] as [String: Any]])

    private func body(_ messages: [LLMMessage], tools: [LLMToolDefinition] = []) throws -> (json: [String: Any], text: String) {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "k", model: "claude-sonnet-4-6"))
        let request = LLMRequest(model: "claude-sonnet-4-6", messages: messages, tools: tools)
        let data = try #require(try provider.prepareRequest(request, stream: false).httpBody)
        let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        return (json, String(decoding: data, as: UTF8.self))
    }

    private func marks(_ text: String) -> Int { text.components(separatedBy: #""cache_control""#).count - 1 }

    private func isMarked(_ message: [String: Any]) -> Bool {
        (message["content"] as? [[String: Any]])?.last?["cache_control"] != nil
    }

    private func step(_ id: String) -> [LLMMessage] {
        var call = LLMMessage(role: .assistant, content: "")
        call.toolCalls = [LLMToolCall(id: id, name: "read_file", arguments: "{}")]
        var result = LLMMessage(role: .tool, content: "text of \(id)")
        result.toolCallId = id
        return [call, result]
    }

    /// Mid-run: the previous request ended at the tool result before the
    /// newest step.
    @Test func thePreviousRequestsLastBlockIsMarked() throws {
        let (json, text) = try body([.system("sys"), .user("go")] + step("a") + step("b"), tools: [Self.tool])
        let messages = try #require(json["messages"] as? [[String: Any]])
        // user, assistant(a), user(result a), assistant(b), user(result b)
        #expect(messages.count == 5)
        #expect(isMarked(messages[2]))     // where the previous request ended
        #expect(isMarked(messages[4]))     // the newest message
        #expect(!isMarked(messages[0]) && !isMarked(messages[1]) && !isMarked(messages[3]))
        #expect(marks(text) == 4)          // tools, system, previous turn, last message
    }

    /// A new user turn: the previous request ended at the last tool result
    /// before the final answer.
    @Test func aNewTurnMarksWhereThePreviousRunEnded() throws {
        let (json, _) = try body([.system("sys"), .user("go")] + step("a") + [.assistant("done"), .user("again")],
                                 tools: [Self.tool])
        let messages = try #require(json["messages"] as? [[String: Any]])
        // user, assistant(a), user(result a), assistant(done), user(again)
        #expect(isMarked(messages[2]))
        #expect(isMarked(messages[4]))
        #expect(!isMarked(messages[3]))
    }

    @Test func aFirstMessageHasNoPreviousTurn() throws {
        let (_, text) = try body([.system("sys"), .user("hi")])
        #expect(marks(text) == 2)          // system, last message
    }

    @Test func neverMoreThanFour() throws {
        var messages: [LLMMessage] = [.system("sys"), .user("go")]
        for i in 0..<12 { messages += step("s\(i)") }
        let (_, text) = try body(messages, tools: [Self.tool])
        #expect(marks(text) == 4)
    }

    /// Parallel tool results merge into one user entry first; the previous
    /// turn's breakpoint lands on that entry's last block only.
    @Test func parallelResultsAreMarkedOnTheMergedEntry() throws {
        var call = LLMMessage(role: .assistant, content: "")
        call.toolCalls = (0..<3).map { LLMToolCall(id: "p\($0)", name: "read_file", arguments: "{}") }
        let results: [LLMMessage] = (0..<3).map {
            var r = LLMMessage(role: .tool, content: "text of p\($0)"); r.toolCallId = "p\($0)"; return r
        }
        let (json, text) = try body([.system("sys"), .user("go"), call] + results + step("b"), tools: [Self.tool])
        let messages = try #require(json["messages"] as? [[String: Any]])
        // user, assistant(p0..p2), user(results p0..p2), assistant(b), user(result b)
        #expect(messages.count == 5)
        let merged = try #require(messages[2]["content"] as? [[String: Any]])
        #expect(merged.count == 3)
        #expect(merged[2]["cache_control"] != nil)
        #expect(merged[0]["cache_control"] == nil && merged[1]["cache_control"] == nil)
        #expect(marks(text) == 4)
    }

    /// Same conversation, same body, call after call.
    @Test func placementIsDeterministic() throws {
        let messages: [LLMMessage] = [.system("sys"), .user("go")] + step("a") + step("b")
        #expect(try body(messages, tools: [Self.tool]).text == body(messages, tools: [Self.tool]).text)
    }
}
