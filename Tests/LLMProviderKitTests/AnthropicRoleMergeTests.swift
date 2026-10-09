import Foundation
import Testing
@testable import LLMProviderKit
@testable import LLMProviderKitAnthropic

/// The Messages API wants user and assistant turns to alternate. An agent's
/// history does not always: a progress note can follow a tool-result turn, a
/// stored nudge can precede that note, and receipts kept in their step's place
/// can put two assistant turns side by side. Runs of one role are merged
/// client-side into ONE entry, content blocks in order, so the body is valid
/// and byte-identical for identical input without relying on the server.
///
/// Shape: two plain texts become two text blocks (never one joined string), so
/// no separator is invented and each message's bytes survive unchanged.
struct AnthropicRoleMergeTests {

    private func bodies(_ messages: [LLMMessage]) throws -> [(stream: Bool, data: Data, json: [String: Any])] {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "k", model: "claude-sonnet-4-6"))
        let request = LLMRequest(model: "claude-sonnet-4-6", messages: messages)
        return try [false, true].map { stream in
            let data = try #require(provider.prepareRequest(request, stream: stream).httpBody)
            let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            return (stream, data, json)
        }
    }

    private func toolTurn() -> [LLMMessage] {
        var assistant = LLMMessage(role: .assistant, content: "")
        assistant.toolCalls = [LLMToolCall(id: "a1", name: "read_file", arguments: "{}"),
                               LLMToolCall(id: "b2", name: "list_dir", arguments: "{}")]
        var r1 = LLMMessage(role: .tool, content: "file text"); r1.toolCallId = "a1"
        var r2 = LLMMessage(role: .tool, content: "dir listing"); r2.toolCallId = "b2"
        return [LLMMessage(role: .user, content: "go"), assistant, r1, r2]
    }

    private func roles(_ json: [String: Any]) -> [String] {
        ((json["messages"] as? [[String: Any]]) ?? []).compactMap { $0["role"] as? String }
    }

    // (a)
    @Test func aNoteAfterToolResultsJoinsTheirUserEntry() throws {
        for b in try bodies(toolTurn() + [LLMMessage(role: .user, content: "Progress: 2 of 5")]) {
            let messages = try #require(b.json["messages"] as? [[String: Any]])
            #expect(roles(b.json) == ["user", "assistant", "user"], "stream=\(b.stream)")
            let blocks = try #require(messages[2]["content"] as? [[String: Any]])
            #expect(blocks.map { $0["type"] as? String } == ["tool_result", "tool_result", "text"])
            #expect(blocks.map { $0["tool_use_id"] as? String } == ["a1", "b2", nil])
            #expect(blocks[2]["text"] as? String == "Progress: 2 of 5")
            // The breakpoint lands on the merged entry's last block only.
            #expect((blocks[2]["cache_control"] as? [String: Any])?["type"] as? String == "ephemeral")
            #expect(blocks[0]["cache_control"] == nil && blocks[1]["cache_control"] == nil)
        }
    }

    // (b)
    @Test func twoPlainUserMessagesBecomeOneEntryWithTwoTextBlocks() throws {
        let msgs = toolTurn() + [LLMMessage(role: .assistant, content: "thinking out loud"),
                                 LLMMessage(role: .user, content: "Stop reasoning and act."),
                                 LLMMessage(role: .user, content: "Progress: 3 of 5")]
        for b in try bodies(msgs) {
            let messages = try #require(b.json["messages"] as? [[String: Any]])
            #expect(roles(b.json) == ["user", "assistant", "user", "assistant", "user"], "stream=\(b.stream)")
            let blocks = try #require(messages[4]["content"] as? [[String: Any]])
            #expect(blocks.map { $0["type"] as? String } == ["text", "text"])
            #expect(blocks.map { $0["text"] as? String } == ["Stop reasoning and act.", "Progress: 3 of 5"])
            #expect(blocks[0]["cache_control"] == nil)
            #expect((blocks[1]["cache_control"] as? [String: Any])?["type"] as? String == "ephemeral")
        }
    }

    // (c)
    @Test func consecutiveAssistantMessagesMergeInOrder() throws {
        var call = LLMMessage(role: .assistant, content: "next step")
        call.toolCalls = [LLMToolCall(id: "c3", name: "write_file", arguments: "{}")]
        var r3 = LLMMessage(role: .tool, content: "written"); r3.toolCallId = "c3"
        let msgs = [LLMMessage(role: .user, content: "go"),
                    LLMMessage(role: .assistant, content: "Receipt: read a.swift"),
                    call, r3]
        for b in try bodies(msgs) {
            let messages = try #require(b.json["messages"] as? [[String: Any]])
            #expect(roles(b.json) == ["user", "assistant", "user"], "stream=\(b.stream)")
            let blocks = try #require(messages[1]["content"] as? [[String: Any]])
            #expect(blocks.map { $0["type"] as? String } == ["text", "text", "tool_use"])
            #expect(blocks.map { $0["text"] as? String } == ["Receipt: read a.swift", "next step", nil])
            #expect(blocks[2]["id"] as? String == "c3")
        }
    }

    @Test func emptyTextIsDroppedWhenMerging() throws {
        // An empty text block is a 400; merging must not create one.
        let msgs = [LLMMessage(role: .user, content: "a"), LLMMessage(role: .user, content: ""),
                    LLMMessage(role: .user, content: "b")]
        for b in try bodies(msgs) {
            let messages = try #require(b.json["messages"] as? [[String: Any]])
            #expect(messages.count == 1)
            let blocks = try #require(messages[0]["content"] as? [[String: Any]])
            #expect(blocks.map { $0["text"] as? String } == ["a", "b"])
        }
    }

    // (d)
    @Test func alternatingConversationsAreUnchanged() throws {
        let msgs = toolTurn() + [LLMMessage(role: .assistant, content: "done"),
                                 LLMMessage(role: .user, content: "thanks")]
        for b in try bodies(msgs) {
            let messages = try #require(b.json["messages"] as? [[String: Any]])
            #expect(roles(b.json) == ["user", "assistant", "user", "assistant", "user"])
            // Lone plain messages keep their string form (the last one becomes a
            // single marked text block for the cache breakpoint, as before).
            #expect(messages[0]["content"] as? String == "go")
            #expect(messages[3]["content"] as? String == "done")
            let last = try #require(messages[4]["content"] as? [[String: Any]])
            #expect(last.count == 1 && last[0]["text"] as? String == "thanks")
            #expect((messages[2]["content"] as? [[String: Any]])?.count == 2)
        }
    }

    // (e)
    @Test func mergedBodiesKeepSortedKeysAndAreByteStable() throws {
        let msgs = toolTurn() + [LLMMessage(role: .user, content: "Stop reasoning and act."),
                                 LLMMessage(role: .user, content: "Progress: 2 of 5")]
        let first = try bodies(msgs)
        let second = try bodies(msgs)
        for (a, b) in zip(first, second) {
            #expect(a.data == b.data, "stream=\(a.stream)")
            let text = try #require(String(data: a.data, encoding: .utf8))
            // Sorted keys: inside every block "content"/"text" precede "type",
            // and at top level "max_tokens" < "messages" < "model" < "stream".
            #expect(text.contains(#"{"cache_control":{"type":"ephemeral"},"text":"Progress: 2 of 5","type":"text"}"#))
            #expect(text.contains(#"{"text":"Stop reasoning and act.","type":"text"}"#))
            let keys = ["\"max_tokens\"", "\"messages\"", "\"model\"", "\"stream\""]
                .compactMap { text.range(of: $0)?.lowerBound }
            #expect(keys.count == 4 && keys == keys.sorted())
        }
    }
}
