import Foundation
import Testing
@testable import LLMProviderKit
@testable import LLMProviderKitAnthropic

/// Claude streams a tool call's arguments as `input_json_delta` pieces after a
/// `content_block_start` that carries an empty `input`. The parser used to emit
/// the call with `{}` and drop the pieces; once the agent stopped re-asking the
/// turn without streaming, every Claude tool call reached its tool with no
/// arguments ("run_shell requires a non-empty command", 2026-10-02). With
/// adaptive thinking a thinking block comes first, so the call is not block 0.
struct AnthropicStreamToolCallTests {

    /// Runs lines through the parser and the assembler, the way `stream()` does.
    private func streamed(_ lines: [String]) throws -> [LLMStreamChunk] {
        let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "k", model: "claude-sonnet-4-6"))
        let request = LLMRequest(model: "claude-sonnet-4-6", messages: [])
        var assembler = StreamToolCallAssembler()
        var out: [LLMStreamChunk] = []
        for line in lines {
            for chunk in try provider.parseStreamLine(line, request: request) {
                out += assembler.consume(chunk)
            }
        }
        return out + assembler.flush()
    }

    private func calls(_ chunks: [LLMStreamChunk]) -> [LLMToolCall] {
        chunks.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }
    }

    private func args(_ call: LLMToolCall) -> [String: String] {
        (try? JSONSerialization.jsonObject(with: Data(call.arguments.utf8))) as? [String: String] ?? [:]
    }

    @Test func argumentsStreamedInPiecesAfterAThinkingBlockArriveWhole() throws {
        let chunks = try streamed([
            #"data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[]}}"#,
            #"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            #"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"List it."}}"#,
            #"data: {"type":"content_block_stop","index":0}"#,
            #"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"run_shell","input":{}}}"#,
            #"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}"#,
            #"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"command\": \"ls ~/Nas"}}"#,
            #"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"eem/Demos\"}"}}"#,
            #"data: {"type":"content_block_stop","index":1}"#,
            #"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":40}}"#,
        ])
        let found = calls(chunks)
        #expect(found.count == 1)
        #expect(found.first?.id == "toolu_1")
        #expect(found.first?.name == "run_shell")
        #expect(found.first.map(args) == ["command": "ls ~/Naseem/Demos"])
    }

    @Test func parallelCallsKeepTheirOwnArguments() throws {
        let chunks = try streamed([
            #"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            #"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Checking both."}}"#,
            #"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"a","name":"read_file","input":{}}}"#,
            #"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"a.txt\"}"}}"#,
            #"data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"b","name":"read_file","input":{}}}"#,
            #"data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#,
            #"data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"b.txt\"}"}}"#,
            #"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        ])
        let found = calls(chunks)
        #expect(found.map(\.id) == ["a", "b"])
        #expect(found.map(args) == [["path": "a.txt"], ["path": "b.txt"]])
        #expect(found.allSatisfy { $0.providerMetadata[StreamToolCallAssembler.indexKey] == nil })
    }

    /// A tool with no parameters streams no pieces at all; it still gets `{}`.
    @Test func aCallWithNoArgumentsIsAnEmptyObject() throws {
        let found = calls(try streamed([
            #"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t","name":"sim_list","input":{}}}"#,
            #"data: {"type":"content_block_stop","index":0}"#,
            #"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
        ]))
        #expect(found.count == 1)
        #expect(found.first?.arguments == "{}")
    }
}
