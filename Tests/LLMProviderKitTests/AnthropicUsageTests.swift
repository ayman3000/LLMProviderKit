import Testing
import Foundation
@testable import LLMProviderKit
@testable import LLMProviderKitAnthropic

/// Anthropic's `input_tokens` leaves out what was read from or written to the
/// prompt cache. On every other provider `promptTokens` is the whole prompt
/// and `cachedTokens` a part of it, so Anthropic reports the sum too —
/// otherwise cached ÷ prompt exceeds 100% and an app's context size (and its
/// compaction trigger) counts only the uncached part.
struct AnthropicUsageTests {
    let provider = AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: "k", model: "claude-sonnet-4-6"))
    let request = LLMRequest(model: "claude-sonnet-4-6", messages: [LLMMessage(role: .user, content: "hi")])

    @Test func aStreamedReplyReportsTheWholePrompt() throws {
        let line = #"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":12,"cache_read_input_tokens":900,"cache_creation_input_tokens":88,"output_tokens":5}}"#
        let chunks = try provider.parseStreamLine(line, request: request)
        guard case .finish(reason: _, usage: let usage)? = chunks.first else {
            Issue.record("no finish chunk"); return
        }
        #expect(usage?.promptTokens == 1_000)
        #expect(usage?.cachedTokens == 900)
        #expect(usage?.completionTokens == 5)
        #expect(usage?.totalTokens == 1_005)
    }

    @Test func aWholeReplyReportsTheWholePrompt() throws {
        let body = #"{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"cache_read_input_tokens":900,"cache_creation_input_tokens":88,"output_tokens":5}}"#
        let response = try provider.parseResponse(Data(body.utf8), request: request)
        #expect(response.usage?.promptTokens == 1_000)
        #expect(response.usage?.cachedTokens == 900)
    }

    @Test func withoutCacheFieldsThePromptIsTheInput() throws {
        let line = #"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":12,"output_tokens":5}}"#
        guard case .finish(reason: _, usage: let usage)? = try provider.parseStreamLine(line, request: request).first else {
            Issue.record("no finish chunk"); return
        }
        #expect(usage?.promptTokens == 12)
        #expect(usage?.cachedTokens == nil)
    }

    @Test func theSumIsNilOnlyWhenNothingWasReported() {
        #expect(AnthropicProvider.usage(input: nil, output: 3, cacheRead: nil, cacheWrite: nil).promptTokens == nil)
        #expect(AnthropicProvider.usage(input: nil, output: nil, cacheRead: 7, cacheWrite: nil).promptTokens == 7)
    }

    /// Cache writes cost more than plain input (1.25x for 5-minute writes), so
    /// a cost estimate needs them apart from the rest of the prompt.
    @Test func cacheWritesAreReportedApart() {
        let u = AnthropicProvider.usage(input: 50, output: 3, cacheRead: 900, cacheWrite: 40)
        #expect(u.promptTokens == 990)
        #expect(u.cachedTokens == 900)
        #expect(u.cacheWriteTokens == 40)
        #expect(LLMUsage(promptTokens: 1).cacheWriteTokens == nil)
    }
}
