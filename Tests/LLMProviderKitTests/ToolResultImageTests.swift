import Foundation
@testable import LLMProviderKit
@testable import LLMProviderKitAnthropic
import Testing

/// Proves the tricky Anthropic path for multimodal tool results: an image a tool
/// returned rides inside the `tool_result.content` array in the SAME user turn,
/// so it reaches a vision model without breaking Anthropic's user/assistant
/// alternation. This is a deterministic check on the exact request we'd send.
struct ToolResultImageTests {

    private func decodedBody(_ messages: [LLMMessage]) throws -> [String: Any] {
        let provider = AnthropicProvider(configuration:
            AnthropicProvider.anthropic(apiKey: "test", model: "claude-3-5-sonnet-20241022"))
        let request = LLMRequest(model: "claude-3-5-sonnet-20241022", messages: messages)
        let body = try #require(try provider.prepareRequest(request, stream: false).httpBody)
        return try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
    }

    @Test func toolResultImageRidesInsideToolResultContent() throws {
        // A minimal 1x1 PNG stands in for a browser screenshot.
        let png = Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")!
        let image = LLMImage(data: png, mimeType: "image/png")

        let messages: [LLMMessage] = [
            .user("Take a screenshot and describe it"),
            .assistant(content: "", toolCalls: [
                LLMToolCall(id: "tu_1", name: "take_snapshot", arguments: "{}")
            ]),
            .tool("Screenshot captured.", images: [image], toolCallId: "tu_1"),
        ]

        let body = try decodedBody(messages)
        let msgs = try #require(body["messages"] as? [[String: Any]])

        // Alternation: user → assistant → user (tool results map to a user turn).
        #expect(msgs.map { $0["role"] as? String } == ["user", "assistant", "user"])

        // The last user turn's tool_result block carries the image inline.
        let toolTurn = msgs[2]
        let blocks = try #require(toolTurn["content"] as? [[String: Any]])
        let toolResult = try #require(blocks.first { $0["type"] as? String == "tool_result" })
        #expect(toolResult["tool_use_id"] as? String == "tu_1")

        let inner = try #require(toolResult["content"] as? [[String: Any]])
        let textBlock = try #require(inner.first { $0["type"] as? String == "text" })
        #expect(textBlock["text"] as? String == "Screenshot captured.")

        let imageBlock = try #require(inner.first { $0["type"] as? String == "image" })
        let source = try #require(imageBlock["source"] as? [String: Any])
        #expect(source["type"] as? String == "base64")
        #expect(source["media_type"] as? String == "image/png")
        #expect(source["data"] as? String == image.base64)
    }

    @Test func textOnlyToolResultStaysAString() throws {
        // Without images, tool_result.content remains a plain string (unchanged).
        let messages: [LLMMessage] = [
            .assistant(content: "", toolCalls: [LLMToolCall(id: "tu_9", name: "noop", arguments: "{}")]),
            .tool("done", toolCallId: "tu_9"),
        ]
        let body = try decodedBody(messages)
        let msgs = try #require(body["messages"] as? [[String: Any]])
        let blocks = try #require(msgs[1]["content"] as? [[String: Any]])
        let toolResult = try #require(blocks.first { $0["type"] as? String == "tool_result" })
        #expect(toolResult["content"] as? String == "done")
    }
}
