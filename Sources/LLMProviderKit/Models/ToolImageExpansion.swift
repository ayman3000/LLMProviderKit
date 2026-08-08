import Foundation

public extension LLMMessage {
    /// Move any images off `.tool` messages onto a following `.user` message.
    ///
    /// OpenAI/Ollama `role:"tool"` messages are text-only and reject image
    /// content, so a tool-result image must be delivered as normal image input
    /// on an adjacent user turn. For each `.tool` message that carries images,
    /// this yields the tool message with its `images` cleared, immediately
    /// followed by a `.user` message holding those images with a short caption.
    /// Non-tool messages (and tool messages without images) pass through.
    static func expandingToolImagesToUserMessages(_ messages: [LLMMessage]) -> [LLMMessage] {
        var out: [LLMMessage] = []
        for message in messages {
            guard message.role == .tool, !message.images.isEmpty else {
                out.append(message)
                continue
            }
            out.append(LLMMessage(role: .tool, content: message.content,
                                  images: [], toolCallId: message.toolCallId))
            out.append(.user("[image output from the previous tool call]", images: message.images))
        }
        return out
    }
}
