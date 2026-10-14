import Foundation

public extension LLMMessage {
    /// Move any images off `.tool` messages onto a following `.user` message.
    ///
    /// OpenAI-style `role:"tool"` messages (and Responses `function_call_output`)
    /// are text-only, so a tool-result image travels as ordinary image input on
    /// a user message. That message goes AFTER the whole run of consecutive
    /// tool results, never between two of them: a reply that made several tool
    /// calls must have all their results right after it, and a user message
    /// wedged between them breaks that on strict wires. So each run of tool
    /// messages yields the tool messages with `images` cleared, then — only if
    /// any carried images — ONE user message with all of them, in call order.
    /// Everything else passes through.
    static func expandingToolImagesToUserMessages(_ messages: [LLMMessage]) -> [LLMMessage] {
        var out: [LLMMessage] = []
        var pendingImages: [LLMImage] = []
        var imageCalls = 0

        func flush() {
            guard !pendingImages.isEmpty else { return }
            let caption = imageCalls == 1
                ? "[image output from the previous tool call]"
                : "[image output from the previous \(imageCalls) tool calls, in call order]"
            out.append(.user(caption, images: pendingImages))
            pendingImages = []
            imageCalls = 0
        }

        for message in messages {
            guard message.role == .tool else {
                flush()
                out.append(message)
                continue
            }
            if message.images.isEmpty {
                out.append(message)
            } else {
                out.append(LLMMessage(role: .tool, content: message.content,
                                      images: [], toolCallId: message.toolCallId))
                pendingImages += message.images
                imageCalls += 1
            }
        }
        flush()
        return out
    }
}
