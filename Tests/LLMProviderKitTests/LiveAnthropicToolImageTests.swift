import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import LLMProviderKit
@testable import LLMProviderKitAnthropic
import Testing

/// LIVE test (gated on ANTHROPIC_API_KEY): proves the real Anthropic API accepts
/// our tool-result-with-image structure AND that the model actually sees the
/// image. Sends a solid-red screenshot as a tool result and expects the model to
/// name the color "red".
///
/// Run with:  ANTHROPIC_API_KEY=sk-ant-… swift test --filter LiveAnthropicToolImageTests
struct LiveAnthropicToolImageTests {

    /// A solid-color PNG rendered without UIKit/AppKit (CoreGraphics + ImageIO).
    private func solidPNG(red: CGFloat, green: CGFloat, blue: CGFloat, size: Int = 64) -> Data {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                            bytesPerRow: 0, space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: red, green: green, blue: blue, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        let image = ctx.makeImage()!
        let out = NSMutableData()
        let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
        return out as Data
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] != nil))
    func modelSeesToolResultImage() async throws {
        let key = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]!
        let model = "claude-3-5-sonnet-20241022"
        let provider = AnthropicProvider(configuration:
            AnthropicProvider.anthropic(apiKey: key, model: model))

        let screenshot = LLMImage(data: solidPNG(red: 1, green: 0, blue: 0), mimeType: "image/png")
        let tool = LLMToolDefinition(
            name: "take_snapshot",
            description: "Capture a screenshot of the current page.",
            parameters: ["type": "object", "properties": [:]])

        let messages: [LLMMessage] = [
            .user("Call take_snapshot, then answer with ONLY the single dominant color you see in the screenshot."),
            .assistant(content: "", toolCalls: [
                LLMToolCall(id: "tu_1", name: "take_snapshot", arguments: "{}")
            ]),
            .tool("Screenshot captured.", images: [screenshot], toolCallId: "tu_1"),
        ]

        let request = LLMRequest(model: model, messages: messages, maxTokens: 100, tools: [tool])
        let response = try await provider.complete(request)

        let text = response.text.lowercased()
        // The API accepted the tool_result-with-image AND the model saw a red image.
        #expect(text.contains("red"), "Expected the model to see red; got: \(response.text)")
    }
}
