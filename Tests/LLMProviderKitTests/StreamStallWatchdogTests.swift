import Foundation
@testable import LLMProviderKit
import Testing

/// A stream that stops making progress must end with `LLMStreamStalled` once
/// `request.stallTimeout` passes — and keep-alives must not count as progress.
/// Found in a real run: one model call sat for 8.5 minutes, undetected,
/// because bytes still trickled in, so the connection's idle timeout never fired.

/// Scripted server: sends `lines` with `gap` seconds between them, then either
/// finishes or goes silent forever (the stall).
private final class ScriptedStreamURLProtocol: URLProtocol {
    nonisolated(unsafe) static var lines: [String] = []
    nonisolated(unsafe) static var gap: TimeInterval = 0.2
    nonisolated(unsafe) static var finish = true
    private var stopped = false
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": "text/event-stream"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        let (lines, gap, finish) = (Self.lines, Self.gap, Self.finish)
        DispatchQueue.global().async {
            for line in lines {
                Thread.sleep(forTimeInterval: gap)
                if self.stopped { return }
                self.client?.urlProtocol(self, didLoad: Data((line + "\n").utf8))
            }
            if finish, !self.stopped { self.client?.urlProtocolDidFinishLoading(self) }
        }
    }
    override func stopLoading() { stopped = true }
}

private struct ScriptedProvider: LLMProvider {
    static let name = "scripted"
    let configuration = LLMProviderConfiguration(name: ScriptedProvider.name,
                                                 baseURL: URL(string: "https://example.invalid")!, apiKey: "k", defaultModel: "m")
    let urlSession: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [ScriptedStreamURLProtocol.self]
        return URLSession(configuration: c)
    }()
    func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        var r = URLRequest(url: configuration.baseURL); r.httpMethod = "POST"; r.httpBody = Data("{}".utf8); return r
    }
    func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        line.hasPrefix("data: ") ? [.text(String(line.dropFirst(6)))] : []
    }
    func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        LLMResponse(text: "", finishReason: .stop, request: request, providerName: Self.name)
    }
}

@Suite(.serialized, .timeLimit(.minutes(1)))
struct StreamStallWatchdogTests {
    private func run(lines: [String], gap: TimeInterval, finish: Bool, stallTimeout: TimeInterval?) async -> (text: String, error: (any Error)?, seconds: TimeInterval) {
        ScriptedStreamURLProtocol.lines = lines
        ScriptedStreamURLProtocol.gap = gap
        ScriptedStreamURLProtocol.finish = finish
        var request = LLMRequest(model: "m", messages: [.user("hi")])
        request.stallTimeout = stallTimeout
        let start = Date()
        var text = ""
        do {
            for try await chunk in ScriptedProvider().stream(request) { if case .text(let t) = chunk { text += t } }
            return (text, nil, Date().timeIntervalSince(start))
        } catch { return (text, error, Date().timeIntervalSince(start)) }
    }

    @Test func aSilentStreamEndsWithStalled() async {
        let r = await run(lines: ["data: a"], gap: 0.1, finish: false, stallTimeout: 1)
        #expect(r.error is LLMStreamStalled, "\(String(describing: r.error))")
        #expect(r.seconds < 3, "gave up after \(r.seconds)s")
        #expect(r.text == "a")
    }

    @Test func keepAlivesAreNotProgress() async {
        let keepAlives = Array(repeating: ": OPENROUTER PROCESSING", count: 12)
        let r = await run(lines: ["data: a"] + keepAlives, gap: 0.25, finish: false, stallTimeout: 1)
        #expect(r.error is LLMStreamStalled, "keep-alives must not reset the watchdog: \(String(describing: r.error))")
    }

    @Test func aStreamThatKeepsProgressingIsNeverCut() async {
        let lines = (0..<10).map { "data: \($0)" }
        let r = await run(lines: lines, gap: 0.3, finish: true, stallTimeout: 1)
        #expect(r.error == nil, "\(String(describing: r.error))")
        #expect(r.text == "0123456789")
    }

    @Test func eventLinesWithoutParsedChunksStillCountAsProgress() async {
        // e.g. tool-call argument deltas a parser ignores until the call completes.
        let lines = ["data: a"] + Array(repeating: "event: response.function_call_arguments.delta", count: 8) + ["data: b"]
        let r = await run(lines: lines, gap: 0.3, finish: true, stallTimeout: 1)
        #expect(r.error == nil, "\(String(describing: r.error))")
    }

    @Test func noStallTimeoutMeansNoWatchdog() async {
        let r = await run(lines: ["data: a", "data: b"], gap: 0.6, finish: true, stallTimeout: nil)
        #expect(r.error == nil)
        #expect(r.text == "ab")
    }
}
