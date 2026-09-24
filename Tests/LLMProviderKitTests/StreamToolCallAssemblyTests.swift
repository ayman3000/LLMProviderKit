import Foundation
@testable import LLMProviderKit
@testable import LLMProviderKitOpenAI
import Testing

/// OpenAI-style streams send one tool call as fragments: the first carries
/// id + name + the start of the arguments, later ones only more argument text,
/// all tagged with the call's `index`. Passed on raw, a nameless fragment makes
/// the agent ask the model the whole question again.
struct StreamToolCallAssemblyTests {
    private func calls(_ chunks: [LLMStreamChunk]) -> [LLMToolCall] {
        chunks.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }
    }

    @Test func openAIFragmentCarriesItsIndex() throws {
        let provider = OpenAIProvider(configuration: OpenAIProvider.openAI(apiKey: "k", model: "x"))
        let request = LLMRequest(model: "x", messages: [.user("Hi")])
        let first = try provider.parseStreamLine(
            #"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"get_weather","arguments":"{\"ci"}}]}}]}"#,
            request: request)
        let later = try provider.parseStreamLine(
            #"data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"ty\":\"Cairo\"}"}}]}}]}"#,
            request: request)
        #expect(calls(first).first?.providerMetadata[StreamToolCallAssembler.indexKey] == "1")
        #expect(calls(later).first?.providerMetadata[StreamToolCallAssembler.indexKey] == "1")
        #expect(calls(later).first?.name == "")
    }

    @Test func fragmentsBecomeWholeCallsBeforeFinish() {
        var a = StreamToolCallAssembler()
        let k = StreamToolCallAssembler.indexKey
        var out: [LLMStreamChunk] = []
        out += a.consume(.text("Checking."))
        out += a.consume(.toolCall(LLMToolCall(id: "call_a", name: "get_weather", arguments: #"{"ci"#, providerMetadata: [k: "0"])))
        out += a.consume(.toolCall(LLMToolCall(id: "call_b", name: "get_time", arguments: "", providerMetadata: [k: "1"])))
        out += a.consume(.toolCall(LLMToolCall(id: "x1", name: "", arguments: #"ty":"Cairo"}"#, providerMetadata: [k: "0"])))
        out += a.consume(.toolCall(LLMToolCall(id: "x2", name: "", arguments: "{}", providerMetadata: [k: "1"])))
        out += a.consume(.finish(reason: .toolCalls, usage: nil))

        let whole = calls(out)
        #expect(whole.count == 2)
        #expect(whole[0].id == "call_a" && whole[0].name == "get_weather" && whole[0].arguments == #"{"city":"Cairo"}"#)
        #expect(whole[1].id == "call_b" && whole[1].name == "get_time" && whole[1].arguments == "{}")
        #expect(whole.allSatisfy { $0.providerMetadata[k] == nil })
        // Order: text, the whole calls, then finish.
        if case .text = out.first {} else { Issue.record("text must come first") }
        if case .finish = out.last {} else { Issue.record("finish must come last") }
    }

    @Test func wholeCallsWithoutAnIndexPassStraightThrough() {
        var a = StreamToolCallAssembler()
        let call = LLMToolCall(id: "t1", name: "read_file", arguments: #"{"path":"a"}"#)
        #expect(calls(a.consume(.toolCall(call))) == [call])
        #expect(a.flush().isEmpty)
    }

    @Test func pendingFragmentsAreFlushedWhenTheStreamEndsWithoutFinish() {
        var a = StreamToolCallAssembler()
        let k = StreamToolCallAssembler.indexKey
        _ = a.consume(.toolCall(LLMToolCall(id: "call_a", name: "ls", arguments: "{", providerMetadata: [k: "0"])))
        _ = a.consume(.toolCall(LLMToolCall(id: "y", name: "", arguments: "}", providerMetadata: [k: "0"])))
        let flushed = calls(a.flush())
        #expect(flushed.count == 1 && flushed[0].name == "ls" && flushed[0].arguments == "{}")
    }

    @Test func aCallWithNoArgumentTextGetsAnEmptyObject() {
        var a = StreamToolCallAssembler()
        let k = StreamToolCallAssembler.indexKey
        _ = a.consume(.toolCall(LLMToolCall(id: "call_a", name: "now", arguments: "", providerMetadata: [k: "0"])))
        #expect(calls(a.consume(.finish(reason: .toolCalls, usage: nil))).first?.arguments == "{}")
    }
}

// MARK: - End to end through the shared stream loop

private final class SSEStreamURLProtocol: URLProtocol {
    nonisolated(unsafe) static var body = Data()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": "text/event-stream"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Self.body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

/// An OpenAI-wire provider on a stubbed session, parsing with the real OpenAI parser.
private struct StubbedOpenAIWire: LLMProvider {
    static let name = "openai-wire-test"
    let configuration: LLMProviderConfiguration
    let urlSession: URLSession
    private let inner: OpenAIProvider

    init() {
        let c = URLSessionConfiguration.ephemeral
        c.protocolClasses = [SSEStreamURLProtocol.self]
        urlSession = URLSession(configuration: c)
        configuration = OpenAIProvider.openAI(apiKey: "k", model: "m")
        inner = OpenAIProvider(configuration: configuration)
    }
    func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        var r = URLRequest(url: URL(string: "https://example.invalid/v1/chat/completions")!)
        r.httpMethod = "POST"; r.httpBody = Data("{}".utf8)
        return r
    }
    func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        try inner.parseStreamLine(line, request: request)
    }
    func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        try inner.parseResponse(data, request: request)
    }
}

@Suite(.serialized)
struct StreamLoopToolCallAssemblyTests {
    @Test func theStreamDeliversOneWholeCallFromFragments() async throws {
        SSEStreamURLProtocol.body = Data([
            #"data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant","content":"Let me check."}}]}"#,
            #"data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#,
            #"data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\""}}]}}]}"#,
            #"data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"Cairo\"}"}}]}}]}"#,
            #"data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            "data: [DONE]",
            "",
        ].joined(separator: "\n").utf8)

        var calls: [LLMToolCall] = []
        var text = ""
        for try await chunk in StubbedOpenAIWire().stream(LLMRequest(model: "m", messages: [.user("Weather?")])) {
            switch chunk {
            case .toolCall(let c): calls.append(c)
            case .text(let t): text += t
            default: break
            }
        }
        #expect(text == "Let me check.")
        #expect(calls.count == 1)
        #expect(calls.first?.id == "call_1")
        #expect(calls.first?.name == "get_weather")
        #expect(calls.first?.arguments == #"{"city":"Cairo"}"#)
    }
}
