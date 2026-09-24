import Foundation

/// When the model last made progress on a stream.
public final class LLMProgressClock: @unchecked Sendable {
    private let lock = NSLock()
    private var last = Date()
    public init() {}
    public func touch() { lock.lock(); last = Date(); lock.unlock() }
    public var idle: TimeInterval { lock.lock(); defer { lock.unlock() }; return Date().timeIntervalSince(last) }
}

/// Ends a stream that stops making progress.
///
/// Progress is any real event from the model — text, thinking, tool-call
/// pieces (even ones a parser holds until the call is whole). Keep-alives are
/// not: OpenRouter's `: OPENROUTER PROCESSING` comments and Anthropic's
/// `ping` events arrive while nothing is happening, which is exactly how one
/// real call sat for 8.5 minutes: the connection never went idle, so its
/// timeout never fired.
public enum LLMStreamWatchdog {
    /// A stream line that keeps the connection alive but is not model progress.
    public static func isKeepAlive(_ line: String) -> Bool {
        let t = line.trimmingCharacters(in: .whitespaces)
        if t.isEmpty || t.hasPrefix(":") { return true }                  // SSE comment / separator
        if t == "event: ping" || t == "event:ping" { return true }       // Anthropic
        if t.hasPrefix("data:"), t.contains(#""type":"ping""#) || t.contains(#""type": "ping""#) { return true }
        return false
    }

    /// Run `body` (which calls `clock.touch()` on progress). If `limit` seconds
    /// pass without progress, `body` is cancelled and `LLMStreamStalled` thrown.
    /// A nil limit runs `body` unwatched.
    public static func run(limit: TimeInterval?, clock: LLMProgressClock,
                           _ body: @escaping @Sendable () async throws -> Void) async throws {
        guard let limit else { return try await body() }
        clock.touch()
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await body() }
            group.addTask {
                while true {
                    try await Task.sleep(nanoseconds: 250_000_000)
                    if clock.idle > limit { throw LLMStreamStalled(seconds: limit) }
                }
            }
            try await group.next()      // body finished, or the watch fired
            group.cancelAll()
        }
    }
}
