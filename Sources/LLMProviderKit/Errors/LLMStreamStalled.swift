import Foundation

/// A streamed response made no progress for `seconds` (see
/// `LLMRequest.stallTimeout`). Keep-alives had arrived or the connection was
/// silent; either way the model produced nothing. Safe to retry when no answer
/// text has been shown yet.
public struct LLMStreamStalled: Error, LocalizedError, Sendable, Equatable {
    public let seconds: TimeInterval
    public init(seconds: TimeInterval) { self.seconds = seconds }
    public var errorDescription: String? {
        "The model stopped responding (no progress for \(Int(seconds)) s)."
    }
}
