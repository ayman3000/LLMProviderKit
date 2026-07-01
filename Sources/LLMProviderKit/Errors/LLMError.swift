import Foundation

/// Errors thrown by `LLMProviderKit` and its provider targets.
public enum LLMError: Error, Sendable, Equatable, LocalizedError, CustomStringConvertible {
    /// The request could not be encoded for the provider.
    case invalidRequest(String)

    /// The response could not be decoded.
    case invalidResponse(String)

    /// The provider returned an HTTP error status.
    case httpError(Int, Data?)

    /// A streaming chunk could not be parsed.
    case streamingError(String)

    /// The provider emitted an error inside the stream.
    case providerError(String)

    /// A network or URLSession failure.
    case networkError(String)

    /// The provider does not support the requested operation.
    case unsupportedOperation(String)

    /// No provider is registered under the requested name.
    case unknownProvider(String)

    public var errorDescription: String? { description }

    public var localizedDescription: String { description }

    public var description: String {
        switch self {
        case .invalidRequest(let reason):
            return "Invalid request: \(reason)"
        case .invalidResponse(let reason):
            return "Invalid response: \(reason)"
        case .httpError(let code, let data):
            if let data, let body = String(data: data, encoding: .utf8), !body.isEmpty {
                return "HTTP error \(code): \(body.prefix(2_000))"
            }
            return "HTTP error: \(code)"
        case .streamingError(let reason):
            return "Streaming error: \(reason)"
        case .providerError(let reason):
            return "Provider error: \(reason)"
        case .networkError(let reason):
            return "Network error: \(reason)"
        case .unsupportedOperation(let reason):
            return "Unsupported operation: \(reason)"
        case .unknownProvider(let name):
            return "Unknown provider: \(name)"
        }
    }
}
