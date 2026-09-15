import Foundation

/// The sentence a person should read when a provider refuses a request, and
/// the raw body for whoever has to debug it.
///
/// Providers explain themselves well and then bury it. Google's 404 says "this
/// model is no longer available to new users — use gemini-3.1-pro-preview",
/// which is exactly what the user needs, wrapped in `{"error":{"code":404,…}}`
/// that makes it read like a crash. The wrapper is the problem, not the words.
///
/// So: pull the vendor's own sentence out and show that. Invent copy only when
/// there is none to find.
public struct LLMErrorMessage: Sendable, Equatable {
    /// One sentence, fit to show a user.
    public let summary: String
    /// The whole response body, for a details disclosure and for support.
    /// Nil when the failure carried none.
    public let details: String?

    public init(summary: String, details: String? = nil) {
        self.summary = summary
        self.details = details
    }
}

extension LLMError {
    /// A message worth showing someone, extracted from whatever the provider
    /// sent back.
    public var userMessage: LLMErrorMessage {
        guard case .httpError(let code, let data) = self else {
            return LLMErrorMessage(summary: description)
        }
        let body = data.flatMap { String(data: $0, encoding: .utf8) }
        guard let body, !body.isEmpty else {
            return LLMErrorMessage(summary: Self.cause(for: code))
        }
        guard let sentence = Self.vendorMessage(in: body) else {
            // No recognised envelope: the cause plus the body is still better
            // than a bare status, and the body is all we have to go on.
            return LLMErrorMessage(summary: Self.cause(for: code), details: body)
        }
        return LLMErrorMessage(summary: sentence, details: body)
    }

    /// What a status code means in plain words. Used only when the provider
    /// said nothing useful — its own sentence is always better than ours.
    static func cause(for code: Int) -> String {
        switch code {
        case 401, 403: "The provider rejected your API key. Check it in Settings → Providers."
        case 404:      "That model isn't available on this account."
        case 408:      "The provider took too long to respond."
        case 429:      "You've hit the provider's rate limit. Wait a moment and try again."
        case 500...599: "The provider is having trouble right now. Try again shortly."
        default:       "The provider refused the request (HTTP \(code))."
        }
    }

    /// The vendor's own sentence, from the envelopes they actually use:
    /// `{"error":{"message":…}}` (Google, OpenAI, Anthropic), `{"message":…}`,
    /// and `{"error":"…"}` where the error is a bare string.
    static func vendorMessage(in body: String) -> String? {
        guard let data = body.data(using: .utf8),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        if let error = root["error"] as? [String: Any],
           let message = error["message"] as? String, !message.isEmpty {
            return message
        }
        if let error = root["error"] as? String, !error.isEmpty { return error }
        if let message = root["message"] as? String, !message.isEmpty { return message }
        // Ollama answers with a bare {"error":"…"} too, and some gateways nest
        // one level deeper under "detail".
        if let detail = root["detail"] as? [String: Any],
           let message = detail["message"] as? String, !message.isEmpty {
            return message
        }
        if let detail = root["detail"] as? String, !detail.isEmpty { return detail }
        return nil
    }
}

extension Error {
    /// A user-facing message for any error, so a call site does not have to
    /// know whether it is holding an `LLMError`.
    public var llmUserMessage: LLMErrorMessage {
        (self as? LLMError)?.userMessage ?? LLMErrorMessage(summary: localizedDescription)
    }
}
