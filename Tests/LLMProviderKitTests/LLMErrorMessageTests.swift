import Testing
import Foundation
@testable import LLMProviderKit

/// Providers explain themselves well and then bury it. The job here is to show
/// the vendor's own sentence and keep the envelope out of a user's way.
struct LLMErrorMessageTests {
    private func message(_ code: Int, _ body: String?) -> LLMErrorMessage {
        LLMError.httpError(code, body.map { Data($0.utf8) }).userMessage
    }

    /// The real one, verbatim from Google — the case that started this.
    @Test func googlesSentenceIsWhatTheUserReads() {
        let body = """
        {"error":{"code":404,"message":"This model models/gemini-2.5-pro is no longer \
        available to new users. Please update your code to use \
        models/gemini-3.1-pro-preview.","status":"NOT_FOUND"}}
        """
        let m = message(404, body)
        #expect(m.summary.hasPrefix("This model models/gemini-2.5-pro is no longer available"))
        // No envelope in the sentence a person reads.
        #expect(!m.summary.contains("{"))
        #expect(!m.summary.contains("NOT_FOUND"))
        // …and the whole body is still there for support.
        #expect(m.details?.contains("NOT_FOUND") == true)
    }

    /// The other real one: an unsupported effort level.
    @Test func anInvalidParameterReadsAsASentence() {
        let body = """
        {"error":{"message":"Invalid value: 'ultra'. Supported values are: 'low', \
        'medium', 'high'.","type":"invalid_request_error","param":"reasoning.effort"}}
        """
        #expect(message(400, body).summary.hasPrefix("Invalid value: 'ultra'"))
    }

    @Test func theOtherEnvelopesAreUnderstood() {
        #expect(message(500, #"{"error":"model runner crashed"}"#).summary == "model runner crashed")
        #expect(message(400, #"{"message":"bad request"}"#).summary == "bad request")
        #expect(message(422, #"{"detail":{"message":"nope"}}"#).summary == "nope")
        #expect(message(422, #"{"detail":"nope"}"#).summary == "nope")
    }

    /// With no body there is nothing to quote, so say what the status means
    /// rather than printing the number alone.
    @Test func aBodylessFailureStillSaysSomethingUseful() {
        #expect(message(401, nil).summary.contains("API key"))
        #expect(message(429, nil).summary.contains("rate limit"))
        #expect(message(503, nil).summary.contains("trouble"))
        #expect(message(418, nil).summary.contains("418"))
        #expect(message(401, nil).details == nil)
    }

    /// An unparseable body must not be lost — it is all the evidence there is.
    @Test func anUnrecognisedBodyIsKeptAsDetails() {
        let m = message(500, "<html>gateway exploded</html>")
        #expect(m.summary.contains("trouble"))
        #expect(m.details == "<html>gateway exploded</html>")
    }

    /// Non-HTTP errors pass through unchanged rather than being reshaped.
    @Test func otherErrorsKeepTheirOwnDescription() {
        #expect(LLMError.networkError("offline").userMessage.summary.contains("offline"))
        #expect((URLError(.timedOut) as Error).llmUserMessage.summary.isEmpty == false)
    }
}
