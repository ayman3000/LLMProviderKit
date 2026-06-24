import Foundation

/// A convenience facade that lets you pick a provider by name.
///
/// Apps can either use provider types directly (`OllamaProvider`) or configure
/// an `LLMService` once and call `complete`/`stream` through it.
public actor LLMService {
    private var providers: [String: any LLMProvider] = [:]

    public init() {}

    /// Register a provider instance.
    public func register(_ provider: any LLMProvider) {
        providers[type(of: provider).name] = provider
    }

    /// Register multiple providers.
    public func register(_ providers: [any LLMProvider]) {
        for provider in providers {
            register(provider)
        }
    }

    /// Get a registered provider by name.
    public func provider(named name: String) throws -> any LLMProvider {
        guard let provider = providers[name] else {
            throw LLMError.unknownProvider(name)
        }
        return provider
    }

    /// Non-streaming completion through the named provider.
    public func complete(
        provider name: String,
        request: LLMRequest
    ) async throws -> LLMResponse {
        let provider = try provider(named: name)
        return try await provider.complete(request)
    }

    /// Streaming completion through the named provider.
    public func stream(
        provider name: String,
        request: LLMRequest
    ) throws -> AsyncThrowingStream<LLMStreamChunk, Error> {
        let provider = try provider(named: name)
        return provider.stream(request)
    }
}
