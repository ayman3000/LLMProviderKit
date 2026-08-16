import Foundation
import Testing
@testable import LLMProviderKit
@testable import LLMProviderKitAnthropic

// MARK: - Static-list mock providers

private func testModel(
    _ id: String,
    provider: String,
    releaseStage: LLMModelReleaseStage? = nil,
    isDeprecated: Bool = false
) -> LLMModelInfo {
    LLMModelInfo(
        id: id,
        providerName: provider,
        displayName: id,
        contextWindow: nil,
        capabilities: [.chat],
        categories: [.text],
        releaseStage: releaseStage,
        isDeprecated: isDeprecated
    )
}

private struct StaticModelsProviderA: LLMProvider {
    static let name = "static-a"
    let configuration = LLMProviderConfiguration(name: name, baseURL: URL(string: "inprocess://a")!)
    let models: [LLMModelInfo]

    func availableModels() async throws -> [LLMModelInfo] { models }
    func complete(_ request: LLMRequest) async throws -> LLMResponse {
        LLMResponse(text: "", finishReason: .stop, request: request, providerName: Self.name)
    }
    func stream(_ request: LLMRequest) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}

private struct StaticModelsProviderB: LLMProvider {
    static let name = "static-b"
    let configuration = LLMProviderConfiguration(name: name, baseURL: URL(string: "inprocess://b")!)
    let models: [LLMModelInfo]

    func availableModels() async throws -> [LLMModelInfo] { models }
    func complete(_ request: LLMRequest) async throws -> LLMResponse {
        LLMResponse(text: "", finishReason: .stop, request: request, providerName: Self.name)
    }
    func stream(_ request: LLMRequest) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}

private struct FailingModelsProvider: LLMProvider {
    static let name = "failing"
    let configuration = LLMProviderConfiguration(name: name, baseURL: URL(string: "inprocess://fail")!)

    func availableModels() async throws -> [LLMModelInfo] {
        throw LLMError.networkError("simulated outage")
    }
    func complete(_ request: LLMRequest) async throws -> LLMResponse {
        LLMResponse(text: "", finishReason: .stop, request: request, providerName: Self.name)
    }
    func stream(_ request: LLMRequest) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { $0.finish() }
    }
}

// MARK: - refreshAll

struct RefreshAllTests {
    @Test func refreshAllAggregatesAcrossProvidersAndFiltersObsolete() async throws {
        let providerA = StaticModelsProviderA(models: [
            testModel("a-current", provider: "static-a", releaseStage: .stable),
            testModel("a-legacy", provider: "static-a", releaseStage: .legacy),
        ])
        let providerB = StaticModelsProviderB(models: [
            testModel("b-current", provider: "static-b"),
            testModel("b-deprecated", provider: "static-b", isDeprecated: true),
        ])

        let registry = LLMModelRegistry()
        let result = await registry.refreshAll(from: [providerA, providerB])

        #expect(result.failures.isEmpty)
        #expect(result.models.map(\.id) == ["a-current", "b-current"])
        // Registry itself keeps everything; filtering applies to the returned list.
        #expect(await registry.allModels().count == 4)
    }

    @Test func refreshAllCanIncludeDeprecatedModels() async throws {
        let providerA = StaticModelsProviderA(models: [
            testModel("a-current", provider: "static-a", releaseStage: .stable),
            testModel("a-legacy", provider: "static-a", releaseStage: .legacy),
        ])

        let registry = LLMModelRegistry()
        let result = await registry.refreshAll(from: [providerA], includeDeprecated: true)

        #expect(result.models.map(\.id) == ["a-current", "a-legacy"])
    }

    @Test func refreshAllReturnsPartialResultsWhenAProviderFails() async throws {
        let providerA = StaticModelsProviderA(models: [
            testModel("a-current", provider: "static-a", releaseStage: .stable),
        ])
        let failing = FailingModelsProvider()

        let registry = LLMModelRegistry()
        let result = await registry.refreshAll(from: [failing, providerA])

        #expect(result.models.map(\.id) == ["a-current"])
        #expect(result.failures.count == 1)
        #expect(result.failures["failing"] != nil)
    }

    @Test func modelInfoIsObsoleteCoversDeprecatedAndLegacy() {
        #expect(testModel("m", provider: "p", releaseStage: .legacy).isObsolete)
        #expect(testModel("m", provider: "p", releaseStage: .deprecated).isObsolete)
        #expect(testModel("m", provider: "p", isDeprecated: true).isObsolete)
        #expect(!testModel("m", provider: "p", releaseStage: .stable).isObsolete)
        #expect(!testModel("m", provider: "p").isObsolete)
    }
}

// MARK: - Anthropic live model listing

private final class AnthropicModelsMockURLProtocol: URLProtocol {
    private static let lock = NSLock()
    private static var responseDataByAPIKey: [String: Data] = [:]
    private static var failingAPIKeys: Set<String> = []

    static func setResponseData(_ data: Data, forAPIKey apiKey: String) {
        lock.lock(); defer { lock.unlock() }
        responseDataByAPIKey[apiKey] = data
    }

    static func setFailure(forAPIKey apiKey: String) {
        lock.lock(); defer { lock.unlock() }
        failingAPIKeys.insert(apiKey)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let apiKey = request.value(forHTTPHeaderField: "x-api-key") ?? ""
        let (data, shouldFail): (Data?, Bool) = Self.lock.withLock {
            (Self.responseDataByAPIKey[apiKey], Self.failingAPIKeys.contains(apiKey))
        }
        if shouldFail {
            client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
            return
        }
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data ?? Data())
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

struct AnthropicAvailableModelsTests {
    private func provider(apiKey: String) -> AnthropicProvider {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [AnthropicModelsMockURLProtocol.self]
        return AnthropicProvider(
            configuration: AnthropicProvider.anthropic(apiKey: apiKey),
            urlSession: URLSession(configuration: configuration)
        )
    }

    @Test func anthropicAvailableModelsFetchesLiveAndEnrichesWithCuratedMetadata() async throws {
        let apiKey = "anthropic-live-models-test"
        AnthropicModelsMockURLProtocol.setResponseData("""
        {
          "data": [
            {"type": "model", "id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6", "created_at": "2025-09-29T00:00:00Z"},
            {"type": "model", "id": "claude-brand-new-model", "display_name": "Claude Brand New", "created_at": "2026-08-01T00:00:00Z"}
          ],
          "has_more": false
        }
        """.data(using: .utf8)!, forAPIKey: apiKey)

        let models = try await provider(apiKey: apiKey).availableModels()

        #expect(models.map(\.id) == ["claude-sonnet-4-6", "claude-brand-new-model"])
        // Live record enriched with curated metadata (context window comes from the curated list).
        let sonnet = try #require(models.first { $0.id == "claude-sonnet-4-6" })
        #expect(sonnet.contextWindow == 1_000_000)
        // A model unknown to the curated list still comes through with chat capabilities.
        let fresh = try #require(models.first { $0.id == "claude-brand-new-model" })
        #expect(fresh.displayName == "Claude Brand New")
        #expect(fresh.capabilities.contains(.chat))
    }

    @Test func anthropicAvailableModelsFallsBackToCuratedOnNetworkFailure() async throws {
        let apiKey = "anthropic-offline-test"
        AnthropicModelsMockURLProtocol.setFailure(forAPIKey: apiKey)

        let models = try await provider(apiKey: apiKey).availableModels()

        #expect(models == AnthropicProvider.curatedModels)
    }
}
