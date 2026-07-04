import Foundation

/// Optional provider capability for image generation.
///
/// This deliberately does not live on `LLMProvider` so chat-only providers and
/// existing clients remain unaffected.
public protocol ImageGenerationProvider: LLMProvider {
    func generateImage(_ request: LLMImageGenerationRequest) async throws -> LLMImageGenerationResponse
}

/// Optional provider capability for speech-to-text transcription.
public protocol AudioTranscriptionProvider: LLMProvider {
    func transcribeAudio(_ request: LLMAudioTranscriptionRequest) async throws -> LLMAudioTranscriptionResponse
}

/// Optional provider capability for text-to-speech synthesis.
public protocol TextToSpeechProvider: LLMProvider {
    func synthesizeSpeech(_ request: LLMTextToSpeechRequest) async throws -> LLMTextToSpeechResponse
}

/// Optional provider capability for embedding vectors.
public protocol EmbeddingProvider: LLMProvider {
    func embed(_ request: LLMEmbeddingRequest) async throws -> LLMEmbeddingResponse
}

public struct LLMImageGenerationRequest: Sendable, Codable, Hashable {
    public let prompt: String
    public let model: String?
    public let size: String?
    public let count: Int

    public init(prompt: String, model: String? = nil, size: String? = nil, count: Int = 1) {
        self.prompt = prompt
        self.model = model
        self.size = size
        self.count = count
    }
}

public struct LLMGeneratedImage: Sendable, Codable, Hashable {
    public let data: Data?
    public let url: URL?
    public let mimeType: String?

    public init(data: Data? = nil, url: URL? = nil, mimeType: String? = nil) {
        self.data = data
        self.url = url
        self.mimeType = mimeType
    }
}

public struct LLMImageGenerationResponse: Sendable, Codable, Hashable {
    public let images: [LLMGeneratedImage]
    public let providerName: String
    public let rawData: Data?

    public init(images: [LLMGeneratedImage], providerName: String, rawData: Data? = nil) {
        self.images = images
        self.providerName = providerName
        self.rawData = rawData
    }
}

public struct LLMAudioTranscriptionRequest: Sendable, Hashable {
    public let audioData: Data
    public let mimeType: String
    public let model: String?
    public let language: String?

    public init(audioData: Data, mimeType: String, model: String? = nil, language: String? = nil) {
        self.audioData = audioData
        self.mimeType = mimeType
        self.model = model
        self.language = language
    }
}

public struct LLMAudioTranscriptionResponse: Sendable, Codable, Hashable {
    public let text: String
    public let providerName: String
    public let rawData: Data?

    public init(text: String, providerName: String, rawData: Data? = nil) {
        self.text = text
        self.providerName = providerName
        self.rawData = rawData
    }
}

public struct LLMTextToSpeechRequest: Sendable, Codable, Hashable {
    public let text: String
    public let model: String?
    public let voice: String?
    public let format: String?

    public init(text: String, model: String? = nil, voice: String? = nil, format: String? = nil) {
        self.text = text
        self.model = model
        self.voice = voice
        self.format = format
    }
}

public struct LLMTextToSpeechResponse: Sendable, Hashable {
    public let audioData: Data
    public let mimeType: String
    public let providerName: String
    public let rawData: Data?

    public init(audioData: Data, mimeType: String, providerName: String, rawData: Data? = nil) {
        self.audioData = audioData
        self.mimeType = mimeType
        self.providerName = providerName
        self.rawData = rawData
    }
}

public struct LLMEmbeddingRequest: Sendable, Codable, Hashable {
    public let input: [String]
    public let model: String?

    public init(input: [String], model: String? = nil) {
        self.input = input
        self.model = model
    }
}

public struct LLMEmbeddingResponse: Sendable, Codable, Hashable {
    public let embeddings: [[Double]]
    public let providerName: String
    public let rawData: Data?

    public init(embeddings: [[Double]], providerName: String, rawData: Data? = nil) {
        self.embeddings = embeddings
        self.providerName = providerName
        self.rawData = rawData
    }
}
