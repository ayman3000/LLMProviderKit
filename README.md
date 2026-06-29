<p align="center">
  <h1 align="center">LLMProviderKit</h1>
  <p align="center"><strong>One protocol, every LLM provider. Native Swift.</strong></p>
  <p align="center">Ollama · OpenAI · Gemini · Anthropic · Streaming · Tool calling · Vision</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Swift-5.9%2B-orange" alt="Swift 5.9+">
  <img src="https://img.shields.io/badge/platforms-macOS%2013%2B%20%7C%20iOS%2016%2B%20%7C%20tvOS%2016%2B%20%7C%20watchOS%209%2B%20%7C%20visionOS%201%2B-blue" alt="Platforms">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
  <img src="https://img.shields.io/badge/zero%20deps-Foundation%20only-success" alt="Zero deps">
</p>

---

> **Change the provider, keep the rest of your app unchanged.**

A protocol-oriented Swift package that unifies chat completions, streaming, tool calling, and vision across multiple LLM providers — with native provider APIs, not OpenAI-compat wrappers.

```swift
let provider: any LLMProvider

provider = OllamaProvider(configuration: .local(model: "llama3.2"))
provider = OpenAIProvider(configuration: .openAI(apiKey: "sk-…", model: "gpt-4o"))
provider = GeminiProvider(configuration: .gemini(apiKey: "…", model: "gemini-2.5-flash"))
provider = AnthropicProvider(configuration: .anthropic(apiKey: "…", model: "claude-sonnet-4"))

let response = try await provider.complete(request)      // non-streaming
for try await chunk in provider.stream(request) { … }    // streaming
```

Currently supported providers:

- **Ollama** (`LLMProviderKitOllama`) — local-first, auto-discovers available models and picks the first one if none is configured. Custom host URL supported.
- **OpenAI** (`LLMProviderKitOpenAI`) — also works with any OpenAI-compatible endpoint (Groq, Together, etc.).
- **Google Gemini** (`LLMProviderKitGemini`)
- **Anthropic** (`LLMProviderKitAnthropic`)

> **LLMProviderKit talks to models. [SwiftAgentKit](https://github.com/ayman3000/SwiftAgentKit) lets models act** — it adds tools, memory, planning, sessions, and a ReAct loop on top of LLMProviderKit.

---

## Table of Contents

- [Why LLMProviderKit?](#why-llmproviderkit)
- [Who is this for?](#who-is-this-for)
- [Features](#features)
- [Design](#design)
- [Installation](#installation)
- [Usage](#usage)
- [Design Principles](#design-principles)
- [Adding a Provider](#adding-a-provider)
- [Testing](#testing)
- [Support](#support)
- [License](#license)

---

## Why LLMProviderKit?

Every provider has a different API, streaming format, auth style, tool format, and vision format.

LLMProviderKit hides that behind one Swift protocol while still using each provider's native API — not an OpenAI-compat shim.

| Without LLMProviderKit | With LLMProviderKit |
|---|---|
| Different request/response shapes per provider | One `LLMRequest` → one `LLMResponse` |
| Different streaming formats (SSE vs chunked) | One `AsyncThrowingStream<LLMStreamChunk>` |
| Different tool-call wire formats | One `LLMToolDefinition` → one `LLMToolCall` |
| Different image encoding per provider | One `LLMImage` → native per-provider encoding |
| Rewrite your app when switching providers | Change one line: `provider = …` |

---

## Who is this for?

LLMProviderKit is for Swift developers who need:

- **Multi-provider LLM access** — one protocol, swap providers with one line
- **Local-first apps** — Ollama support with auto model discovery
- **Streaming chat** — token-by-token streaming for any provider
- **Tool calling** — native tool definitions and tool-call parsing for all 4 providers
- **Vision/multimodal** — send images to vision-capable models
- **A model picker UI** — offline-friendly model registry with curated lists
- **A lightweight networking layer** — Foundation only, zero external dependencies

If you just need to talk to an LLM from Swift — without rewriting your code when you switch providers — LLMProviderKit is built for you.

---

## Features

| Feature | Description |
|---|---|
| 🔄 **Multi-provider** | One `LLMProvider` protocol — Ollama, OpenAI, Gemini, Anthropic. Swap with one line. |
| 🌊 **Streaming** | Token-by-token streaming via `AsyncThrowingStream` for all providers. |
| 🔧 **Tool calling** | Native tool definitions + tool-call parsing for all 4 providers. |
| 🖼️ **Vision/multimodal** | Send images to vision-capable models. Per-provider native encoding. |
| 🖥️ **Local LLMs** | Full Ollama support with auto model discovery (`GET /api/tags`). |
| 📋 **Model registry** | Offline-friendly `LLMModelRegistry` with curated lists and merge strategies. |
| 🏗️ **Per-provider SPM targets** | Import only the providers you need — keeps binary size small. |
| 🔌 **OpenAI-compatible endpoints** | Groq, Together, any OpenAI-compat API — just pass a custom base URL. |
| 📡 **Unified service facade** | `LLMService` for multi-provider routing in one call. |
| ⚡ **Async/await** | Native Swift concurrency throughout. No completion handlers. |
| 📦 **Zero external deps** | Foundation + URLSession only. No third-party packages. |

---

## Design

| Layer | Responsibility |
|-------|----------------|
| `LLMProviderKit` | Provider-agnostic models (`LLMRequest`, `LLMResponse`, `LLMMessage`, `LLMStreamChunk`), the `LLMProvider` protocol, and the `LLMService` facade. |
| `LLMProviderKitOllama` | Ollama `api/chat` implementation. |
| `LLMProviderKitOpenAI` | OpenAI `/chat/completions` implementation. Also works with any OpenAI-compatible endpoint. |
| `LLMProviderKitGemini` | Gemini `generateContent` / `streamGenerateContent` implementation. |
| `LLMProviderKitAnthropic` | Anthropic Messages API implementation with SSE streaming. |

No external dependencies. Uses `Foundation.URLSession` only.

---

## Installation

Add the package to your `Package.swift`:

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MyApp",
    dependencies: [
        .package(url: "https://github.com/ayman3000/LLMProviderKit.git", from: "0.1.0-alpha.1")
    ],
    targets: [
        .executableTarget(
            name: "MyApp",
            dependencies: [
                .product(name: "LLMProviderKit", package: "LLMProviderKit"),
                .product(name: "LLMProviderKitOllama", package: "LLMProviderKit"),
                .product(name: "LLMProviderKitOpenAI", package: "LLMProviderKit"),
                .product(name: "LLMProviderKitGemini", package: "LLMProviderKit"),
                .product(name: "LLMProviderKitAnthropic", package: "LLMProviderKit"),
            ]
        )
    ]
)
```

Or add it in Xcode via **File ▸ Add Package Dependencies** → `https://github.com/ayman3000/LLMProviderKit`

> Only import the provider products your app actually uses to keep binary size small.

> **⚠️ App Sandbox:** If your macOS app uses **App Sandbox** (enabled by default in new Xcode projects), add the `com.apple.security.network.client` entitlement to your `.entitlements` file — otherwise all network calls, including `localhost:11434`, will silently fail.

---

## Usage

### Ollama specifics

```swift
import LLMProviderKitOllama

// Default localhost
let ollama = OllamaProvider(configuration: OllamaProvider.local())

// Custom host (e.g. LAN server or Docker)
let remote = OllamaProvider(
    configuration: OllamaProvider.local(baseURL: URL(string: "http://192.168.1.50:11434")!)
)

// Auto-resolve: leave request model empty, provider fetches /api/tags and uses first model.
let request = LLMRequest(model: "", messages: [.user("Hi")])
let response = try await ollama.complete(request)
print("Resolved model: \(response.request.model)")
```

> **⚠️ App Sandbox:** If your macOS app uses **App Sandbox** (enabled by default), add the `com.apple.security.network.client` entitlement — otherwise all network calls fail silently. See [docs/xcode.md](docs/xcode.md) for full instructions.

### 1. Direct provider

```swift
import LLMProviderKit
import LLMProviderKitOllama

let ollama = OllamaProvider(configuration: OllamaProvider.local(model: "llama3.2"))

let request = LLMRequest(
    model: "llama3.2",
    messages: [
        .system("You are a helpful coding assistant."),
        .user("Write a Swift function that reverses a string.")
    ],
    temperature: 0.7,
    maxTokens: 512
)

let response = try await ollama.complete(request)
print(response.text)
print(response.usage?.totalTokens ?? 0)
```

### 2. Streaming

```swift
let stream = ollama.stream(request)
for try await chunk in stream {
    switch chunk {
    case .text(let text):
        print(text, terminator: "")
    case .finish(let reason, let usage):
        print("\nDone: \(String(describing: reason)), tokens: \(usage?.totalTokens ?? 0)")
    case .error(let error):
        print("Error: \(error)")
    }
}
```

### 3. Unified service facade

```swift
import LLMProviderKit
import LLMProviderKitOllama
import LLMProviderKitOpenAI
import LLMProviderKitGemini

let service = LLMService()
await service.register([
    OllamaProvider(configuration: OllamaProvider.local(model: "llama3.2")),
    OpenAIProvider(configuration: OpenAIProvider.openAI(apiKey: myOpenAIKey, model: "gpt-4o-mini")),
    GeminiProvider(configuration: GeminiProvider.gemini(apiKey: myGeminiKey, model: "gemini-2.0-flash")),
    AnthropicProvider(configuration: AnthropicProvider.anthropic(apiKey: myAnthropicKey, model: "claude-3-5-sonnet-20241022"))
])

let response = try await service.complete(
    provider: "openai",
    request: LLMRequest(model: "gpt-4o-mini", messages: [.user("Hello")])
)
print(response.text)
```

### 4. OpenAI-compatible third-party endpoint

```swift
let config = LLMProviderConfiguration(
    name: "groq",
    baseURL: URL(string: "https://api.groq.com/openai/v1")!,
    apiKey: myGroqKey,
    defaultModel: "llama-3.1-70b"
)
let groq = OpenAIProvider(configuration: config)
```

The `OpenAIProvider` type works for any service using the OpenAI chat-completions JSON shape.

### 5. Model constants

Every provider has a small namespace of well-known model names. They are purely
convenience sugar — you can always pass any string as `LLMRequest.model`.

```swift
import LLMProviderKitOpenAI
import LLMProviderKitGemini
import LLMProviderKitAnthropic
import LLMProviderKitOllama

let request = LLMRequest(
    model: OpenAIModel.gpt4o,
    messages: [.user("Hi")]
)

_ = GeminiModel.flash
_ = AnthropicModel.sonnet
_ = OllamaModel.llama3_2
```

### 6. Model discovery

Providers that expose a model list endpoint can be queried directly:

```swift
let ollama = OllamaProvider(configuration: .local(model: "llama3.2"))
let models = try await ollama.availableModels()
for model in models {
    print("\(model.displayName ?? model.id) - \(model.id)")
}
```

Supported:

- `OllamaProvider.availableModels()` → `GET /api/tags`
- `OpenAIProvider.availableModels()` → `GET /v1/models`
- `GeminiProvider.availableModels()` → `GET /v1beta/models`
- `AnthropicProvider.availableModels()` → returns a curated static list (Anthropic has no public list endpoint)

### 7. Offline-friendly model registry

Apps that need a model picker can use `LLMModelRegistry`. It supports:

- Online refresh from a provider.
- Offline seeding from a developer-curated list.
- Merging online results with curated overrides.

```swift
let registry = LLMModelRegistry()

// Offline: seed a curated list (works without network)
await registry.register([
    LLMModelInfo(id: "gpt-4o", providerName: "openai", displayName: "GPT-4o"),
    LLMModelInfo(id: "gpt-4o-mini", providerName: "openai", displayName: "GPT-4o Mini"),
], for: "openai")

// Online: refresh when the device has connectivity
let openai = OpenAIProvider(configuration: .openAI(apiKey: myKey))
try await registry.refresh(from: openai, strategy: .merge)

// Query
let pickerModels = await registry.models(for: "openai")
let defaultID = try await registry.defaultModelID(for: "openai", configuration: openai.configuration, provider: openai)
```

Merge strategies:

- `.replace` — overwrite with fetched models.
- `.append` — keep existing models, add only new IDs.
- `.merge` — overwrite with fetched models, preserving any existing models for IDs not returned by the provider.

### 8. Multimodal image input

LLMProviderKit supports sending images to vision-capable models. Images are
attached to `LLMMessage` as `LLMImage` values — raw `Data` + a MIME type.
No UIKit or AppKit import is required.

```swift
import LLMProviderKit
import LLMProviderKitOllama

// Load image data (e.g. from a file or NSImage/UIImage representation)
let imageData = Data(contentsOf: URL(fileURLWithPath: "/path/to/photo.png"))!

let request = LLMRequest(
    model: "llama3.2-vision",
    messages: [
        .user("What's in this image?", images: [
            LLMImage(data: imageData, mimeType: "image/png")
        ])
    ]
)

let ollama = OllamaProvider(configuration: OllamaProvider.local())
let response = try await ollama.complete(request)
print(response.text)
```

Each provider encodes images in its native format:

| Provider | Encoding |
|---|---|
| **Ollama** | `"images": ["<base64>"]` in the message object |
| **OpenAI** | `content` becomes `[{type:"text",...}, {type:"image_url",image_url:{url:"data:<mime>;base64,<b64>"}}]` |
| **Anthropic** | `content` becomes `[{type:"text",...}, {type:"image",source:{type:"base64",media_type":"<mime>","data":"<b64>"}}]` |
| **Gemini** | `parts` includes `{inlineData:{mimeType:"<mime>",data:"<b64>"}}` |

Text-only requests are **completely unchanged** — image fields are only emitted
when `message.images` is non-empty.

> Images are only valid for vision-capable models (e.g. `gpt-4o`, `llama3.2-vision`,
> `gemini-2.5-flash`, `claude-3-5-sonnet`). If the model doesn't support images,
> the provider will return its standard HTTP error.

### 9. Tool calling (all providers)

LLMProviderKit supports native tool calling across all four providers — the model
decides whether to call a tool, and the provider handles the per-provider wire format.

```swift
import LLMProviderKit
import LLMProviderKitOpenAI

let provider = OpenAIProvider(configuration: .openAI(apiKey: "sk-…", model: "gpt-4o"))

let request = LLMRequest(
    model: "gpt-4o",
    messages: [.user("What's the weather in Tokyo?")],
    tools: [
        LLMToolDefinition(
            name: "get_weather",
            description: "Get current weather for a city",
            parameters: [
                "type": "object",
                "properties": [
                    "city": ["type": "string", "description": "City name"]
                ],
                "required": ["city"]
            ]
        )
    ]
)

let response = try await provider.complete(request)

// Model chose to call a tool
if let toolCalls = response.toolCalls, let call = toolCalls.first {
    print("Tool: \(call.name)")       // get_weather
    print("Args: \(call.arguments)")  // {"city":"Tokyo"}
    print("ID:  \(call.id)")          // call_abc123
}
```

Each provider maps tool calls to its native format:

| Provider | Tool definitions | Tool calls in response | Tool results |
|---|---|---|---|
| **Ollama** | `{"tools": [...]}` | `tool_calls` array | `role: "tool"` + `tool_call_id` |
| **OpenAI** | `{"tools": [...], "tool_choice": "auto"}` | `tool_calls` array | `role: "tool"` + `tool_call_id` |
| **Anthropic** | `{"tools": [{name, description, input_schema}]}` | `tool_use` content blocks | `role: "user"` + `tool_result` blocks |
| **Gemini** | `{"tools": [{"functionDeclarations": [...]}]}` | `functionCall` parts | `role: "model"` + `functionResponse` parts |

To close the tool loop, send the tool result back as a `.tool(...)` message
with the matching `toolCallId`:

```swift
// After executing the tool, continue the conversation
let followUp = LLMRequest(
    model: "gpt-4o",
    messages: [
        .user("What's the weather in Tokyo?"),
        .assistant(toolCalls: response.toolCalls!),
        .tool(content: "{\"temp\": 22, \"condition\": \"clear\"}", toolCallId: call.id)
    ],
    tools: request.tools
)

let final = try await provider.complete(followUp)
print(final.text)  // "The weather in Tokyo is clear, around 22°C."
```

> For a full agent loop with parallel tool dispatch, dedup, repair-retry, and
> planner support, see [SwiftAgentKit](https://github.com/ayman3000/SwiftAgentKit)
> which builds on LLMProviderKit.

---

## Design Principles

1. **One protocol.** `LLMProvider` is the only contract. Conform to it, and your provider works everywhere.
2. **Native APIs, not wrappers.** Each provider speaks its own wire format — Ollama `api/chat`, Anthropic Messages API, Gemini `generateContent`. No OpenAI-compat shims unless the endpoint is actually OpenAI-compatible.
3. **Per-provider SPM targets.** Import only what you need. Using Ollama? Don't ship OpenAI code.
4. **Zero external dependencies.** Foundation + URLSession. No SwiftOpenAI, no SwiftAnthropic, no Alamofire, nothing.
5. **Async/await everywhere.** Native Swift concurrency. `AsyncThrowingStream` for streaming.
6. **Provider-agnostic models.** `LLMRequest`, `LLMResponse`, `LLMMessage` are the same regardless of provider. Your app code doesn't change when you swap.

---

## Adding a Provider

1. Add a new target in `Package.swift` (e.g. `LLMProviderKitAnthropic`).
2. Create `Sources/LLMProviderKitAnthropic/AnthropicProvider.swift`.
3. Conform to `LLMProvider`:

```swift
import Foundation
import LLMProviderKit

public struct AnthropicProvider: LLMProvider {
    public static let name = "anthropic"
    public let configuration: LLMProviderConfiguration

    public init(configuration: LLMProviderConfiguration) {
        self.configuration = configuration
    }

    public func prepareRequest(_ request: LLMRequest, stream: Bool) throws -> URLRequest {
        // Build URLRequest for messages API
    }

    public func parseStreamLine(_ line: String, request: LLMRequest) throws -> [LLMStreamChunk] {
        // Convert one SSE line into chunks
    }

    public func parseResponse(_ data: Data, request: LLMRequest) throws -> LLMResponse {
        // Convert full JSON into LLMResponse
    }
}
```

That's it. No changes needed in `LLMProviderKit` core.

---

## Alpha Status

LLMProviderKit is early but usable. APIs may still evolve before beta.

Tested with Ollama and Gemini, with expanding coverage for OpenAI and Anthropic.

If you try it in a real Swift app, feedback is very welcome.

---

## Testing

```bash
swift build
swift test
```

Includes 22 unit tests for parsing, streaming logic, model registry, tool calling, and image encoding for all four providers — no network calls.

---

## Requirements

- Swift 5.9+
- macOS 13+ / iOS 16+ / tvOS 16+ / watchOS 9+ / visionOS 1+

---

## Support

If LLMProviderKit helps you:

- ⭐ **Star the repository**
- 🐛 **[Open an issue](https://github.com/ayman3000/LLMProviderKit/issues)** — bugs, feature requests, provider requests
- ☕ **Support development on [Ko-fi](https://ko-fi.com/W7W61DDVO5)**

**Follow the author:**
- LinkedIn: [Ayman Hamed](https://www.linkedin.com/in/ayman-hamed-moustafa/)
- Explore macOS AI products: [kommanda.app](https://www.kommanda.app)

**Related:**
- [SwiftAgentKit](https://github.com/ayman3000/SwiftAgentKit) — AI agent framework built on LLMProviderKit

---

## License

MIT