# LLMProviderKit

A small, protocol-oriented Swift package that unifies chat completions and streaming across multiple LLM providers.

Currently supported providers:

- **Ollama** (`LLMProviderKitOllama`) — local-first, auto-discovers available models and picks the first one if none is configured. Custom host URL supported.
- **OpenAI** (`LLMProviderKitOpenAI`)
- **Google Gemini** (`LLMProviderKitGemini`)
- **Anthropic** (`LLMProviderKitAnthropic`)

Designed to make adding **xAI**, **DeepSeek**, **OpenRouter**, or any other provider a matter of conforming to one protocol.

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
        .package(path: "path/to/LLMProviderKit")
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

Or add it in Xcode via **File ▸ Add Package Dependencies**.

---

## Xcode Integration

The easiest way to use `LLMProviderKit` in an Xcode project is with **Swift Package Manager**.

### Add the package

1. Open your project in Xcode.
2. Select the project file in the navigator, then the **Package Dependencies** tab.
3. Tap **+** and enter the package URL or local path:
   - Local: `file:///Users/aymanmoustafa/projects/my_repos/packages/swift/LLMProviderKit`
   - Remote (when published): `https://github.com/yourusername/LLMProviderKit`
4. Select the version rule or branch.
5. Add the products you need to your app target:
   - `LLMProviderKit` (always required)
   - `LLMProviderKitOllama`
   - `LLMProviderKitOpenAI`
   - `LLMProviderKitGemini`
   - `LLMProviderKitAnthropic`

> Only import the provider products your app actually uses to keep binary size small.

### Link products to your target

After adding the package:

1. Select your app target.
2. Go to **Build Phases ▸ Link Binary With Libraries**.
3. Add the `LLMProviderKit` and provider libraries you imported.

### Practical SwiftUI example

```swift
import SwiftUI
import LLMProviderKit
import LLMProviderKitOllama
import LLMProviderKitOpenAI

@main
struct MyAIApp: App {
    var body: some Scene {
        WindowGroup {
            ChatView()
        }
    }
}

@Observable
final class ChatModel {
    var messages: [String] = []
    var input: String = ""
    var isThinking = false

    private let ollama = OllamaProvider(configuration: .local())
    private let openAI = OpenAIProvider(configuration: .openAI(apiKey: "sk-..."))

    func send() async {
        let text = input
        input = ""
        messages.append("You: \(text)")
        isThinking = true

        let request = LLMRequest(
            model: "",
            messages: [
                .system("You are a helpful assistant."),
                .user(text)
            ]
        )

        do {
            let response = try await ollama.complete(request)
            messages.append("Ollama: \(response.text)")
        } catch {
            messages.append("Error: \(error.localizedDescription)")
        }

        isThinking = false
    }
}

struct ChatView: View {
    @State private var model = ChatModel()

    var body: some View {
        VStack {
            List(model.messages, id: \.self) { message in
                Text(message)
            }

            HStack {
                TextField("Message", text: $model.input)
                    .textFieldStyle(.roundedBorder)
                Button("Send") {
                    Task { await model.send() }
                }
                .disabled(model.isThinking)
            }
            .padding()
        }
    }
}
```

### Switching providers from a picker

```swift
import SwiftUI
import LLMProviderKit
import LLMProviderKitOllama
import LLMProviderKitOpenAI
import LLMProviderKitGemini
import LLMProviderKitAnthropic

@Observable
final class AIViewModel {
    var selectedProvider = "ollama"

    private let providers: [String: any LLMProvider] = [
        "ollama": OllamaProvider(configuration: .local()),
        "openai": OpenAIProvider(configuration: .openAI(apiKey: "sk-...")),
        "gemini": GeminiProvider(configuration: .gemini(apiKey: "...")),
        "anthropic": AnthropicProvider(configuration: .anthropic(apiKey: "..."))
    ]

    func complete(_ request: LLMRequest) async throws -> LLMResponse {
        guard let provider = providers[selectedProvider] else {
            throw LLMError.unknownProvider(selectedProvider)
        }
        return try await provider.complete(request)
    }
}
```

### Using the model picker in SwiftUI

```swift
@Observable
final class ModelPickerModel {
    let provider = OllamaProvider(configuration: .local())
    let registry = LLMModelRegistry()
    var models: [LLMModelInfo] = []
    var selectedModel: String = ""

    func load() async {
        try? await registry.refresh(from: provider)
        models = await registry.models(for: "ollama")
        selectedModel = models.first?.id ?? ""
    }
}

struct ModelPickerView: View {
    @State private var model = ModelPickerModel()

    var body: some View {
        Picker("Model", selection: $model.selectedModel) {
            ForEach(model.models) { m in
                Text(m.displayName ?? m.id).tag(m.id)
            }
        }
        .task { await model.load() }
    }
}
```

> For `LLMModelInfo` to work in `ForEach`, it already conforms to `Identifiable`.

---

## Usage

### Ollama specifics

```swift
import LLMProviderKitOllama

// Default localhost
let ollama = OllamaProvider(configuration: .local())

// Custom host (e.g. LAN server or Docker)
let remote = OllamaProvider(
    configuration: .local(baseURL: URL(string: "http://192.168.1.50:11434")!)
)

// Auto-resolve: leave request model empty, provider fetches /api/tags and uses first model.
let request = LLMRequest(model: "", messages: [.user("Hi")])
let response = try await ollama.complete(request)
print("Resolved model: \(response.request.model)")
```

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

---

## How LLMProviderKit compares

There are other multi-provider LLM packages for Swift. Here is how this one differs:

| Feature | LLMProviderKit | [PolyAI](https://github.com/jamesrochabrun/PolyAI) | [LLMChatOpenAI](https://github.com/kevinhermawan/swift-llm-chat-openai) | [SwiftAI](https://github.com/mi12labs/SwiftAI) |
|---|---|---|---|---|
| Zero external dependencies | ✅ Foundation only | ❌ SwiftOpenAI + SwiftAnthropic | ❌ | ❌ |
| Per-provider SPM targets | ✅ | ❌ | ❌ | ❌ |
| Native Ollama `/api/chat` | ✅ | ❌ OpenAI-compat only | ❌ OpenAI-compat only | ? |
| Native Anthropic Messages API | ✅ | ✅ via SwiftAnthropic | ❌ | ? |
| Native Gemini API | ✅ | ✅ | ❌ | ? |
| Offline model registry + curated lists | ✅ | ❌ | ❌ | ❌ |
| Auto-resolve first Ollama model | ✅ | ❌ | ❌ | ❌ |
| macOS 13 / iOS 16 support | ✅ | iOS 15+ | ? | iOS 17+ |
| Tools / structured outputs | ❌ planned | ✅ | ? | ✅ |

LLMProviderKit is intentionally lightweight: a pure networking abstraction with one protocol and native request/response mapping per provider. If you need tool calling or structured outputs today, PolyAI or SwiftAI are further ahead. If you want a dependency-free, portable core that stays current with new provider APIs without waiting for upstream wrappers, LLMProviderKit is the simpler fit.

---

## Cross-platform naming

If you plan to publish the same abstraction for Flutter and Rust, you have two strategies:

### Option A: same brand name everywhere

| Platform | Package name | Example |
|---|---|---|
| Swift | `llmkit` or `LLMProviderKit` | `LLMProviderKit` |
| Flutter / Dart | `llmkit` | `llmkit: ^1.0.0` |
| Rust | `llmkit` | `llmkit = "1.0"` |

**Pros**
- One name across all platforms and docs.
- Easier marketing: "LLMProviderKit for Swift", "LLMProviderKit for Dart", "LLMProviderKit for Rust".
- Users can migrate between platforms without learning new vocabulary.

**Cons**
- Naming conflicts become harder. `llmkit` on crates.io / pub.dev / Swift Package Index may already be taken.
- You need to secure the name on all three ecosystems now.

### Option B: platform-specific names

| Platform | Package name | Example |
|---|---|---|
| Swift | `LLMProviderKit` |
| Flutter / Dart | `llmkit_dart` or `llmkit_flutter` |
| Rust | `llmkit_rs` or `llmkit_rust` |

**Pros**
- Easier to get a unique package id on each registry.
- Follows common conventions (`package_foo` on pub.dev, `foo-rs` on crates.io).

**Cons**
- Weakens the cross-platform brand.
- Docs and examples have to explain that these are the same project.

### My recommendation

Use the **same brand name** if you can secure it:

1. Check availability **today** on:
   - [pub.dev](https://pub.dev) for Flutter/Dart
   - [crates.io](https://crates.io) for Rust
   - [GitHub](https://github.com) org/repo name
   - Swift Package Index (by publishing or reserving the GitHub repo)

2. If `llmkit` is taken on any one registry, switch the **global brand** to a name that is free everywhere. Do not ship with a different name per platform — it fragments your ecosystem.

Good global brand candidates to check:

- `omnillm`
- `llmbridge`
- `polyllm`
- `modelforge`
- `aillm`

> Because `LLMProviderKit` already exists on GitHub (`spprichard/LLMProviderKit`), I would treat the current repo as a working title and rebrand before the Flutter/Rust packages ship.

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

That’s it. No changes needed in `LLMProviderKit` core.

---

## Testing

```bash
cd LLMProviderKit
swift test
```

Tests cover parsing and streaming logic for all three providers without making real network calls.

---

## Requirements

- Swift 5.9+
- macOS 13+ / iOS 16+ / tvOS 16+ / watchOS 9+ / visionOS 1+

---

## License

MIT
