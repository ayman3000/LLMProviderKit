# Xcode Integration

Detailed setup for using LLMProviderKit in Xcode projects, including SwiftUI examples, provider picker, model picker, and App Sandbox configuration.

## Add the package

1. Open your project in Xcode.
2. Select the project file in the navigator, then the **Package Dependencies** tab.
3. Tap **+** and enter the package URL:
   - `https://github.com/ayman3000/LLMProviderKit`
4. Select the version rule or branch.
5. Add the products you need to your app target:
   - `LLMProviderKit` (always required)
   - `LLMProviderKitOllama`
   - `LLMProviderKitOpenAI`
   - `LLMProviderKitGemini`
   - `LLMProviderKitAnthropic`

> Only import the provider products your app actually uses to keep binary size small.

## Link products to your target

After adding the package:

1. Select your app target.
2. Go to **Build Phases ▸ Link Binary With Libraries**.
3. Add the `LLMProviderKit` and provider libraries you imported.

## Practical SwiftUI example

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

## Switching providers from a picker

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

## Using the model picker in SwiftUI

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

## App Sandbox & network entitlements

macOS apps with **App Sandbox** enabled (the default for new Xcode projects) will **block all outgoing network connections**, including `localhost`. This means `OllamaProvider.availableModels()` and any `complete()` / `stream()` call will fail with:

```
A server with the specified hostname could not be found.
```

To fix this, add the following entitlements to your app's `.entitlements` file:

```xml
<key>com.apple.security.network.client</key>
<true/>
```

If your app also needs to serve incoming connections (e.g. for callbacks), add:

```xml
<key>com.apple.security.network.server</key>
<true/>
```

For **development only**, you can disable App Sandbox entirely:

```xml
<key>com.apple.security.app-sandbox</key>
<false/>
```

> ⚠️ Disabling App Sandbox is fine for development but should not be shipped to production. For App Store distribution, keep the sandbox enabled and only add the `network.client` entitlement.

To add the entitlements file in Xcode:

1. Select your app target.
2. Go to **Signing & Capabilities**.
3. Click **+ Capability** and add **App Sandbox** (if not already present).
4. Check **Outgoing Connections (Client)**.
5. If you need incoming connections, also check **Incoming Connections (Server)**.