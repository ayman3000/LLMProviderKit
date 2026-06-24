// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "LLMProviderKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .watchOS(.v9),
        .visionOS(.v1)
    ],
    products: [
        // Core models and protocols — always required.
        .library(name: "LLMProviderKit", targets: ["LLMProviderKit"]),
        // Individual providers — import only the ones you use.
        .library(name: "LLMProviderKitOllama", targets: ["LLMProviderKitOllama"]),
        .library(name: "LLMProviderKitOpenAI", targets: ["LLMProviderKitOpenAI"]),
        .library(name: "LLMProviderKitGemini", targets: ["LLMProviderKitGemini"]),
        .library(name: "LLMProviderKitAnthropic", targets: ["LLMProviderKitAnthropic"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "LLMProviderKit",
            dependencies: [],
            path: "Sources/LLMProviderKit"
        ),
        .target(
            name: "LLMProviderKitOllama",
            dependencies: ["LLMProviderKit"],
            path: "Sources/LLMProviderKitOllama"
        ),
        .target(
            name: "LLMProviderKitOpenAI",
            dependencies: ["LLMProviderKit"],
            path: "Sources/LLMProviderKitOpenAI"
        ),
        .target(
            name: "LLMProviderKitGemini",
            dependencies: ["LLMProviderKit"],
            path: "Sources/LLMProviderKitGemini"
        ),
        .target(
            name: "LLMProviderKitAnthropic",
            dependencies: ["LLMProviderKit"],
            path: "Sources/LLMProviderKitAnthropic"
        ),
        .testTarget(
            name: "LLMProviderKitTests",
            dependencies: [
                "LLMProviderKit",
                "LLMProviderKitOllama",
                "LLMProviderKitOpenAI",
                "LLMProviderKitGemini",
                "LLMProviderKitAnthropic"
            ],
            path: "Tests/LLMProviderKitTests"
        ),
    ]
)
