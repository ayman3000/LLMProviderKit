# Changelog

All notable changes to LLMProviderKit will be documented in this file.

## Unreleased

### Fixed
- Decode streaming response lines from complete UTF-8 byte buffers instead of byte-by-byte Unicode scalars, preserving multi-byte characters such as Arabic, CJK, accents, and emoji.
- Send a valid `anthropic-version` header (`2023-06-01`) instead of the branding string, which Anthropic rejected with HTTP 400; moved branding to `User-Agent`.
- Send Gemini tool results (`functionResponse` parts) in a `user` turn instead of `model`; Gemini only accepts `user`/`model` roles and mishandled tool results sent as `model`.

### Added
- **In-process provider support.** `complete(_:)` and `stream(_:)` are now `LLMProvider` protocol requirements (with the existing HTTP implementations as defaults), so an on-device provider — e.g. an MLX or llama.cpp backend — can override them directly and be dispatched correctly through `any LLMProvider`, instead of the extension methods being statically bound to the HTTP path. The HTTP-shaped hooks (`prepareRequest`, `parseStreamLine`, `parseResponse`) gain default (throwing/empty) implementations so an in-process provider only implements `complete`/`stream` + `name`/`configuration`. Existing HTTP providers are unchanged. Regression coverage: an in-process provider overrides `complete`/`stream` and wins dispatch through the existential.
- Latest curated models: OpenAI GPT-5.6 (`gpt-5.6`), Gemini 3.6 Flash (`gemini-3.6-flash`) and Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`), and Anthropic Opus 4.7 / Opus 4.6. Gemini's default preset model is now Gemini 3.6 Flash.
- Regression coverage for non-ASCII streaming text.
- Regression coverage for the Anthropic version header and Gemini tool-result role.
- MIT license file.

## 0.1.0-alpha.4 - 2026-07-06

### Fixed
- Made provider model-capability heuristics more conservative for OpenAI and Gemini model IDs.
- Improved Gemini model ID normalization and tool metadata handling.

### Added
- Curated model metadata, model registry helpers, and provider capability tests.
