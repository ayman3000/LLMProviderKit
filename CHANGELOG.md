# Changelog

All notable changes to LLMProviderKit will be documented in this file.

## Unreleased

### Fixed
- Decode streaming response lines from complete UTF-8 byte buffers instead of byte-by-byte Unicode scalars, preserving multi-byte characters such as Arabic, CJK, accents, and emoji.

### Added
- Regression coverage for non-ASCII streaming text.
- MIT license file.

## 0.1.0-alpha.4 - 2026-07-06

### Fixed
- Made provider model-capability heuristics more conservative for OpenAI and Gemini model IDs.
- Improved Gemini model ID normalization and tool metadata handling.

### Added
- Curated model metadata, model registry helpers, and provider capability tests.
