import Foundation

/// Joins tool calls that a stream delivers in fragments.
///
/// OpenAI-style streams (OpenAI, OpenRouter, Ollama Cloud, most compatible
/// servers) send one call as pieces: the first carries the id, the name and the
/// start of the arguments; later ones only carry more argument text. Every
/// piece names the call it belongs to by `index`, which a provider passes along
/// in `providerMetadata[indexKey]`.
///
/// Passed on raw, a nameless fragment reads as "tool use with no usable call",
/// and an agent then asks the model the whole question again. The assembler
/// holds fragments and releases whole calls, in index order, just before the
/// stream's `.finish`. Calls without an index are already whole and pass
/// straight through, as does every other chunk.
public struct StreamToolCallAssembler: Sendable {
    /// The `providerMetadata` key a provider sets on a fragment.
    public static let indexKey = "streamToolCallIndex"

    private struct Pending: Sendable {
        var id: String
        var name: String
        var arguments: String
        var metadata: [String: String]
    }

    private var pending: [Int: Pending] = [:]

    public init() {}

    /// Feed one chunk; get back the chunks to pass on now.
    public mutating func consume(_ chunk: LLMStreamChunk) -> [LLMStreamChunk] {
        switch chunk {
        case .toolCall(let call):
            guard let raw = call.providerMetadata[Self.indexKey], let index = Int(raw) else {
                return [chunk]
            }
            if var existing = pending[index] {
                if existing.name.isEmpty { existing.name = call.name }
                existing.arguments += call.arguments
                pending[index] = existing
            } else {
                var metadata = call.providerMetadata
                metadata[Self.indexKey] = nil
                pending[index] = Pending(id: call.id, name: call.name, arguments: call.arguments, metadata: metadata)
            }
            return []
        case .finish:
            return flush() + [chunk]
        default:
            return [chunk]
        }
    }

    /// Release every held call. Call at the end of a stream that never sent `.finish`.
    public mutating func flush() -> [LLMStreamChunk] {
        let calls = pending.keys.sorted().compactMap { pending[$0] }.map { p in
            LLMStreamChunk.toolCall(LLMToolCall(
                id: p.id,
                name: p.name,
                arguments: p.arguments.isEmpty ? "{}" : p.arguments,
                providerMetadata: p.metadata))
        }
        pending.removeAll()
        return calls
    }
}
