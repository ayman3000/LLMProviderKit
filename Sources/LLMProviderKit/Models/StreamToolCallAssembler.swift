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
/// holds fragments and releases whole calls, in arrival order, just before the
/// stream's `.finish`. Calls without an index are already whole and pass
/// straight through, as does every other chunk.
///
/// Some Ollama-compatible endpoints number every call in a parallel batch 0 and
/// tell them apart only by id, so a NEW id at an index already in use opens a
/// new call. A fragment with no id (the norm after the first piece) extends the
/// newest call at its index.
public struct StreamToolCallAssembler: Sendable {
    /// The `providerMetadata` key a provider sets on a fragment.
    public static let indexKey = "streamToolCallIndex"

    private struct Pending: Sendable {
        var id: String
        var name: String
        var arguments: String
        var metadata: [String: String]
    }

    /// Calls in arrival order; `slotForIndex` points each wire index at its newest call.
    private var slots: [Pending] = []
    private var slotForIndex: [Int: Int] = [:]

    public init() {}

    /// Feed one chunk; get back the chunks to pass on now.
    public mutating func consume(_ chunk: LLMStreamChunk) -> [LLMStreamChunk] {
        switch chunk {
        case .toolCall(let call):
            guard let raw = call.providerMetadata[Self.indexKey], let index = Int(raw) else {
                return [chunk]
            }
            if let slot = slotForIndex[index], call.id.isEmpty || call.id == slots[slot].id || slots[slot].id.isEmpty {
                if slots[slot].id.isEmpty { slots[slot].id = call.id }
                if slots[slot].name.isEmpty { slots[slot].name = call.name }
                slots[slot].arguments += call.arguments
            } else {
                var metadata = call.providerMetadata
                metadata[Self.indexKey] = nil
                slots.append(Pending(id: call.id, name: call.name, arguments: call.arguments, metadata: metadata))
                slotForIndex[index] = slots.count - 1
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
        let calls = slots.map { p in
            LLMStreamChunk.toolCall(LLMToolCall(
                id: p.id.isEmpty ? UUID().uuidString : p.id,
                name: p.name,
                arguments: p.arguments.isEmpty ? "{}" : p.arguments,
                providerMetadata: p.metadata))
        }
        slots.removeAll()
        slotForIndex.removeAll()
        return calls
    }
}
