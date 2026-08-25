import Foundation

enum CosyVoiceStreamingError: LocalizedError {
    case invalidEndpoint
    case invalidResponse(Int)
    case emptyAudio

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            "CosyVoice 스트리밍 서버 주소가 올바르지 않습니다."
        case let .invalidResponse(status):
            "CosyVoice 서버가 HTTP \(status) 오류를 반환했습니다."
        case .emptyAudio:
            "CosyVoice 서버가 음성을 반환하지 않았습니다."
        }
    }
}

/// Client for the product TTS boundary. The server response is raw signed
/// little-endian PCM16, 24 kHz, mono. Fixed 100 ms chunks keep playback latency
/// bounded and avoid buffering an entire utterance before speaking.
final class CosyVoiceStreamingClient: @unchecked Sendable {
    private struct RequestBody: Encodable {
        let text: String
        let language: String
        let voiceID: String?

        enum CodingKeys: String, CodingKey {
            case text, language
            case voiceID = "voice_id"
        }
    }

    static let sampleRate = 24_000.0
    static let framesPerChunk = 2_400

    let endpoint: URL
    private let authorization: String?
    private let session: URLSession

    init(endpoint: URL, authorization: String? = nil, session: URLSession = .shared) {
        self.endpoint = endpoint
        self.authorization = authorization
        self.session = session
    }

    static func configured(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> CosyVoiceStreamingClient? {
        guard let raw = environment["AI_INTERPRETER_TTS_SERVER_URL"],
              let url = URL(string: raw),
              ["http", "https"].contains(url.scheme?.lowercased() ?? "") else { return nil }
        return CosyVoiceStreamingClient(
            endpoint: url,
            authorization: environment["AI_INTERPRETER_TTS_SERVER_TOKEN"]
        )
    }

    static func configured(
        from configuration: CUDAStreamingServerConfiguration,
        session: URLSession = .shared
    ) -> CosyVoiceStreamingClient {
        CosyVoiceStreamingClient(
            endpoint: configuration.endpoint,
            authorization: configuration.authorization,
            session: session
        )
    }

    func synthesize(
        text: String,
        language: Language,
        voiceID: String? = nil,
        onChunk: @Sendable (AudioChunk) -> Void
    ) async throws {
        var request = URLRequest(url: endpoint.appending(path: "v1/tts/stream"))
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("audio/L16;rate=24000;channels=1", forHTTPHeaderField: "Accept")
        if let authorization {
            request.setValue("Bearer \(authorization)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONEncoder().encode(RequestBody(
            text: text, language: language.rawValue, voiceID: voiceID
        ))

        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw CosyVoiceStreamingError.invalidResponse(-1)
        }
        guard (200..<300).contains(http.statusCode) else {
            throw CosyVoiceStreamingError.invalidResponse(http.statusCode)
        }

        let targetBytes = Self.framesPerChunk * MemoryLayout<Int16>.size
        var pending = Data()
        var emitted = false
        for try await byte in bytes {
            pending.append(byte)
            if pending.count >= targetBytes {
                let chunkData = pending.prefix(targetBytes)
                pending.removeFirst(targetBytes)
                onChunk(Self.makeChunk(Data(chunkData)))
                emitted = true
            }
        }
        if !pending.isEmpty {
            if pending.count.isMultiple(of: MemoryLayout<Int16>.size) {
                onChunk(Self.makeChunk(pending))
                emitted = true
            }
        }
        guard emitted else { throw CosyVoiceStreamingError.emptyAudio }
    }

    static func decodePCM16(_ data: Data) -> [Float] {
        data.withUnsafeBytes { rawBuffer in
            let samples = rawBuffer.bindMemory(to: Int16.self)
            return samples.map { Float(Int16(littleEndian: $0)) / Float(Int16.max) }
        }
    }

    private static func makeChunk(_ data: Data) -> AudioChunk {
        AudioChunk(
            samples: decodePCM16(data),
            format: AudioFormatInfo(sampleRate: sampleRate, channelCount: 1, bitsPerChannel: 16),
            capturedAt: .now
        )
    }
}
