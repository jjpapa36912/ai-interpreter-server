import Foundation

enum CosyVoiceStreamingError: LocalizedError {
    case invalidEndpoint
    case invalidResponse(Int)
    case emptyAudio
    case invalidMessage
    case server(String)

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            "GPU 음성 서버 주소가 올바르지 않습니다."
        case let .invalidResponse(status):
            "GPU 음성 서버가 HTTP \(status) 오류를 반환했습니다."
        case .emptyAudio:
            "GPU 음성 서버가 음성을 반환하지 않았습니다."
        case .invalidMessage:
            "GPU 음성 서버 응답 형식이 올바르지 않습니다."
        case let .server(message):
            "GPU 음성 서버 오류: \(message)"
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

    private struct StreamMessage: Decodable {
        let type: String
        let sampleRate: Double?
        let error: String?

        enum CodingKeys: String, CodingKey {
            case type, error
            case sampleRate = "sample_rate"
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
            endpoint: configuration.voiceEndpoint,
            authorization: configuration.voiceAuthorization,
            session: session
        )
    }

    func synthesize(
        text: String,
        language: Language,
        voiceID: String? = nil,
        onChunk: @Sendable (AudioChunk) -> Void
    ) async throws {
        try await synthesizeWebSocket(
            text: text, language: language, voiceID: voiceID, onChunk: onChunk
        )
    }

    func webSocketURL() throws -> URL {
        var components = URLComponents(
            url: endpoint.appending(path: "v1/tts/ws"),
            resolvingAgainstBaseURL: false
        )
        components?.scheme = endpoint.scheme == "https" ? "wss" : "ws"
        var queryItems = [URLQueryItem(name: "connection_id", value: UUID().uuidString)]
        if let authorization, !authorization.isEmpty {
            queryItems.append(URLQueryItem(name: "token", value: authorization))
        }
        components?.queryItems = queryItems
        guard let url = components?.url else {
            throw CosyVoiceStreamingError.invalidEndpoint
        }
        return url
    }

    private func synthesizeWebSocket(
        text: String,
        language: Language,
        voiceID: String?,
        onChunk: @Sendable (AudioChunk) -> Void
    ) async throws {
        let socketSession = URLSession(configuration: .ephemeral)
        let socket = socketSession.webSocketTask(with: try webSocketURL())
        socket.resume()
        defer {
            socket.cancel(with: .normalClosure, reason: nil)
            socketSession.invalidateAndCancel()
        }
        let body = try JSONEncoder().encode(RequestBody(
            text: text, language: language.rawValue, voiceID: voiceID
        ))
        guard let request = String(data: body, encoding: .utf8) else {
            throw CosyVoiceStreamingError.invalidMessage
        }
        try await socket.send(.string(request))

        let playbackGroupID = UUID()
        var sampleRate = Self.sampleRate
        var emitted = false
        while true {
            let message = try await socket.receive()
            switch message {
            case let .data(data):
                guard data.count.isMultiple(of: MemoryLayout<Int16>.size) else {
                    throw CosyVoiceStreamingError.invalidMessage
                }
                onChunk(AudioChunk(
                    samples: Self.decodePCM16(data),
                    format: AudioFormatInfo(
                        sampleRate: sampleRate, channelCount: 1, bitsPerChannel: 16
                    ),
                    capturedAt: .now,
                    playbackGroupID: playbackGroupID
                ))
                emitted = true
            case let .string(value):
                let decoded = try JSONDecoder().decode(
                    StreamMessage.self, from: Data(value.utf8)
                )
                if decoded.type == "tts_start" {
                    sampleRate = decoded.sampleRate ?? Self.sampleRate
                } else if decoded.type == "tts_end" {
                    guard emitted else { throw CosyVoiceStreamingError.emptyAudio }
                    return
                } else if decoded.type == "tts_error" {
                    throw CosyVoiceStreamingError.server(decoded.error ?? "unknown")
                }
            @unknown default:
                throw CosyVoiceStreamingError.invalidMessage
            }
        }
    }

    private func synthesizeHTTP(
        text: String,
        language: Language,
        voiceID: String? = nil,
        onChunk: @Sendable (AudioChunk) -> Void
    ) async throws {
        let body = try JSONEncoder().encode(RequestBody(
            text: text, language: language.rawValue, voiceID: voiceID
        ))
        var bytes: URLSession.AsyncBytes?
        var successfulRetrySession: URLSession?
        var lastStatus = -1
        let retryDelays = [50, 75, 100, 150, 225, 350, 500]
        for attempt in 0...retryDelays.count {
            var request = URLRequest(url: endpoint.appending(path: "v1/tts/stream"))
            request.httpMethod = "POST"
            request.timeoutInterval = 30
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("audio/L16;rate=24000;channels=1", forHTTPHeaderField: "Accept")
            if let authorization {
                request.setValue("Bearer \(authorization)", forHTTPHeaderField: "Authorization")
            }
            request.httpBody = body

            // A 503 from G-Cube can poison the keep-alive connection. Reusing
            // that connection made every retry hit the same dead upstream.
            let requestSession = URLSession(configuration: .ephemeral)
            let result = try await requestSession.bytes(for: request)
            guard let http = result.1 as? HTTPURLResponse else {
                requestSession.invalidateAndCancel()
                throw CosyVoiceStreamingError.invalidResponse(-1)
            }
            lastStatus = http.statusCode
            if (200..<300).contains(http.statusCode) {
                bytes = result.0
                successfulRetrySession = requestSession
                break
            }
            requestSession.invalidateAndCancel()
            guard [502, 503, 504].contains(http.statusCode), attempt < retryDelays.count else {
                throw CosyVoiceStreamingError.invalidResponse(http.statusCode)
            }
            try await Task.sleep(for: .milliseconds(retryDelays[attempt]))
        }
        guard let bytes else { throw CosyVoiceStreamingError.invalidResponse(lastStatus) }
        defer { successfulRetrySession?.finishTasksAndInvalidate() }

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
