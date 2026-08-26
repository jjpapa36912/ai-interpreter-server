import Foundation

enum CUDAStreamingProviderError: LocalizedError {
    case notConfigured
    case invalidEndpoint
    case serverNotReady(String)
    case invalidMessage

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            "GPU 동시통역 서버 주소가 설정되지 않았습니다."
        case .invalidEndpoint:
            "GPU 동시통역 서버 주소가 올바르지 않습니다."
        case let .serverNotReady(detail):
            "GPU 동시통역 서버가 준비되지 않았습니다: \(detail)"
        case .invalidMessage:
            "GPU 음성 인식 서버 응답이 올바르지 않습니다."
        }
    }
}

struct CUDAStreamingServerConfiguration: Sendable, Equatable {
    let endpoint: URL
    let authorization: String?
    let voiceEndpoint: URL
    let voiceAuthorization: String?

    static func configured(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> CUDAStreamingServerConfiguration? {
        guard environment["AI_INTERPRETER_DISABLE_GPU_STREAMING"] != "1",
              let raw = environment["AI_INTERPRETER_STREAMING_SERVER_URL"],
              let endpoint = URL(string: raw),
              ["http", "https"].contains(endpoint.scheme?.lowercased() ?? "")
        else { return nil }
        let voiceEndpoint: URL
        if let rawVoice = environment["AI_INTERPRETER_TTS_SERVER_URL"],
           let configuredVoice = URL(string: rawVoice),
           ["http", "https"].contains(configuredVoice.scheme?.lowercased() ?? "") {
            voiceEndpoint = configuredVoice
        } else {
            voiceEndpoint = endpoint
        }
        return CUDAStreamingServerConfiguration(
            endpoint: endpoint,
            authorization: environment["AI_INTERPRETER_STREAMING_SERVER_TOKEN"],
            voiceEndpoint: voiceEndpoint,
            voiceAuthorization: environment["AI_INTERPRETER_TTS_SERVER_TOKEN"]
                ?? environment["AI_INTERPRETER_STREAMING_SERVER_TOKEN"]
        )
    }

    var requiresVoicePortHandoff: Bool {
        endpoint.absoluteURL.standardized == voiceEndpoint.absoluteURL.standardized
    }

    func webSocketURL(sourceLanguage: Language) throws -> URL {
        var components = URLComponents(
            url: endpoint.appending(path: "v1/asr/stream"),
            resolvingAgainstBaseURL: false
        )
        components?.scheme = endpoint.scheme == "https" ? "wss" : "ws"
        var queryItems = [
            URLQueryItem(name: "language", value: sourceLanguage.rawValue),
            URLQueryItem(name: "sample_rate", value: "16000"),
        ]
        // Some managed WebSocket proxies don't forward an Authorization header
        // from URLSession's upgrade request. The server accepts the same bearer
        // credential as an encrypted WSS query parameter for this handshake.
        if let authorization, !authorization.isEmpty {
            queryItems.append(URLQueryItem(name: "token", value: authorization))
        }
        components?.queryItems = queryItems
        guard let url = components?.url else {
            throw CUDAStreamingProviderError.invalidEndpoint
        }
        return url
    }
}

/// Cross-platform service boundary for live ASR and translation. The only
/// ASR, translation and neural speech synthesis all run behind the same GPU
/// service boundary, so this path has no Apple speech/translation dependency.
actor CUDAStreamingTranslationProvider: TranslationProvider {
    private struct QueuedSource: Sendable {
        let text: String
        let translation: String?
        let translationMilliseconds: Double?
        let enqueuedAt: ContinuousClock.Instant
    }

    private struct ReadyResponse: Decodable {
        let status: String
        let preloadError: String?

        enum CodingKeys: String, CodingKey {
            case status
            case preloadError = "preload_error"
        }
    }

    private struct ASRMessage: Decodable {
        let type: String
        let hypothesis: String?
        let committedDelta: String?
        let latencyMilliseconds: Double?
        let translation: String?
        let translationMilliseconds: Double?

        enum CodingKeys: String, CodingKey {
            case type, hypothesis
            case committedDelta = "committed_delta"
            case latencyMilliseconds = "latency_ms"
            case translation
            case translationMilliseconds = "translation_latency_ms"
        }
    }

    nonisolated let translatedAudio: AsyncStream<AudioChunk>

    private let configuration: CUDAStreamingServerConfiguration
    private let session: URLSession
    private let translationClient: CUDATranslationClient
    private let neuralTTS: CosyVoiceStreamingClient
    private let outputContinuation: AsyncStream<AudioChunk>.Continuation
    private var socketSession: URLSession?
    private var isReleasingSocketForVoice = false
    private var sourceLanguage: Language?
    private var targetLanguage: Language?
    private var socket: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var translationTask: Task<Void, Never>?
    private var translationContinuation: AsyncStream<QueuedSource>.Continuation?
    private var pipeline = AudioChunkPipeline()
    private var silenceGate = StreamingSilenceGate()
    private var pipelineTask: Task<Void, Never>?
    private var metrics = AudioPipelineMetrics()
    private var sessionStartedAt: ContinuousClock.Instant?
    private var pendingError: Error?

    init(
        configuration: CUDAStreamingServerConfiguration,
        session: URLSession = .shared
    ) {
        self.configuration = configuration
        self.session = session
        translationClient = CUDATranslationClient(
            endpoint: configuration.endpoint,
            authorization: configuration.authorization
        )
        neuralTTS = CosyVoiceStreamingClient.configured(
            from: configuration,
            session: session
        )
        let output = AsyncStream<AudioChunk>.makeStream(
            bufferingPolicy: .bufferingNewest(AppConfiguration.maximumTranslatedAudioChunks)
        )
        translatedAudio = output.stream
        outputContinuation = output.continuation
    }

    static func configured(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> CUDAStreamingTranslationProvider? {
        guard let configuration = CUDAStreamingServerConfiguration.configured(
            environment: environment
        ) else { return nil }
        return CUDAStreamingTranslationProvider(configuration: configuration)
    }

    func startSession(sourceLanguage: Language, targetLanguage: Language) async throws {
        guard socket == nil else { return }
        // Connect directly. G-Cube's HTTP proxy can intermittently return 503
        // for /ready while the WebSocket route is healthy. The WebSocket endpoint
        // already closes with 1013/1011 when models aren't actually available.
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        sessionStartedAt = .now
        metrics = AudioPipelineMetrics()
        pendingError = nil
        // G-Cube exposes this workload through a single public connection slot.
        // While that slot is briefly handed from ASR WebSocket to HTTP TTS,
        // retain enough source audio to replay it after reconnecting. This is
        // input audio, not a TTS backlog, so preserving it prevents skipped
        // speech without extending translated playback latency.
        pipeline = AudioChunkPipeline(maximumBufferedChunks: 100)
        silenceGate.reset()

        let (connectedSession, webSocket) = try await connectSocket(
            sourceLanguage: sourceLanguage
        )
        socketSession = connectedSession
        socket = webSocket

        let translations = AsyncStream<QueuedSource>.makeStream()
        translationContinuation = translations.continuation
        translationTask = Task { [weak self] in
            for await source in translations.stream {
                guard !Task.isCancelled else { break }
                await self?.translateAndSynthesize(source)
            }
        }
        receiveTask = Task { [weak self] in
            await self?.receiveMessages()
        }
        let audio = pipeline.output
        pipelineTask = Task { [weak self] in
            do {
                for await chunk in audio {
                    guard !Task.isCancelled else { break }
                    try await self?.sendPCM(chunk.data)
                }
            } catch {
                await self?.recordFailure(error)
            }
        }
    }

    func sendAudio(_ chunk: AudioChunk) async throws {
        // During the deliberate ASR -> TTS port handoff there is briefly no
        // socket. Accept and buffer source audio instead of reporting a broken
        // session to AppState (which would stop capture and cancel TTS).
        guard socket != nil || isReleasingSocketForVoice else {
            throw TranslationProviderError.notConnected
        }
        if let pendingError {
            self.pendingError = nil
            throw pendingError
        }
        metrics.inputChunks += 1
        // Do not send startup/digital silence to Whisper. Decoding an all-zero
        // prefix can hallucinate short polite phrases (for example "감사해요")
        // which would otherwise be committed and spoken before media starts.
        guard silenceGate.shouldForward(chunk) else { return }
        await pipeline.ingest(chunk)
    }

    func currentPipelineMetrics() async -> AudioPipelineMetrics {
        var current = metrics
        let conversion = await pipeline.currentMetrics()
        current.droppedChunks += conversion.droppedChunks
        current.bufferedSamples = conversion.bufferedSamples
        return current
    }

    func stopSession() async {
        if let socket {
            try? await socket.send(.string("finish"))
            socket.cancel(with: .normalClosure, reason: nil)
        }
        socket = nil
        socketSession?.invalidateAndCancel()
        socketSession = nil
        isReleasingSocketForVoice = false
        receiveTask?.cancel()
        translationTask?.cancel()
        pipelineTask?.cancel()
        receiveTask = nil
        translationTask = nil
        pipelineTask = nil
        translationContinuation?.finish()
        translationContinuation = nil
        await pipeline.finish()
        sourceLanguage = nil
        targetLanguage = nil
        sessionStartedAt = nil
    }

    private func connectSocket(
        sourceLanguage: Language
    ) async throws -> (URLSession, URLSessionWebSocketTask) {
        var lastError: Error = URLError(.cannotConnectToHost)
        let delays = [50, 75, 100, 150, 225, 350, 500]
        for attempt in 0...delays.count {
            var components = URLComponents(
                url: try configuration.webSocketURL(sourceLanguage: sourceLanguage),
                resolvingAgainstBaseURL: false
            )
            var queryItems = components?.queryItems ?? []
            queryItems.append(URLQueryItem(
                name: "connection_attempt",
                value: "\(attempt)-\(UUID().uuidString)"
            ))
            components?.queryItems = queryItems
            guard let url = components?.url else {
                throw CUDAStreamingProviderError.invalidEndpoint
            }

            let candidateSession = URLSession(configuration: .ephemeral)
            let candidate = candidateSession.webSocketTask(with: URLRequest(url: url))
            candidate.resume()
            do {
                try await candidate.send(.string("reset"))
                let reply = try await withThrowingTaskGroup(
                    of: URLSessionWebSocketTask.Message.self
                ) { group in
                    group.addTask { try await candidate.receive() }
                    group.addTask {
                        try await Task.sleep(for: .seconds(4))
                        throw URLError(.timedOut)
                    }
                    let value = try await group.next()!
                    group.cancelAll()
                    return value
                }
                let data: Data
                switch reply {
                case let .data(value): data = value
                case let .string(value): data = Data(value.utf8)
                @unknown default: throw CUDAStreamingProviderError.invalidMessage
                }
                let ack = try JSONDecoder().decode(ASRMessage.self, from: data)
                guard ack.type == "reset" else {
                    throw CUDAStreamingProviderError.invalidMessage
                }
                return (candidateSession, candidate)
            } catch {
                lastError = error
                candidate.cancel(with: .goingAway, reason: nil)
                candidateSession.invalidateAndCancel()
            }
            if attempt < delays.count {
                try await Task.sleep(for: .milliseconds(delays[attempt]))
            }
        }
        throw lastError
    }

    private func requireReady() async throws {
        var lastDetail = "invalid response"
        for attempt in 0..<4 {
            do {
                var request = URLRequest(url: configuration.endpoint.appending(path: "ready"))
                request.timeoutInterval = 15
                if let authorization = configuration.authorization {
                    request.setValue("Bearer \(authorization)", forHTTPHeaderField: "Authorization")
                }
                let (data, response) = try await session.data(for: request)
                let status = (response as? HTTPURLResponse)?.statusCode
                if status.map({ (200..<300).contains($0) }) == true,
                   let ready = try? JSONDecoder().decode(ReadyResponse.self, from: data),
                   ready.status == "ready" {
                    return
                }
                lastDetail = "HTTP \(status ?? -1): " +
                    (String(data: data, encoding: .utf8) ?? "invalid response")
                guard status == 502 || status == 503 || status == 504 else { break }
            } catch {
                lastDetail = error.localizedDescription
                guard (error as? URLError)?.code == .badServerResponse ||
                        (error as? URLError)?.code == .networkConnectionLost else { break }
            }
            if attempt < 3 {
                try await Task.sleep(for: .milliseconds(250 * (attempt + 1)))
            }
        }
        throw CUDAStreamingProviderError.serverNotReady(lastDetail)
    }

    private func sendPCM(_ data: Data) async throws {
        // A translated phrase temporarily releases G-Cube's only public
        // connection so the TTS HTTP request can reach the same workload.
        // Wait here instead of failing the pipeline; AudioChunkPipeline keeps
        // the source PCM bounded while the socket is being restored.
        for _ in 0..<160 {
            if let socket {
                try await socket.send(.data(data))
                return
            }
            try await Task.sleep(for: .milliseconds(50))
        }
        throw TranslationProviderError.notConnected
    }

    private func releaseSocketForVoice() {
        isReleasingSocketForVoice = true
        receiveTask?.cancel()
        receiveTask = nil
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        socketSession?.invalidateAndCancel()
        socketSession = nil
    }

    private func restoreSocketAfterVoice() async throws {
        guard socket == nil, let sourceLanguage else { return }
        let (connectedSession, webSocket) = try await connectSocket(
            sourceLanguage: sourceLanguage
        )
        socketSession = connectedSession
        socket = webSocket
        isReleasingSocketForVoice = false
        receiveTask = Task { [weak self] in
            await self?.receiveMessages()
        }
    }

    private func receiveMessages() async {
        guard let socket else { return }
        do {
            while !Task.isCancelled {
                let message = try await socket.receive()
                let data: Data
                switch message {
                case let .data(value): data = value
                case let .string(value): data = Data(value.utf8)
                @unknown default: throw CUDAStreamingProviderError.invalidMessage
                }
                let decoded = try JSONDecoder().decode(ASRMessage.self, from: data)
                if let latency = decoded.latencyMilliseconds {
                    metrics.asrMilliseconds = latency
                }
                if let hypothesis = decoded.hypothesis, !hypothesis.isEmpty {
                    metrics.sourceHypothesis = hypothesis
                    if metrics.firstHypothesisMilliseconds == nil,
                       let sessionStartedAt {
                        metrics.firstHypothesisMilliseconds = sessionStartedAt.duration(to: .now).milliseconds
                    }
                }
                let delta = decoded.committedDelta?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                if !delta.isEmpty {
                    metrics.committedSource = [metrics.committedSource, delta]
                        .compactMap { $0 }.joined(separator: " ")
                    if metrics.firstStableCommitMilliseconds == nil,
                       let sessionStartedAt {
                        metrics.firstStableCommitMilliseconds = sessionStartedAt.duration(to: .now).milliseconds
                    }
                    translationContinuation?.yield(QueuedSource(
                        text: delta,
                        translation: decoded.translation,
                        translationMilliseconds: decoded.translationMilliseconds,
                        enqueuedAt: .now
                    ))
                }
                if decoded.type == "finished" { break }
            }
        } catch is CancellationError {
            return
        } catch {
            if isReleasingSocketForVoice { return }
            recordFailure(error)
        }
    }

    private func translateAndSynthesize(_ request: QueuedSource) async {
        guard let sourceLanguage, let targetLanguage else { return }
        guard !TTSQueuePolicy.isStale(
            .simultaneousCommitted,
            text: request.text,
            enqueuedAt: request.enqueuedAt
        ) else {
            metrics.droppedChunks += 1
            return
        }
        do {
            let response: (text: String, milliseconds: Double)
            if let translation = request.translation?.trimmingCharacters(
                in: .whitespacesAndNewlines
            ), !translation.isEmpty {
                response = (translation, request.translationMilliseconds ?? 0)
            } else {
                response = try await withThrowingTaskGroup(
                    of: (text: String, milliseconds: Double).self
                ) { group in
                    group.addTask { [translationClient] in
                        try await translationClient.translate(
                            text: request.text,
                            sourceLanguage: sourceLanguage,
                            targetLanguage: targetLanguage
                        )
                    }
                    group.addTask {
                        try await Task.sleep(for: TranslationQueuePolicy.remoteWatchdogTimeout)
                        throw URLError(.timedOut)
                    }
                    let result = try await group.next()!
                    group.cancelAll()
                    return result
                }
            }
            metrics.translationMilliseconds = response.milliseconds
            metrics.translationPhrase = response.text
            if metrics.firstStableTranslationMilliseconds == nil,
               let sessionStartedAt {
                metrics.firstStableTranslationMilliseconds = sessionStartedAt.duration(to: .now).milliseconds
            }
            guard !TTSQueuePolicy.isStale(
                .simultaneousCommitted,
                text: response.text,
                enqueuedAt: request.enqueuedAt
            ) else {
                metrics.droppedChunks += 1
                return
            }
            let synthesisStarted = ContinuousClock.now
            let groupID = UUID()
            let continuation = outputContinuation
            // Legacy one-service deployments need a short transport handoff.
            // Production uses a dedicated TTS service URL, allowing ASR and
            // true-streaming voice generation to run concurrently.
            if configuration.requiresVoicePortHandoff {
                releaseSocketForVoice()
                try? await Task.sleep(for: .milliseconds(120))
            }
            do {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    group.addTask { [neuralTTS] in
                        try await neuralTTS.synthesize(
                            text: response.text,
                            language: targetLanguage
                        ) { chunk in
                            switch continuation.yield(chunk.groupedForPlayback(groupID)) {
                            case .enqueued, .dropped, .terminated: break
                            @unknown default: break
                            }
                        }
                    }
                    group.addTask {
                        try await Task.sleep(for: TTSQueuePolicy.synthesisWatchdogTimeout)
                        throw URLError(.timedOut)
                    }
                    try await group.next()
                    group.cancelAll()
                }
            } catch is CancellationError {
                if configuration.requiresVoicePortHandoff {
                    do {
                        try await restoreSocketAfterVoice()
                    } catch {
                        isReleasingSocketForVoice = false
                        recordFailure(error)
                    }
                }
                return
            } catch {
                // A transient voice-service outage must never terminate ASR
                // or translation. Drop only this stale utterance; the next
                // confirmed phrase makes a fresh synthesis request.
                metrics.droppedChunks += 1
                if configuration.requiresVoicePortHandoff {
                    do {
                        try await restoreSocketAfterVoice()
                    } catch {
                        isReleasingSocketForVoice = false
                        recordFailure(error)
                    }
                }
                return
            }
            if configuration.requiresVoicePortHandoff {
                do {
                    try await restoreSocketAfterVoice()
                } catch {
                    recordFailure(error)
                    return
                }
            }
            metrics.synthesisMilliseconds = synthesisStarted.duration(to: .now).milliseconds
            metrics.outputChunks += 1
        } catch is CancellationError {
            return
        } catch {
            recordFailure(error)
        }
    }

    private func recordFailure(_ error: Error) {
        metrics.droppedChunks += 1
        pendingError = error
    }
}

private extension Duration {
    var milliseconds: Double {
        let value = components
        return Double(value.seconds) * 1_000 + Double(value.attoseconds) / 1e15
    }
}
