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

    static func configured(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> CUDAStreamingServerConfiguration? {
        guard environment["AI_INTERPRETER_DISABLE_GPU_STREAMING"] != "1",
              let raw = environment["AI_INTERPRETER_STREAMING_SERVER_URL"],
              let endpoint = URL(string: raw),
              ["http", "https"].contains(endpoint.scheme?.lowercased() ?? "")
        else { return nil }
        return CUDAStreamingServerConfiguration(
            endpoint: endpoint,
            authorization: environment["AI_INTERPRETER_STREAMING_SERVER_TOKEN"]
        )
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
/// platform-specific component left in this macOS client is final speech
/// rendering; capture, ASR, commit ordering and translation live on the GPU.
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
    private let outputContinuation: AsyncStream<AudioChunk>.Continuation
    private var sourceLanguage: Language?
    private var targetLanguage: Language?
    private var socket: URLSessionWebSocketTask?
    private var receiveTask: Task<Void, Never>?
    private var translationTask: Task<Void, Never>?
    private var translationContinuation: AsyncStream<QueuedSource>.Continuation?
    private var pipeline = AudioChunkPipeline()
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
        pipeline = AudioChunkPipeline()

        let request = URLRequest(url: try configuration.webSocketURL(sourceLanguage: sourceLanguage))
        let webSocket = session.webSocketTask(with: request)
        socket = webSocket
        webSocket.resume()

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
        guard socket != nil else { throw TranslationProviderError.notConnected }
        if let pendingError {
            self.pendingError = nil
            throw pendingError
        }
        metrics.inputChunks += 1
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
        guard let socket else { throw TranslationProviderError.notConnected }
        try await socket.send(.data(data))
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
            let chunks = try await withThrowingTaskGroup(of: [AudioChunk].self) { group in
                group.addTask {
                    try await AppleLocalTranslationProvider.synthesizeText(
                        response.text, language: targetLanguage
                    )
                }
                group.addTask {
                    try await Task.sleep(for: TTSQueuePolicy.synthesisWatchdogTimeout)
                    throw URLError(.timedOut)
                }
                let result = try await group.next()!
                group.cancelAll()
                return result
            }
            metrics.synthesisMilliseconds = synthesisStarted.duration(to: .now).milliseconds
            guard !TTSQueuePolicy.isStale(
                .simultaneousCommitted,
                text: response.text,
                enqueuedAt: request.enqueuedAt
            ) else {
                metrics.droppedChunks += 1
                return
            }
            let groupID = UUID()
            for chunk in chunks {
                guard !Task.isCancelled else { return }
                switch outputContinuation.yield(chunk.groupedForPlayback(groupID)) {
                case .enqueued: metrics.outputChunks += 1
                case .dropped:
                    metrics.outputChunks += 1
                    metrics.droppedChunks += 1
                    return
                case .terminated: return
                @unknown default: return
                }
            }
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
