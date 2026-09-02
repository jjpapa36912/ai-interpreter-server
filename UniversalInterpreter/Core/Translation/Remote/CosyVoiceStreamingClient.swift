@preconcurrency import Foundation

enum CosyVoiceStreamingError: LocalizedError {
    case invalidEndpoint
    case invalidResponse(Int)
    case emptyAudio
    case invalidMessage
    case firstAudioTimeout
    case jobTransportUnavailable
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
        case .firstAudioTimeout:
            "GPU 음성 서버가 첫 오디오 제한 시간 안에 응답하지 않았습니다."
        case .jobTransportUnavailable:
            "GPU 음성 서버가 PCM 조각 전송을 지원하지 않습니다."
        case let .server(message):
            "GPU 음성 서버 오류: \(message)"
        }
    }
}

/// Receives URLSession's native `Data` callbacks instead of iterating
/// `URLSession.AsyncBytes` one byte at a time. One persistent session keeps the
/// warmed HTTP/TLS connection alive, while delegate callbacks preserve a real
/// streaming first-byte path when the server emits incremental PCM.
private final class HTTPBodyStreamSession: NSObject, URLSessionDataDelegate,
    @unchecked Sendable {
    private final class RequestState: @unchecked Sendable {
        let continuation: CheckedContinuation<HTTPURLResponse, Error>
        let onData: @Sendable (Data) -> Void
        var response: HTTPURLResponse?

        init(
            continuation: CheckedContinuation<HTTPURLResponse, Error>,
            onData: @escaping @Sendable (Data) -> Void
        ) {
            self.continuation = continuation
            self.onData = onData
        }
    }

    private final class CancellationBox: @unchecked Sendable {
        private let lock = NSLock()
        private var task: URLSessionTask?
        private var cancelled = false

        func store(_ task: URLSessionTask) {
            let shouldCancel = lock.withLock { () -> Bool in
                if cancelled { return true }
                self.task = task
                return false
            }
            if shouldCancel { task.cancel() }
        }

        func cancel() {
            let task = lock.withLock { () -> URLSessionTask? in
                cancelled = true
                return self.task
            }
            task?.cancel()
        }
    }

    private let lock = NSLock()
    private var states: [Int: RequestState] = [:]
    private var session: URLSession!

    override init() {
        super.init()
        let queue = OperationQueue()
        queue.name = "AIInterpreter.CosyVoiceHTTPStream"
        queue.maxConcurrentOperationCount = 1
        session = URLSession(
            configuration: .default, delegate: self, delegateQueue: queue
        )
    }

    func perform(
        _ request: URLRequest,
        onData: @escaping @Sendable (Data) -> Void
    ) async throws -> HTTPURLResponse {
        let cancellation = CancellationBox()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let task = session.dataTask(with: request)
                let state = RequestState(
                    continuation: continuation, onData: onData
                )
                lock.withLock { states[task.taskIdentifier] = state }
                cancellation.store(task)
                task.resume()
            }
        } onCancel: {
            cancellation.cancel()
        }
    }

    func invalidate() { session.invalidateAndCancel() }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            return
        }
        lock.withLock {
            states[dataTask.taskIdentifier]?.response = http
        }
        completionHandler(.allow)
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask,
        didReceive data: Data
    ) {
        let state = lock.withLock { states[dataTask.taskIdentifier] }
        guard let state,
              let status = state.response?.statusCode,
              (200..<300).contains(status) else { return }
        state.onData(data)
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let state = lock.withLock({ states.removeValue(
            forKey: task.taskIdentifier
        ) }) else { return }
        if let error {
            state.continuation.resume(throwing: error)
        } else if let response = state.response {
            state.continuation.resume(returning: response)
        } else {
            state.continuation.resume(
                throwing: CosyVoiceStreamingError.invalidResponse(-1)
            )
        }
    }
}

/// Reassembles arbitrary network packets into one latency-first 50 ms PCM
/// frame followed by 200 ms steady-state frames. The short first frame keeps
/// first-audio latency unchanged; larger later frames cut MainActor/
/// AVAudioPlayerNode scheduling pressure by 4x during a long session.
private final class PCMTransportAccumulator: @unchecked Sendable {
    private let lock = NSLock()
    private let initialTargetBytes: Int
    private let steadyTargetBytes: Int
    private let onChunk: @Sendable (AudioChunk) -> Void
    private var pending = Data()
    private var emitted = false

    init(
        initialFramesPerChunk: Int = CosyVoiceStreamingClient.initialFramesPerChunk,
        onChunk: @escaping @Sendable (AudioChunk) -> Void
    ) {
        initialTargetBytes = max(1, initialFramesPerChunk)
            * MemoryLayout<Int16>.size
        steadyTargetBytes = CosyVoiceStreamingClient.steadyFramesPerChunk
            * MemoryLayout<Int16>.size
        self.onChunk = onChunk
    }

    func append(_ data: Data) {
        guard !data.isEmpty else { return }
        let chunks = lock.withLock { () -> [Data] in
            pending.append(data)
            var ready: [Data] = []
            var offset = 0
            while true {
                let targetBytes = (emitted || !ready.isEmpty)
                    ? steadyTargetBytes : initialTargetBytes
                guard pending.count - offset >= targetBytes else { break }
                ready.append(pending.subdata(in: offset..<(offset + targetBytes)))
                offset += targetBytes
            }
            if offset > 0 {
                // Compact once per native network callback. Repeatedly calling
                // `removeFirst` for every 50 ms frame turns a large HTTP packet
                // into quadratic copying and blocks all URLSession callbacks.
                pending.removeSubrange(0..<offset)
            }
            if !ready.isEmpty { emitted = true }
            return ready
        }
        for chunk in chunks {
            onChunk(CosyVoiceStreamingClient.makeTransportChunk(chunk))
        }
    }

    func finish() throws {
        let tail = lock.withLock { () -> Data in
            let tail = pending
            pending.removeAll(keepingCapacity: false)
            if !tail.isEmpty { emitted = true }
            return tail
        }
        guard tail.count.isMultiple(of: MemoryLayout<Int16>.size) else {
            throw CosyVoiceStreamingError.invalidMessage
        }
        if !tail.isEmpty {
            onChunk(CosyVoiceStreamingClient.makeTransportChunk(tail))
        }
    }

    var hasEmitted: Bool { lock.withLock { emitted } }
}

/// Client for the product TTS boundary. The server response is raw signed
/// little-endian PCM16, 24 kHz, mono. Native URLSession data callbacks are
/// repacketized into fixed 50 ms chunks without buffering a whole utterance or
/// paying one Swift async suspension per byte.
final class CosyVoiceStreamingClient: @unchecked Sendable {
    private struct RequestBody: Encodable {
        let text: String
        let language: String
        let voiceID: String?
        let speed: Double

        enum CodingKeys: String, CodingKey {
            case text, language
            case voiceID = "voice_id"
            case speed
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

    private struct JobCreationResponse: Decodable {
        let jobID: String
        let sampleRate: Double
        let transport: String

        enum CodingKeys: String, CodingKey {
            case transport
            case jobID = "job_id"
            case sampleRate = "sample_rate"
        }
    }

    static let sampleRate = 24_000.0
    // Generate the neural performance itself a little faster instead of
    // making AVAudioUnitTimePitch do all catch-up work after synthesis.
    // CosyVoice preserves its learned phrasing at this modest value while
    // emitting fewer acoustic frames and releasing the serial GPU sooner.
    static let speechGenerationSpeed = 1.30
    // The first packet is 50 ms for low startup latency. Once playback has a
    // foothold, use 200 ms packets to avoid hundreds of tiny buffer schedules
    // when a whole-body proxy response arrives in one burst.
    static let initialFramesPerChunk = 1_200
    // Confirmed follow-up speech gets a 300 ms foothold before playback. The
    // G-Cube HTTP proxy can deliver an otherwise valid PCM body with small
    // timing gaps; starting from only 50 ms lets playback consume those gaps
    // and sounds like repeated mid-phrase cuts. The latency-critical starter
    // deliberately keeps the 50 ms profile, so first speech is not delayed.
    static let continuityInitialFramesPerChunk = 7_200
    static let steadyFramesPerChunk = 4_800
    static let framesPerChunk = initialFramesPerChunk

    let endpoint: URL
    private let authorization: String?
    private let session: URLSession
    private let streamingSession = HTTPBodyStreamSession()

    init(endpoint: URL, authorization: String? = nil, session: URLSession = .shared) {
        self.endpoint = endpoint
        self.authorization = authorization
        self.session = session
    }

    deinit {
        streamingSession.invalidate()
    }

    static func configured(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> CosyVoiceStreamingClient? {
        // A configured launchd URL must not force every session through the
        // remote voice service. This switch lets the validated simultaneous
        // translation pipeline keep its existing boundaries and ordering
        // while selecting the low-latency local voice from the first phrase.
        guard environment["AI_INTERPRETER_DISABLE_REMOTE_TTS"] != "1" else {
            return nil
        }
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
        speed: Double = CosyVoiceStreamingClient.speechGenerationSpeed,
        initialFramesPerChunk: Int = CosyVoiceStreamingClient.initialFramesPerChunk,
        onChunk: @escaping @Sendable (AudioChunk) -> Void
    ) async throws {
        // G-Cube may buffer one long HTTP streaming response until synthesis
        // completes. The v2 transport starts generation in the background and
        // returns each numbered PCM block in its own short response, which the
        // gateway cannot defer behind the rest of the utterance. Fall back only
        // when talking to an older server that does not expose the job route.
        do {
            try await synthesizePCMJob(
                text: text, language: language, voiceID: voiceID,
                speed: speed,
                initialFramesPerChunk: initialFramesPerChunk, onChunk: onChunk
            )
        } catch CosyVoiceStreamingError.jobTransportUnavailable {
            try await synthesizeHTTP(
                text: text, language: language, voiceID: voiceID,
                speed: speed,
                initialFramesPerChunk: initialFramesPerChunk, onChunk: onChunk
            )
        }
    }

    /// Opens and validates the reusable HTTP/TLS connection before source
    /// audio starts. This never synthesizes or plays filler speech; it only
    /// removes DNS, TLS and proxy setup from the first real TTS request.
    func prewarmConnection() async {
        var request = URLRequest(url: endpoint.appending(path: "health"))
        request.httpMethod = "GET"
        request.timeoutInterval = 5
        if let authorization {
            request.setValue("Bearer \(authorization)", forHTTPHeaderField: "Authorization")
        }
        _ = try? await streamingSession.perform(request, onData: { _ in })
    }

    /// Selects the voice route once, before capture begins. A configured URL
    /// only means that the user prefers the GPU voice; it does not prove that
    /// the deployment is currently running or that its models are ready.
    ///
    /// The probe deliberately uses `/ready` rather than `/health`: a freshly
    /// deployed container can accept HTTP while its TTS model is still
    /// unavailable. Two short attempts absorb a single transient G-Cube proxy
    /// reset without making an offline server delay interpretation startup for
    /// several seconds.
    func isReadyForSession() async -> Bool {
        for attempt in 0..<RemoteTTSAvailabilityPolicy.attemptCount {
            guard !Task.isCancelled else { return false }
            var request = URLRequest(url: endpoint.appending(path: "ready"))
            request.httpMethod = "GET"
            request.timeoutInterval = RemoteTTSAvailabilityPolicy.requestTimeoutSeconds
            if let authorization {
                request.setValue("Bearer \(authorization)", forHTTPHeaderField: "Authorization")
            }
            do {
                let response = try await streamingSession.perform(request, onData: { _ in })
                if RemoteTTSAvailabilityPolicy.isReady(statusCode: response.statusCode) {
                    return true
                }
            } catch {
                // A second bounded attempt below distinguishes a dead service
                // from one transient proxy reset. The session then locks to
                // local speech rather than repeatedly blocking on the GPU.
            }
            guard attempt + 1 < RemoteTTSAvailabilityPolicy.attemptCount else { break }
            try? await Task.sleep(for: .milliseconds(
                RemoteTTSAvailabilityPolicy.retryDelayMilliseconds
            ))
        }
        return false
    }

    /// Runs one short, silent neural generation so the model's first real
    /// utterance does not pay lazy CUDA/Metal graph and allocator costs.
    /// Generated PCM is deliberately discarded; this is never user-visible
    /// filler speech and does not enter the application's playback queue.
    func prewarmSynthesis(language: Language) async {
        await prewarmConnection()
        try? await synthesize(
            text: Self.primingText(for: language),
            language: language,
            onChunk: { _ in }
        )
    }

    static func primingText(for language: Language) -> String {
        switch language {
        case .korean: "준비됐습니다."
        case .english: "Ready to begin."
        }
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
        speed: Double = CosyVoiceStreamingClient.speechGenerationSpeed,
        onChunk: @escaping @Sendable (AudioChunk) -> Void
    ) async throws {
        let socketSession = URLSession(configuration: .ephemeral)
        let socket = socketSession.webSocketTask(with: try webSocketURL())
        socket.resume()
        defer {
            socket.cancel(with: .normalClosure, reason: nil)
            socketSession.invalidateAndCancel()
        }
        let body = try JSONEncoder().encode(RequestBody(
            text: text, language: language.rawValue, voiceID: voiceID,
            speed: speed
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
        speed: Double,
        initialFramesPerChunk: Int,
        onChunk: @escaping @Sendable (AudioChunk) -> Void
    ) async throws {
        let body = try JSONEncoder().encode(RequestBody(
            text: text, language: language.rawValue, voiceID: voiceID,
            speed: speed
        ))
        var lastStatus = -1
        let retryDelays = [50, 75, 100, 150, 225, 350, 500]
        for attempt in 0...retryDelays.count {
            var request = URLRequest(url: endpoint.appending(path: "v1/tts/stream"))
            request.httpMethod = "POST"
            // The public G-Cube endpoint can render a complete utterance behind
            // its proxy before returning PCM. Keep a bounded transport timeout,
            // but treat one miss as transient: changing to the platform voice
            // mid-session is more disruptive than retrying the configured GPU
            // voice on the following clause.
            request.timeoutInterval = RemoteTTSCircuitBreakerPolicy
                .firstAudioDeadlineSeconds
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("audio/L16;rate=24000;channels=1", forHTTPHeaderField: "Accept")
            if let authorization {
                request.setValue("Bearer \(authorization)", forHTTPHeaderField: "Authorization")
            }
            request.httpBody = body

            // Keep the normal path on one warmed URLSession. A gateway failure
            // can poison that keep-alive connection, so retries get a fresh
            // delegate-backed session and dispose it after the attempt.
            let retrySession = attempt == 0 ? nil : HTTPBodyStreamSession()
            let requestSession = retrySession ?? streamingSession
            let accumulator = PCMTransportAccumulator(
                initialFramesPerChunk: initialFramesPerChunk,
                onChunk: onChunk
            )
            let http: HTTPURLResponse
            do {
                http = try await requestSession.perform(
                    request, onData: { accumulator.append($0) }
                )
            } catch {
                retrySession?.invalidate()
                if let urlError = error as? URLError,
                   urlError.code == .timedOut,
                   !accumulator.hasEmitted {
                    throw CosyVoiceStreamingError.firstAudioTimeout
                }
                guard !accumulator.hasEmitted, !Task.isCancelled,
                      attempt < retryDelays.count else { throw error }
                try await Task.sleep(for: .milliseconds(retryDelays[attempt]))
                continue
            }
            lastStatus = http.statusCode
            if (200..<300).contains(http.statusCode) {
                try accumulator.finish()
                retrySession?.invalidate()
                guard accumulator.hasEmitted else {
                    throw CosyVoiceStreamingError.emptyAudio
                }
                return
            }
            retrySession?.invalidate()
            guard [502, 503, 504].contains(http.statusCode), attempt < retryDelays.count else {
                throw CosyVoiceStreamingError.invalidResponse(http.statusCode)
            }
            try await Task.sleep(for: .milliseconds(retryDelays[attempt]))
        }
        throw CosyVoiceStreamingError.invalidResponse(lastStatus)
    }

    private func synthesizePCMJob(
        text: String,
        language: Language,
        voiceID: String?,
        speed: Double,
        initialFramesPerChunk: Int,
        onChunk: @escaping @Sendable (AudioChunk) -> Void
    ) async throws {
        let requestID = UUID().uuidString
        let firstAudioStarted = ContinuousClock.now
        let body = try JSONEncoder().encode(RequestBody(
            text: text, language: language.rawValue, voiceID: voiceID,
            speed: speed
        ))
        var creationData = Data()
        var creationHTTP: HTTPURLResponse?
        let createRetryDelays = [60, 100, 180]
        for attempt in 0...createRetryDelays.count {
            var createRequest = URLRequest(
                url: endpoint.appending(path: "v1/tts/jobs")
            )
            createRequest.httpMethod = "POST"
            createRequest.timeoutInterval = RemoteTTSCircuitBreakerPolicy
                .firstAudioDeadlineSeconds
            createRequest.setValue(
                "application/json", forHTTPHeaderField: "Content-Type"
            )
            // A proxy can reset after the server accepted POST. Retrying with
            // the same id lets the server return the existing job instead of
            // synthesizing the sentence twice and blocking every later line.
            createRequest.setValue(requestID, forHTTPHeaderField: "X-TTS-Request-ID")
            if let authorization {
                createRequest.setValue(
                    "Bearer \(authorization)", forHTTPHeaderField: "Authorization"
                )
            }
            createRequest.httpBody = body
            do {
                let (data, response) = try await session.data(for: createRequest)
                guard let http = response as? HTTPURLResponse else {
                    throw CosyVoiceStreamingError.invalidResponse(-1)
                }
                creationData = data
                creationHTTP = http
                if (200..<300).contains(http.statusCode)
                    || http.statusCode == 404 || http.statusCode == 405 {
                    break
                }
                guard [502, 503, 504].contains(http.statusCode),
                      attempt < createRetryDelays.count else { break }
            } catch {
                guard !Task.isCancelled,
                      attempt < createRetryDelays.count else { throw error }
            }
            try await Task.sleep(for: .milliseconds(createRetryDelays[attempt]))
        }
        guard let creationHTTP else {
            throw CosyVoiceStreamingError.invalidResponse(-1)
        }
        if creationHTTP.statusCode == 404 || creationHTTP.statusCode == 405 {
            throw CosyVoiceStreamingError.jobTransportUnavailable
        }
        guard (200..<300).contains(creationHTTP.statusCode) else {
            throw CosyVoiceStreamingError.invalidResponse(creationHTTP.statusCode)
        }
        let job = try JSONDecoder().decode(JobCreationResponse.self, from: creationData)
        guard job.transport == "pcm-pull-v1",
              job.sampleRate == Self.sampleRate,
              !job.jobID.isEmpty else {
            throw CosyVoiceStreamingError.invalidMessage
        }

        let accumulator = PCMTransportAccumulator(
            initialFramesPerChunk: initialFramesPerChunk,
            onChunk: onChunk
        )
        var sequence = 0
        while !Task.isCancelled {
            if !accumulator.hasEmitted,
               firstAudioStarted.duration(to: .now)
                > .seconds(RemoteTTSCircuitBreakerPolicy.firstAudioDeadlineSeconds) {
                throw CosyVoiceStreamingError.firstAudioTimeout
            }
            var components = URLComponents(
                url: endpoint.appending(
                    path: "v1/tts/jobs/\(job.jobID)/chunks/\(sequence)"
                ),
                resolvingAgainstBaseURL: false
            )
            components?.queryItems = [URLQueryItem(name: "wait_ms", value: "800")]
            guard let chunkURL = components?.url else {
                throw CosyVoiceStreamingError.invalidEndpoint
            }
            var chunkRequest = URLRequest(url: chunkURL)
            chunkRequest.httpMethod = "GET"
            chunkRequest.timeoutInterval = 1.8
            chunkRequest.setValue(
                "audio/L16;rate=24000;channels=1", forHTTPHeaderField: "Accept"
            )
            if let authorization {
                chunkRequest.setValue(
                    "Bearer \(authorization)", forHTTPHeaderField: "Authorization"
                )
            }
            let (chunkData, chunkResponse) = try await session.data(for: chunkRequest)
            guard let chunkHTTP = chunkResponse as? HTTPURLResponse else {
                throw CosyVoiceStreamingError.invalidResponse(-1)
            }
            let isDone = chunkHTTP.value(
                forHTTPHeaderField: "X-TTS-Done"
            ) == "1"
            switch chunkHTTP.statusCode {
            case 200..<300 where chunkHTTP.statusCode != 204:
                guard !chunkData.isEmpty,
                      chunkData.count.isMultiple(of: MemoryLayout<Int16>.size)
                else { throw CosyVoiceStreamingError.invalidMessage }
                accumulator.append(chunkData)
                sequence += 1
                if isDone {
                    try accumulator.finish()
                    return
                }
            case 204:
                if isDone {
                    try accumulator.finish()
                    guard accumulator.hasEmitted else {
                        throw CosyVoiceStreamingError.emptyAudio
                    }
                    return
                }
            default:
                if chunkHTTP.statusCode == 404 || chunkHTTP.statusCode == 410 {
                    throw CosyVoiceStreamingError.server("TTS 작업이 만료되었습니다.")
                }
                throw CosyVoiceStreamingError.invalidResponse(chunkHTTP.statusCode)
            }
        }
        throw CancellationError()
    }

    static func decodePCM16(_ data: Data) -> [Float] {
        data.withUnsafeBytes { rawBuffer in
            let samples = rawBuffer.bindMemory(to: Int16.self)
            return samples.map { Float(Int16(littleEndian: $0)) / Float(Int16.max) }
        }
    }

    static func transportChunks(
        _ data: Data,
        initialFramesPerChunk: Int = CosyVoiceStreamingClient.initialFramesPerChunk
    ) -> [Data] {
        guard !data.isEmpty else { return [] }
        let initialBytes = max(1, initialFramesPerChunk) * MemoryLayout<Int16>.size
        let steadyBytes = steadyFramesPerChunk * MemoryLayout<Int16>.size
        var result: [Data] = []
        var offset = 0
        while offset < data.count {
            let target = result.isEmpty ? initialBytes : steadyBytes
            let end = min(offset + target, data.count)
            result.append(data.subdata(in: offset..<end))
            offset = end
        }
        return result
    }

    static func makeTransportChunk(_ data: Data) -> AudioChunk {
        AudioChunk(
            samples: decodePCM16(data),
            format: AudioFormatInfo(sampleRate: sampleRate, channelCount: 1, bitsPerChannel: 16),
            capturedAt: .now
        )
    }
}

struct RemoteTTSAvailabilityPolicy: Sendable {
    static let attemptCount = 2
    static let requestTimeoutSeconds: TimeInterval = 0.9
    static let retryDelayMilliseconds = 80

    static func isReady(statusCode: Int) -> Bool {
        (200..<300).contains(statusCode)
    }
}
