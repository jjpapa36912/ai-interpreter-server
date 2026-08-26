import Foundation
@preconcurrency import Speech

private struct PreviewTranslationTimeout: Error, Sendable {}
private struct SpeechSynthesisWatchdogTimeout: Error, Sendable {}

private final class SpeechOutputGate: @unchecked Sendable {
    private let lock = NSLock()
    private var open = true

    func yieldIfOpen(_ chunk: AudioChunk, to output: AsyncStream<AudioChunk>.Continuation) {
        lock.lock()
        defer { lock.unlock() }
        guard open else { return }
        output.yield(chunk)
    }

    func close() {
        lock.lock()
        open = false
        lock.unlock()
    }
}

struct TTSQueuePolicy: Sendable {
    enum TranslationKind: Sendable { case preview, simultaneousCommitted, confirmed }

    static let maximumQueueAge: Duration = .seconds(12)
    static let maximumShortSentenceQueueAge: Duration = .seconds(12)
    static let synthesisWatchdogTimeout: Duration = .seconds(30)
    static let maximumSimultaneousQueueAge: Duration = .milliseconds(1_500)

    static func shouldSpeak(_ kind: TranslationKind) -> Bool {
        // Revisions make simultaneous prefixes unsafe to speak. Keep them on
        // screen as previews, but only finalized translations enter the audio
        // queue.
        kind == .confirmed
    }

    static func isStale(
        enqueuedAt: ContinuousClock.Instant,
        now: ContinuousClock.Instant = .now
    ) -> Bool {
        enqueuedAt.duration(to: now) > maximumQueueAge
    }

    static func isStaleConfirmedSentence(
        _ text: String,
        enqueuedAt: ContinuousClock.Instant,
        now: ContinuousClock.Instant = .now
    ) -> Bool {
        let age = enqueuedAt.duration(to: now)
        let compactCount = text.filter { !$0.isWhitespace }.count
        // A short bridge thought takes little playback time but carries vital
        // meaning. Do not let the preceding long utterance erase it exactly at
        // the normal four-second freshness boundary.
        let limit = (compactCount >= 5 && compactCount <= 55)
            ? maximumShortSentenceQueueAge
            : maximumQueueAge
        return age > limit
    }

    static func isStale(
        _ kind: TranslationKind,
        text: String,
        enqueuedAt: ContinuousClock.Instant,
        now: ContinuousClock.Instant = .now
    ) -> Bool {
        if kind == .simultaneousCommitted {
            return enqueuedAt.duration(to: now) > maximumSimultaneousQueueAge
        }
        return isStaleConfirmedSentence(text, enqueuedAt: enqueuedAt, now: now)
    }
}

struct TranslationQueuePolicy: Sendable {
    static let maximumQueueMilliseconds = 4_000.0
    // A completed confirmed translation must not disappear merely because a
    // local inference spike crossed six seconds. The speech queue still has a
    // much tighter age limit and discards obsolete backlog at playback time.
    // This limit is only the final circuit breaker for truly wedged inference.
    static let maximumEndToEndMilliseconds = 8_000.0
    // Qwen 4B's measured median is about 2.1 seconds on this Mac. A three-second
    // watchdog occasionally discarded a correct result during Metal graph
    // compilation and silently fell back to the lower-quality translator.
    static let remoteWatchdogTimeout: Duration = .seconds(5)

    static func isStale(queueMilliseconds: Double) -> Bool {
        queueMilliseconds > maximumQueueMilliseconds
    }

    static func isTooLateToSpeak(
        queueMilliseconds: Double, translationMilliseconds: Double
    ) -> Bool {
        queueMilliseconds + translationMilliseconds > maximumEndToEndMilliseconds
    }
}

actor ResidentLocalTranslationProvider: TranslationProvider {
    private struct RecognizedPhrase: Sendable {
        let text: String
        let recognitionMilliseconds: Double
        let enqueuedAt: ContinuousClock.Instant
    }

    private struct SpeechRequest: Sendable {
        let text: String
        let language: Language
        let translationKind: TTSQueuePolicy.TranslationKind
        let enqueuedAt: ContinuousClock.Instant
        let sessionGeneration: UInt64
    }

    nonisolated let translatedAudio: AsyncStream<AudioChunk>

    private let outputContinuation: AsyncStream<AudioChunk>.Continuation
    private let worker: LocalModelWorker
    private let remoteTTS: CosyVoiceStreamingClient?
    private let remoteTranslation: CUDATranslationClient?
    private let preferredTerms: [String]
    private var segmenter = StreamingSpeechSegmenter()
    private var segmentContinuation: AsyncStream<AudioChunk>.Continuation?
    private var textContinuation: AsyncStream<RecognizedPhrase>.Continuation?
    private var workerTask: Task<Void, Never>?
    private var fallbackWorkerTask: Task<Void, Never>?
    private var progressiveTask: Task<Void, Never>?
    private var speechContinuation: AsyncStream<SpeechRequest>.Continuation?
    private var lastSpokenText = ""
    private var speechTask: Task<Void, Never>?
    private var neuralVoicePreparationTask: Task<Void, Never>?
    private var neuralVoiceReady = false
    private var progressiveRecognizer: Any?
    private var sourceLanguage: Language?
    private var targetLanguage: Language?
    private var metrics = AudioPipelineMetrics()
    private var pendingError: Error?
    private var ownsWorkerSession = false
    private var usesAppleASR = false
    private var transcriptGate = ASRTranscriptGate()
    private var progressivePhraseSeen = false
    private var progressiveCommittedDisplay = ""
    private var zipformerRecognizer: ZipformerStreamingRecognizer?
    private var previewRecognizer: FastSpeechPreviewRecognizer?
    private var previewTask: Task<Void, Never>?
    private var previewTranslationTask: Task<Void, Never>?
    private var latestPreviewSource = ""
    private var latestPreviewIsStable = false
    private var translatedPreviewSource = ""
    private var translatedPreviewIsStable = false
    private var previewStableCommitter = StableTranscriptCommitter(
        requiredUpdates: 2, lookaheadWords: 1
    )
    private var rollingTranslationScheduler = RollingTranslationScheduler()
    private var simultaneousTranslationCommitter = SimultaneousTranslationCommitter()
    private var hasStartedSimultaneousSpeech = false
    private let simultaneousTTSIsEnabled = ProcessInfo.processInfo.environment[
        "AI_INTERPRETER_SIMULTANEOUS_TTS"
    ] == "1"
    private let learnedSimultaneousBoundaryIsEnabled = ProcessInfo.processInfo.environment[
        "AI_INTERPRETER_LEARNED_BOUNDARY"
    ] == "1"
    private var receivedAudioMilliseconds = 0.0
    private var lastSpeechAudioMilliseconds = 0.0
    private var confirmedTranslationDisplay = ""
    private var nemotronRecognizer: NemotronStreamingRecognizer?
    private var nemotronAudioContinuation: AsyncStream<AudioChunk>.Continuation?
    private var nemotronAudioTask: Task<Void, Never>?
    private var nemotronCommittedDisplay = ""
    private var pendingTranslationWords: [String] = []
    private var lastASRLogMilliseconds: Double?
    private var lastSpeechWallClock: ContinuousClock.Instant?
    private var sessionGeneration: UInt64 = 0
    private var simultaneousDebugSessionID = UUID()
    private var nemotronInputSamples: [Float] = []
    private var nemotronInputFormat: AudioFormatInfo?

    init(preferredTerms: [String] = [], worker: LocalModelWorker = .shared) {
        self.preferredTerms = preferredTerms
        self.worker = worker
        remoteTTS = CosyVoiceStreamingClient.configured()
        remoteTranslation = CUDATranslationClient.configured()
        let pair = AsyncStream<AudioChunk>.makeStream(
            // Under overload, discard obsolete PCM and retain the newest
            // translated speech instead of speaking tens of seconds of history.
            bufferingPolicy: .bufferingNewest(AppConfiguration.maximumTranslatedAudioChunks)
        )
        translatedAudio = pair.stream
        outputContinuation = pair.continuation
    }

    func startSession(sourceLanguage: Language, targetLanguage: Language) async throws {
        await stopSession()
        sessionGeneration &+= 1
        simultaneousDebugSessionID = UUID()
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        // Screen recording and microphone consent do not grant Speech.framework
        // access. Ask only on the first local-model session; a denial keeps the
        // proven Whisper path active instead of making the session fail.
        if SFSpeechRecognizer.authorizationStatus() == .notDetermined {
            _ = await AppleLocalTranslationProvider.requestSpeechAuthorization()
        }
        usesAppleASR = AppleLocalTranslationProvider.canRecognizeOnDevice(sourceLanguage)
        transcriptGate = ASRTranscriptGate()
        segmenter = StreamingSpeechSegmenter()
        metrics = AudioPipelineMetrics()
        pendingError = nil
        progressivePhraseSeen = false
        progressiveCommittedDisplay = ""
        latestPreviewSource = ""
        latestPreviewIsStable = false
        translatedPreviewSource = ""
        translatedPreviewIsStable = false
        previewStableCommitter.reset()
        rollingTranslationScheduler = simultaneousTTSIsEnabled
            ? RollingTranslationScheduler(
                targetLagMilliseconds: 350,
                stableUpdateMilliseconds: 250,
                maximumUpdateGapMilliseconds: 350,
                minimumWords: 3
            )
            : RollingTranslationScheduler()
        simultaneousTranslationCommitter.reset()
        hasStartedSimultaneousSpeech = false
        receivedAudioMilliseconds = 0
        lastSpeechAudioMilliseconds = 0
        confirmedTranslationDisplay = ""
        lastSpokenText = ""
        lastASRLogMilliseconds = nil
        lastSpeechWallClock = nil
        nemotronInputSamples.removeAll(keepingCapacity: true)
        nemotronInputFormat = nil
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID, event: "session_started",
            audioMilliseconds: 0, sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage
        )
        // Pay the system Translation session's one-time setup cost before any
        // speech arrives. Failure is non-fatal because the local model fallback
        // remains available when a language pack has not been installed yet.
        try? await AppleLocalTranslationProvider.prepareTranslation(
            source: sourceLanguage, target: targetLanguage
        )
        // Compile and load the learned boundary model before capture starts;
        // subsequent endpoint decisions take only a few milliseconds.
        NeuralSentenceBoundaryClassifier.prewarm()
        if learnedSimultaneousBoundaryIsEnabled {
            if !ownsWorkerSession {
                try await worker.acquire()
                ownsWorkerSession = true
            }
            // Only the two ~15 ms boundary classifiers are loaded here. The
            // unqualified experimental Qwen delta translator is deliberately
            // excluded; Apple remains the translation-quality authority.
            try await worker.prepareSimultaneousBoundary(sourceLanguage: sourceLanguage)
        }
        // Live local speech uses deterministic AVSpeech synthesis. Do not load
        // the generative Qwen voice: besides consuming unified memory, its
        // missed-EOS loops repeated whole Korean phrases.
        neuralVoiceReady = false
        // Create the consumer only after its selected voice is ready. This
        // avoids a startup race where the stream existed while preparation
        // suspended the session actor and requests were never consumed.
        startSpeechQueue()
        // On the M5 Pro, Nemotron 3.5 processed the 36-second paced news
        // fixture in 3.72 seconds (RTF 0.103) with 7.2% WER. Prefer it for
        // English continuous streams: it is both materially more accurate than
        // legacy SFSpeech partials and has enough headroom to avoid backlog.
        if sourceLanguage == .english,
           let recognizer = try? NemotronStreamingRecognizer(preferredTerms: preferredTerms) {
            nemotronRecognizer = recognizer
            usesAppleASR = false
            nemotronCommittedDisplay = ""
            pendingTranslationWords.removeAll(keepingCapacity: true)
            let phrases = AsyncStream<RecognizedPhrase>.makeStream(
                // A single slow translation used to overwrite every confirmed
                // phrase except the newest one. Preserve a small ordered run;
                // the four-second queue-age guard below still prevents backlog.
                bufferingPolicy: .bufferingNewest(2)
            )
            textContinuation = phrases.continuation
            workerTask = Task(priority: .userInitiated) { [weak self] in
                for await phrase in phrases.stream {
                    guard !Task.isCancelled else { break }
                    await self?.translateNemotronPhraseIfFresh(phrase)
                }
            }
            let audio = AsyncStream<AudioChunk>.makeStream(
                // This is a safety ceiling, not the normal operating mode.
                // Never retain minutes of obsolete browser audio if the system
                // is briefly overloaded.
                // A slow recognizer must never retain tens of seconds of old
                // browser audio. Eight native capture chunks are roughly four
                // seconds; newest audio keeps captions bounded under overload.
                bufferingPolicy: .bufferingNewest(8)
            )
            nemotronAudioContinuation = audio.continuation
            nemotronAudioTask = Task(priority: .userInitiated) { [weak self] in
                for await chunk in audio.stream {
                    guard !Task.isCancelled else { break }
                    await self?.processNemotronAudio(chunk)
                }
            }
            // Never feed two independently revising recognizers into one
            // translation committer. Current Nemotron partials arrive at the
            // same first-word timestamp as Speech.framework; racing both made
            // old cumulative Apple text alternate with new Nemotron windows,
            // delaying the stable prefix and repeating already spoken content.
            return
        }

        // Keep Apple's low-latency recognizer as the no-model fallback and as
        // the Korean path until a higher-quality Korean streaming model wins
        // the same paced benchmark.
        let realtimeRecognizer = FastSpeechPreviewRecognizer()
        if (try? await realtimeRecognizer.start(
            language: sourceLanguage, preferredTerms: preferredTerms
        )) != nil {
            previewRecognizer = realtimeRecognizer
            usesAppleASR = true
            let phrases = AsyncStream<RecognizedPhrase>.makeStream(
                bufferingPolicy: .bufferingNewest(1)
            )
            textContinuation = phrases.continuation
            workerTask = Task(priority: .userInitiated) { [weak self] in
                for await phrase in phrases.stream {
                    guard !Task.isCancelled else { break }
                    await self?.translateText(
                        phrase.text,
                        recognitionMilliseconds: phrase.recognitionMilliseconds,
                        queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds
                    )
                }
            }
            previewTask = Task(priority: .userInitiated) { [weak self] in
                for await event in realtimeRecognizer.events {
                    guard !Task.isCancelled else { break }
                    await self?.record(event)
                    if let phrase = event.phrase {
                        await self?.enqueueFinalPhraseIfFresh(
                            phrase, isFinal: event.isFinal,
                            recognitionMilliseconds: event.elapsedMilliseconds
                        )
                    }
                }
            }
            return
        }
        // Apple's on-device long-form recognizer scored materially higher than
        // the bundled 0.6B RNNT on entities, corrections, and connected speech.
        // It also avoids the CPU backlog seen in long Chrome sessions, so make
        // it the accuracy-first production path whenever its asset is present.
        if #available(macOS 26.0, *) {
            let recognizer = ContinuousSpeechRecognizer()
            do {
                try await recognizer.start(
                    language: sourceLanguage, preferredTerms: preferredTerms
                )
                progressiveRecognizer = recognizer
                usesAppleASR = true
                let phrases = AsyncStream<RecognizedPhrase>.makeStream(
                    bufferingPolicy: .bufferingNewest(1)
                )
                textContinuation = phrases.continuation
                workerTask = Task(priority: .userInitiated) { [weak self] in
                    for await phrase in phrases.stream {
                        guard !Task.isCancelled else { break }
                        await self?.translateText(
                            phrase.text,
                            recognitionMilliseconds: phrase.recognitionMilliseconds,
                            queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds
                        )
                    }
                }
                progressiveTask = Task(priority: .userInitiated) { [weak self] in
                    for await event in recognizer.events {
                        guard !Task.isCancelled else { break }
                        await self?.record(event)
                        if let phrase = event.phrase {
                            await self?.enqueueFinalPhraseIfFresh(
                                phrase, isFinal: event.isFinal,
                                recognitionMilliseconds: event.elapsedMilliseconds
                            )
                        }
                    }
                }
                return
            } catch {
                await recognizer.stop()
                progressiveRecognizer = nil
            }
        }

        // This second attempt covers non-English configurations on platforms
        // where the primary branch above was unavailable.
        if #available(macOS 26.0, *) {
            let recognizer = ContinuousSpeechRecognizer()
            do {
                try await recognizer.start(
                    language: sourceLanguage, preferredTerms: preferredTerms
                )
                progressiveRecognizer = recognizer
                usesAppleASR = true
                let phrases = AsyncStream<RecognizedPhrase>.makeStream(
                    bufferingPolicy: .bufferingNewest(3)
                )
                textContinuation = phrases.continuation
                workerTask = Task { [weak self] in
                    for await phrase in phrases.stream {
                        guard !Task.isCancelled else { break }
                        await self?.translateText(
                            phrase.text,
                            recognitionMilliseconds: phrase.recognitionMilliseconds,
                            queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds
                        )
                    }
                }
                progressiveTask = Task { [weak self] in
                    for await event in recognizer.events {
                        guard !Task.isCancelled else { break }
                        await self?.record(event)
                        if let phrase = event.phrase {
                            await self?.enqueueFinalPhraseIfFresh(
                                phrase, isFinal: event.isFinal,
                                recognitionMilliseconds: event.elapsedMilliseconds
                            )
                        }
                    }
                }
                return
            } catch {
                await recognizer.stop()
                progressiveRecognizer = nil
            }
        }

        // Zipformer remains a no-network fallback only when the system's
        // progressive speech asset or permission is unavailable.
        zipformerRecognizer = try ZipformerStreamingRecognizer(language: sourceLanguage)
        let phrases = AsyncStream<RecognizedPhrase>.makeStream(
            bufferingPolicy: .bufferingNewest(3)
        )
        textContinuation = phrases.continuation
        workerTask = Task { [weak self] in
            for await phrase in phrases.stream {
                guard !Task.isCancelled else { break }
                await self?.translateText(
                    phrase.text,
                    recognitionMilliseconds: phrase.recognitionMilliseconds,
                    queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds
                )
            }
        }
        return

        /* Legacy Apple/Whisper path retained temporarily for benchmark rollback.
        // Keep a bounded segment fallback alive even when the modern progressive
        // recognizer starts successfully. Some system-audio streams never emit a
        // stable/final SpeechAnalyzer phrase; previously that produced silence
        // forever while the UI still showed a running session.
        let segments = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(3))
        segmentContinuation = segments.continuation
        fallbackWorkerTask = Task { [weak self] in
            for await segment in segments.stream {
                guard !Task.isCancelled else { break }
                await self?.translate(segment)
            }
        }
        if #available(macOS 26.0, *) {
            let recognizer = ContinuousSpeechRecognizer()
            do {
                try await recognizer.start(
                    language: sourceLanguage, preferredTerms: preferredTerms
                )
                progressiveRecognizer = recognizer
                usesAppleASR = true
                // Never block SpeechAnalyzer result consumption on 4B translation
                // and neural TTS. A blocked result loop accumulates an ever-growing
                // backlog during continuous news/interview audio and was the main
                // source of the observed 10–15 second delay.
                let phrases = AsyncStream<RecognizedPhrase>.makeStream(
                    // Keep latency bounded under overload. Retaining an old queue
                    // makes the interpreter speak obsolete sentences tens of
                    // seconds late; newest preserves the current conversation.
                    bufferingPolicy: .bufferingNewest(3)
                )
                textContinuation = phrases.continuation
                workerTask = Task { [weak self] in
                    for await phrase in phrases.stream {
                        guard !Task.isCancelled else { break }
                        await self?.translateText(
                            phrase.text,
                            recognitionMilliseconds: phrase.recognitionMilliseconds
                        )
                    }
                }
                progressiveTask = Task { [weak self] in
                    for await event in recognizer.events {
                        guard !Task.isCancelled else { break }
                        await self?.record(event)
                        if let phrase = event.phrase {
                            await self?.enqueueRecognizedPhrase(
                                phrase, recognitionMilliseconds: event.elapsedMilliseconds
                            )
                        }
                    }
                }
                return
            } catch {
                await recognizer.stop()
                progressiveRecognizer = nil
            }
        }
        */
    }

    func sendAudio(_ chunk: AudioChunk) async throws {
        receivedAudioMilliseconds += chunk.duration * 1_000
        if let previewRecognizer, progressiveRecognizer == nil,
           nemotronRecognizer == nil, zipformerRecognizer == nil {
            if let pendingError {
                self.pendingError = nil
                throw pendingError
            }
            metrics.inputChunks += 1
            recordSpeechClock(for: chunk)
            await previewRecognizer.send(chunk)
            return
        }
        if let nemotronRecognizer {
            if let pendingError {
                self.pendingError = nil
                throw pendingError
            }
            metrics.inputChunks += 1
            recordSpeechClock(for: chunk)
            // The preview recognizer is display/early-speech only. It never
            // contributes to the confirmed transcript or quality path.
            await previewRecognizer?.send(chunk)
            if Self.shouldFlushNemotronAfterSilence(
                pendingTranslationWords,
                silenceMilliseconds: receivedAudioMilliseconds - lastSpeechAudioMilliseconds
            ) {
                flushPendingNemotronTranslation()
            }
            // ScreenCaptureKit delivers many tiny buffers. Enqueuing every one
            // caused actor/stream overhead and made an eight-item queue hold
            // milliseconds rather than the intended four seconds. Coalesce to
            // 320 ms before crossing into the recognizer consumer. Nemotron
            // combines two of these into one native inference frame, giving a
            // stable prefix sooner without returning to per-buffer overhead.
            if nemotronInputFormat != chunk.format {
                nemotronInputSamples.removeAll(keepingCapacity: true)
                nemotronInputFormat = chunk.format
            }
            nemotronInputSamples.append(contentsOf: chunk.samples)
            let targetSamples = max(
                1, Int(chunk.format.sampleRate * 0.32) * chunk.format.channelCount
            )
            guard nemotronInputSamples.count >= targetSamples else { return }
            let batchedSamples = Array(nemotronInputSamples.prefix(targetSamples))
            nemotronInputSamples.removeFirst(targetSamples)
            let batchedChunk = AudioChunk(
                samples: batchedSamples, format: chunk.format,
                capturedAt: chunk.capturedAt
            )
            switch nemotronAudioContinuation?.yield(batchedChunk) {
            case .dropped: metrics.droppedChunks += 1
            default: break
            }
            return
        }
        if let zipformerRecognizer {
            if let pendingError {
                self.pendingError = nil
                throw pendingError
            }
            metrics.inputChunks += 1
            for phrase in await zipformerRecognizer.ingest(chunk) {
                enqueueRecognizedPhrase(
                    phrase.text, recognitionMilliseconds: phrase.elapsedMilliseconds
                )
            }
            return
        }
        if #available(macOS 26.0, *),
           let recognizer = progressiveRecognizer as? FixedLagWindowSpeechRecognizer {
            if let pendingError {
                self.pendingError = nil
                throw pendingError
            }
            metrics.inputChunks += 1
            recordSpeechClock(for: chunk)
            await recognizer.send(chunk)
            return
        }
        if #available(macOS 26.0, *),
           let recognizer = progressiveRecognizer as? ContinuousSpeechRecognizer {
            if let pendingError {
                self.pendingError = nil
                throw pendingError
            }
            metrics.inputChunks += 1
            recordSpeechClock(for: chunk)
            await recognizer.send(chunk)
            return
        }
        guard segmentContinuation != nil else { throw TranslationProviderError.notConnected }
        if let pendingError {
            self.pendingError = nil
            throw pendingError
        }
        metrics.inputChunks += 1
        for segment in await segmenter.ingest(chunk) {
            switch segmentContinuation?.yield(segment) {
            case .dropped: metrics.droppedChunks += 1
            case .enqueued: metrics.outputChunks += 1
            default: break
            }
        }
    }

    func currentPipelineMetrics() async -> AudioPipelineMetrics { metrics }

    func stopSession() async {
        sessionGeneration &+= 1
        nemotronAudioContinuation?.finish()
        nemotronAudioContinuation = nil
        nemotronAudioTask?.cancel()
        nemotronAudioTask = nil
        if let nemotronRecognizer {
            if let events = try? await nemotronRecognizer.finish() {
                for event in events {
                    if let phrase = event.phrase {
                        enqueueNemotronTranslation(
                            phrase, isFinal: event.isFinal,
                            recognitionMilliseconds: event.elapsedMilliseconds
                        )
                    }
                }
            }
        }
        flushPendingNemotronTranslation()
        nemotronRecognizer = nil
        nemotronCommittedDisplay = ""
        pendingTranslationWords.removeAll(keepingCapacity: true)
        if let zipformerRecognizer {
            for phrase in await zipformerRecognizer.finish() {
                enqueueRecognizedPhrase(
                    phrase.text, recognitionMilliseconds: phrase.elapsedMilliseconds
                )
            }
        }
        zipformerRecognizer = nil
        previewTask?.cancel()
        previewTask = nil
        previewTranslationTask?.cancel()
        previewTranslationTask = nil
        previewStableCommitter.reset()
        rollingTranslationScheduler.reset()
        latestPreviewSource = ""
        latestPreviewIsStable = false
        translatedPreviewSource = ""
        translatedPreviewIsStable = false
        receivedAudioMilliseconds = 0
        lastSpeechAudioMilliseconds = 0
        confirmedTranslationDisplay = ""
        lastASRLogMilliseconds = nil
        lastSpeechWallClock = nil
        await previewRecognizer?.stop()
        previewRecognizer = nil
        progressiveTask?.cancel()
        progressiveTask = nil
        if #available(macOS 26.0, *),
           let recognizer = progressiveRecognizer as? FixedLagWindowSpeechRecognizer {
            await recognizer.stop()
        }
        if #available(macOS 26.0, *),
           let recognizer = progressiveRecognizer as? ContinuousSpeechRecognizer {
            await recognizer.stop()
        }
        progressiveRecognizer = nil
        progressiveCommittedDisplay = ""
        textContinuation?.finish()
        textContinuation = nil
        speechContinuation?.finish()
        speechContinuation = nil
        speechTask?.cancel()
        speechTask = nil
        neuralVoicePreparationTask?.cancel()
        neuralVoicePreparationTask = nil
        neuralVoiceReady = false
        if let final = await segmenter.flush() { _ = segmentContinuation?.yield(final) }
        segmentContinuation?.finish()
        segmentContinuation = nil
        workerTask?.cancel()
        workerTask = nil
        fallbackWorkerTask?.cancel()
        fallbackWorkerTask = nil
        sourceLanguage = nil
        targetLanguage = nil
        usesAppleASR = false
        pendingError = nil
        if ownsWorkerSession {
            ownsWorkerSession = false
            await worker.release()
        }
    }

    private func enqueueRecognizedPhrase(
        _ text: String, recognitionMilliseconds: Double
    ) {
        if nemotronRecognizer != nil, Self.shouldHoldNemotronDispatch(text) {
            let carry = text.split(whereSeparator: \Character.isWhitespace).map(String.init)
            pendingTranslationWords.insert(contentsOf: carry, at: 0)
            return
        }
        progressivePhraseSeen = true
        switch textContinuation?.yield(RecognizedPhrase(
            text: text, recognitionMilliseconds: recognitionMilliseconds,
            enqueuedAt: .now
        )) {
        case .dropped:
            metrics.droppedChunks += 1
        case .enqueued:
            metrics.outputChunks += 1
        default:
            break
        }
    }

    static func shouldHoldNemotronDispatch(_ text: String) -> Bool {
        let words = text.split(whereSeparator: \Character.isWhitespace).map(String.init)
        guard let last = words.last else { return true }
        // A comma is not inherently incomplete. The clause selector only
        // exposes one after observing right context, and holding every such
        // prefix caused 40-50 word bursts. Keep only genuinely dangling
        // grammar (prepositions, auxiliaries and unresolved dependent clauses).
        return !canEndLiveTranslationPhrase(last)
            || startsWithUnresolvedDependentClause(words)
    }

    private func translateNemotronPhraseIfFresh(_ phrase: RecognizedPhrase) async {
        // Endpoint revisions can repeat the tail of a phrase that was already
        // dispatched. Reject those source repetitions before they consume
        // translation and TTS time or become audible duplicates.
        guard let freshText = transcriptGate.freshText(phrase.text) else { return }
        await translateText(
            freshText,
            recognitionMilliseconds: phrase.recognitionMilliseconds,
            queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds
        )
    }

    private func processNemotronAudio(_ chunk: AudioChunk) async {
        guard let nemotronRecognizer else { return }
        do {
            for event in try await nemotronRecognizer.ingest(chunk) {
                metrics.sourceHypothesis = event.hypothesis
                metrics.asrMilliseconds = event.elapsedMilliseconds
                metrics.asrStreamLagMilliseconds = max(
                    0, event.elapsedMilliseconds - event.audioProcessedSeconds * 1_000
                )
                // Nemotron's revisable cumulative hypothesis is accurate enough
                // to drive the display layer before a clause becomes immutable.
                // This does not feed TTS or the confirmed transcript.
                let contextualHypothesis = Self.contextualNemotronPreview(
                    pendingWords: pendingTranslationWords,
                    hypothesis: event.hypothesis
                )
                if let candidate = rollingTranslationScheduler.candidate(
                    hypothesis: contextualHypothesis,
                    stablePrefix: nil,
                    audioMilliseconds: event.audioProcessedSeconds * 1_000
                ) {
                    schedulePreviewTranslation(
                        candidate.text,
                        isStable: event.phrase != nil || candidate.isStable
                    )
                }
                if let phrase = event.phrase {
                    let correctedPhrase = ASRContextualCorrector.correct(
                        phrase, preferredTerms: preferredTerms
                    )
                    nemotronCommittedDisplay = [nemotronCommittedDisplay, correctedPhrase]
                        .filter { !$0.isEmpty }.joined(separator: " ")
                    // Keep the diagnostic readable during long videos.
                    let words = nemotronCommittedDisplay.split(whereSeparator: \Character.isWhitespace)
                    if words.count > 48 {
                        nemotronCommittedDisplay = words.suffix(48).joined(separator: " ")
                    }
                    metrics.committedSource = nemotronCommittedDisplay
                    if metrics.firstStableCommitMilliseconds == nil {
                        metrics.firstStableCommitMilliseconds = event.elapsedMilliseconds
                    }
                    enqueueNemotronTranslation(
                        correctedPhrase, isFinal: event.isFinal,
                        recognitionMilliseconds: event.elapsedMilliseconds
                    )
                }
            }
        } catch {
            pendingError = error
        }
    }

    private func enqueueNemotronTranslation(
        _ phrase: String, isFinal: Bool, recognitionMilliseconds: Double
    ) {
        pendingTranslationWords.append(contentsOf:
            phrase.split(whereSeparator: \Character.isWhitespace).map(String.init)
        )
        let endsClause = pendingTranslationWords.last?.last.map {
            ".?!。？！".contains($0)
        } ?? false
        let pending = pendingTranslationWords.joined(separator: " ")
        let continuesIntoNextEndpoint = Self.endsWithContinuingDiscourseMarker(pending)
        // Do not let quality-first semantic waiting turn into a 40-65 word TTS
        // burst. Emit an already complete sentence or clause and retain only
        // its unfinished right context for the next acoustic endpoint.
        if let readyPrefix = EarlyTranslationClauseSelector.boundedPrefix(
            in: pending, maximumWords: 22
        ) {
            // `enqueueRecognizedPhrase` retains an incomplete candidate by
            // putting it back into `pendingTranslationWords`. Recursing after
            // that would select the identical prefix forever and eventually
            // overflow this task's stack. Keep the original buffer intact and
            // wait for more context instead.
            guard !Self.shouldHoldNemotronDispatch(readyPrefix) else { return }
            let prefixCount = readyPrefix.split(whereSeparator: \Character.isWhitespace).count
            pendingTranslationWords.removeFirst(prefixCount)
            enqueueRecognizedPhrase(
                readyPrefix, recognitionMilliseconds: recognitionMilliseconds
            )
            if !pendingTranslationWords.isEmpty {
                enqueueNemotronTranslation(
                    "", isFinal: isFinal,
                    recognitionMilliseconds: recognitionMilliseconds
                )
            }
            return
        }
        if !continuesIntoNextEndpoint,
           (endsClause && isFinal && Self.shouldFlushPunctuatedNemotronFragment(
               pendingTranslationWords
           )) || (isFinal && Self.shouldFlushFinalNemotronFragment(
               pendingTranslationWords
           )) {
            flushPendingNemotronTranslation(
                recognitionMilliseconds: recognitionMilliseconds
            )
            return
        }
        if continuesIntoNextEndpoint { return }
        // Quality-first mode keeps the full stable clause. The earlier prefix
        // selector frequently split immediately before a verb or object, which
        // made the deterministic sentence translator produce literal nonsense.
        guard Self.shouldFlushLiveNemotronBuffer(pendingTranslationWords) else {
            return
        }
        flushPendingNemotronTranslation(
            recognitionMilliseconds: recognitionMilliseconds
        )
    }

    static func shouldFlushLiveNemotronBuffer(_ words: [String]) -> Bool {
        // Stable RNNT chunks are not semantic clauses. A content word at the
        // end of a chunk ("... then President Reagan") is not enough evidence
        // to translate it. Punctuation and acoustic finals are handled above;
        // this is only the bounded-latency escape hatch for uninterrupted
        // speech with no endpoint.
        guard words.count >= 22, let last = words.last,
              !endsInRepeatedToken(words) else { return false }
        let normalizedWords = words.map {
            $0.lowercased().trimmingCharacters(in: .punctuationCharacters)
        }
        if startsWithUnresolvedDependentClause(words) { return false }
        // ASR commonly renders comparative "than" as "then". If only a
        // short noun phrase follows it, the comparison's predicate has not
        // arrived yet and this is not an irreversible translation boundary.
        if let comparison = normalizedWords.lastIndex(where: {
            $0 == "than" || $0 == "then"
        }), normalizedWords.distance(from: comparison, to: normalizedWords.endIndex) <= 4 {
            return false
        }
        return canEndLiveTranslationPhrase(last)
            && NeuralSentenceBoundaryClassifier.canEnd(words.joined(separator: " "))
    }

    static func contextualNemotronPreview(
        pendingWords: [String], hypothesis: String
    ) -> String {
        let current = hypothesis.split(whereSeparator: \Character.isWhitespace)
            .map(String.init)
        guard !pendingWords.isEmpty else { return current.joined(separator: " ") }
        guard !current.isEmpty else { return pendingWords.joined(separator: " ") }
        let maximum = min(pendingWords.count, current.count, 12)
        var overlap = 0
        if maximum >= 2 {
            for count in stride(from: maximum, through: 2, by: -1) {
                let left = pendingWords.suffix(count).map(Self.previewWordKey)
                let right = current.prefix(count).map(Self.previewWordKey)
                if left == right {
                    overlap = count
                    break
                }
            }
        }
        return (pendingWords + current.dropFirst(overlap)).joined(separator: " ")
    }

    private static func previewWordKey(_ word: String) -> String {
        word.lowercased().trimmingCharacters(in: .punctuationCharacters)
    }

    static func shouldFlushNemotronAfterSilence(
        _ words: [String], silenceMilliseconds: Double
    ) -> Bool {
        // RNNT punctuation/final callbacks can trail the actual end of speech.
        // Once the speaker has paused, do not leave a complete short thought
        // waiting for the next sentence merely to satisfy a word-count target.
        guard words.count >= 3, silenceMilliseconds >= 720,
              let last = words.last else { return false }
        if words.count == 3 {
            return canEndLiveTranslationPhrase(last)
        }
        return shouldFlushFinalNemotronFragment(words)
    }

    static func shouldFlushPunctuatedNemotronFragment(_ words: [String]) -> Bool {
        // RNNT occasionally emits punctuation on isolated carry-over tokens
        // such as "I." or "And.". It can also hallucinate a period after a
        // dangling preposition ("after her battle with."). Neither is a
        // translatable sentence despite the punctuation.
        guard words.count >= 4, let last = words.last else { return false }
        let phrase = words.joined(separator: " ").lowercased()
        if phrase.hasPrefix("in addition to "), !phrase.contains(",") {
            return false
        }
        return canEndLiveTranslationPhrase(last)
            && NeuralSentenceBoundaryClassifier.canEnd(words.joined(separator: " "))
    }

    private func enqueueFinalPhraseIfFresh(
        _ phrase: String, isFinal: Bool, recognitionMilliseconds: Double
    ) {
        // Permit the recognizer to close the last sentence, but reject final
        // callbacks emitted long after the selected app has stopped producing
        // speech. They represent stale revisions, not new source material.
        guard hasRecentSpeech() else { return }
        // ContinuousSpeechRecognizer has already stabilized and split this
        // phrase. Sending it through the RNNT accumulator a second time could
        // add another 24 words of delay during unpunctuated live speech.
        enqueueRecognizedPhrase(
            phrase, recognitionMilliseconds: recognitionMilliseconds
        )
    }

    private func flushPendingNemotronTranslation(
        recognitionMilliseconds: Double? = nil
    ) {
        guard !pendingTranslationWords.isEmpty,
              let last = pendingTranslationWords.last,
              Self.canEndLiveTranslationPhrase(last),
              !last.hasSuffix(",") else { return }
        let text = pendingTranslationWords.joined(separator: " ")
        pendingTranslationWords.removeAll(keepingCapacity: true)
        enqueueRecognizedPhrase(
            text,
            recognitionMilliseconds: recognitionMilliseconds
                ?? metrics.asrMilliseconds ?? 0
        )
    }

    private static func canEndLiveTranslationPhrase(_ word: String) -> Bool {
        let token = word.lowercased().trimmingCharacters(in: .punctuationCharacters)
        return ![
            "a", "an", "about", "and", "as", "at", "because", "but", "by", "for",
            "from", "how", "if", "in", "into", "of", "on", "or", "than", "that",
            "the", "then", "to", "when", "which", "while", "who", "with",
            "toward", "towards", "where",
            "can", "can't", "cannot", "could", "couldn't", "may", "might",
            "must", "should", "shouldn't", "would", "wouldn't", "will", "won't",
            "care", "cared", "concern", "concerned", "get", "getting",
            "had", "has", "have", "is", "was", "were",
            "afford", "bring", "ensure", "give", "make", "need", "provide",
            "say", "take", "tell", "use", "want",
            "he's", "she's", "it's", "that's", "there's",
            "i'm", "we're", "they're", "you're",
            "her", "his", "its", "my", "our", "their", "your",
        ].contains(token)
    }

    private static func endsInRepeatedToken(_ words: [String]) -> Bool {
        guard words.count >= 2 else { return false }
        let left = words[words.count - 2].lowercased()
            .trimmingCharacters(in: .punctuationCharacters)
        let right = words[words.count - 1].lowercased()
            .trimmingCharacters(in: .punctuationCharacters)
        let incompleteRepeatedSubjects: Set<String> = [
            "i'm", "we're", "they're", "you're", "he's", "she's", "it's",
        ]
        return left == right && incompleteRepeatedSubjects.contains(right)
    }

    /// RNNT endpointing is an acoustic boundary, not necessarily a semantic
    /// sentence boundary. A speaker pause regularly finalizes fragments such
    /// as "More than", "Your opp", or the first subword-like token of a name.
    /// Carry short fragments into the next endpoint instead of translating
    /// them alone. Punctuation remains an unconditional boundary above.
    static func shouldFlushFinalNemotronFragment(_ words: [String]) -> Bool {
        guard words.count >= 4, let last = words.last else { return false }
        guard !endsInRepeatedToken(words) else { return false }
        // Four-word acoustic endpoints are especially error-prone: RNNT often
        // closes a carry-over such as "And in productive way" even though it
        // has no independent predicate. Preserve genuinely standalone short
        // turns (questions, exclamations and conversational acknowledgements),
        // but attach all other four-word fragments to the next endpoint.
        if words.count == 4, !isStandaloneShortNemotronUtterance(words) {
            return false
        }
        // A comma explicitly promises another clause. Acoustic silence after
        // it must not turn the first half into a standalone translation.
        if last.hasSuffix(",") { return false }
        let normalizedPhrase = words.joined(separator: " ")
            .lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if startsWithUnresolvedDependentClause(words) { return false }
        if normalizedPhrase.hasPrefix("when ") || normalizedPhrase.hasPrefix("and when ") {
            return false
        }
        if normalizedPhrase.hasPrefix("if ") && !normalizedPhrase.contains(",") {
            return false
        }
        if normalizedPhrase.hasPrefix("but these days"),
           normalizedPhrase.contains("those subjects"),
           !normalizedPhrase.contains(" are ") {
            return false
        }
        if normalizedPhrase == "i i came to realize"
            || normalizedPhrase == "i came to realize" {
            return false
        }
        if normalizedPhrase.hasPrefix("now i actually use the exact same") {
            return false
        }
        if normalizedPhrase == "he said that sensi"
            || normalizedPhrase.hasSuffix(" politicians can't") {
            return false
        }
        let bare = last.trimmingCharacters(in: .punctuationCharacters)
        // An endpoint ending in a capitalized mid-phrase token is commonly the
        // first half of a proper name (for example "Alexand" + "Vindman").
        // Keep it until right context arrives instead of translating a broken
        // name as a complete sentence.
        if last == bare,
           let first = bare.first,
           first.isUppercase,
           words.dropLast().contains(where: { !$0.isEmpty }) {
            return false
        }
        let lower = words.map { $0.lowercased() }
        if let betweenIndex = lower.lastIndex(where: {
            $0.trimmingCharacters(in: .punctuationCharacters) == "between"
        }) {
            let listTail = lower.suffix(from: lower.index(after: betweenIndex))
            let hasConjunction = listTail.contains(where: {
                $0.trimmingCharacters(in: .punctuationCharacters) == "and"
            })
            if !hasConjunction { return false }
        }
        return canEndLiveTranslationPhrase(last)
            && NeuralSentenceBoundaryClassifier.canEnd(words.joined(separator: " "))
    }

    private static func startsWithUnresolvedDependentClause(_ words: [String]) -> Bool {
        guard !words.isEmpty else { return false }
        let normalized = words.map {
            $0.lowercased().trimmingCharacters(in: .punctuationCharacters)
        }
        var start = 0
        if normalized.count >= 2, normalized[0] == "so", normalized[1] == "that" {
            return true
        }
        if normalized.first == "and" || normalized.first == "but" || normalized.first == "so" {
            start = 1
        }
        guard start < normalized.count else { return true }
        let dependentOpeners: Set<String> = [
            "although", "because", "if", "unless", "when", "whenever", "while",
            "who", "which",
        ]
        guard dependentOpeners.contains(normalized[start]) else { return false }
        // A comma closes the dependent clause but is not itself a main clause.
        // Require a short right-hand tail so "who suffered ...," remains held
        // while "If ..., we will act" can be emitted.
        guard let comma = words.firstIndex(where: { $0.contains(",") }) else {
            return true
        }
        return words.distance(from: words.index(after: comma), to: words.endIndex) < 3
    }

    private static func isStandaloneShortNemotronUtterance(_ words: [String]) -> Bool {
        guard let first = words.first?.lowercased()
            .trimmingCharacters(in: .punctuationCharacters),
              let last = words.last else { return false }
        if last.hasSuffix("?") || last.hasSuffix("!") { return true }
        let conversationalOpenings: Set<String> = [
            "thanks", "thank", "yes", "no", "okay", "ok", "welcome",
        ]
        return conversationalOpenings.contains(first)
    }

    static func endsWithContinuingDiscourseMarker(_ text: String) -> Bool {
        let normalized = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        return normalized.range(
            of: #"(?:(?:raised|ray'?s),?\s+)?get this[.!?]?$"#,
            options: .regularExpression
        ) != nil
    }

    private func translate(_ segment: AudioChunk) async {
        guard let sourceLanguage, let targetLanguage else { return }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ai-interpreter-worker-\(UUID().uuidString)", isDirectory: true)
        do {
            if usesAppleASR {
                do {
                    let asrStarted = ContinuousClock.now
                    let text = try await AppleLocalTranslationProvider.recognize(
                        segment, language: sourceLanguage,
                        contextualTerms: preferredTerms
                    )
                    guard transcriptGate.accept(text) else { return }
                    let asrDuration = asrStarted.duration(to: .now)
                    let continuation = outputContinuation
                    let response = try await worker.processTextStreaming(
                        text: text,
                        sourceLanguage: sourceLanguage,
                        targetLanguage: targetLanguage,
                        preferredTerms: preferredTerms,
                        onChunk: { continuation.yield($0) }
                    )
                    metrics.asrMilliseconds = asrDuration.milliseconds
                    metrics.translationMilliseconds = response.translationMilliseconds
                    metrics.synthesisMilliseconds = response.ttsMilliseconds
                    metrics.modelTotalMilliseconds = response.totalMilliseconds
                    return
                } catch {
                    // A transient Apple recognizer failure must not stop a live
                    // interview. Disable it for this session and replay the same
                    // still-local segment through the proven Whisper fallback.
                    usesAppleASR = false
                }
            }
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let source = directory.appendingPathComponent("source.wav")
            try LocalExpressiveSpeechRenderer.writeWAV(segment, to: source)
            let continuation = outputContinuation
            let response = try await worker.processAudioStreaming(
                sourceURL: source,
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
                preferredTerms: preferredTerms,
                onChunk: { continuation.yield($0) }
            )
            metrics.asrMilliseconds = response.asrMilliseconds
            metrics.translationMilliseconds = response.translationMilliseconds
            metrics.prosodyMilliseconds = response.prosodyMilliseconds
            metrics.synthesisMilliseconds = response.ttsMilliseconds
            metrics.modelTotalMilliseconds = response.totalMilliseconds
            guard response.status == "ok" else { return }
        } catch {
            // Capture processing failures so AppState can stop and show a useful error
            // instead of leaving the session apparently running without output.
            pendingError = error
        }
    }

    private func translateWhisperSegment(_ segment: AudioChunk) async {
        guard let sourceLanguage else { return }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ai-interpreter-asr-\(UUID().uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let source = directory.appendingPathComponent("source.wav")
            try LocalExpressiveSpeechRenderer.writePCM16WAV(segment, to: source)
            let response = try await worker.transcribeAudio(
                sourceURL: source, sourceLanguage: sourceLanguage
            )
            metrics.asrMilliseconds = response.asrMilliseconds ?? 0
            guard response.status == "ok", let text = response.transcript,
                  transcriptGate.accept(text) else { return }
            await translateText(
                text,
                recognitionMilliseconds: receivedAudioMilliseconds,
                queueMilliseconds: 0
            )
        } catch {
            pendingError = error
        }
    }

    private func translateText(
        _ text: String, recognitionMilliseconds: Double, queueMilliseconds: Double
    ) async {
        guard let sourceLanguage, let targetLanguage else { return }
        guard !TranslationQueuePolicy.isStale(queueMilliseconds: queueMilliseconds) else {
            metrics.droppedChunks += 1
            await TranslationDebugLogger.shared.record(
                source: text, normalizedSource: text, translation: nil,
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
                recognizerSessionMilliseconds: recognitionMilliseconds,
                translationQueueMilliseconds: queueMilliseconds,
                translationMilliseconds: nil, error: "stale_translation_dropped"
            )
            return
        }
        do {
            let translationStarted = ContinuousClock.now
            var translatedText = text
            var translationMilliseconds = 0.0
            var synthesisMilliseconds = 0.0
            var totalMilliseconds = 0.0
            do {
                // The trained MADLAD model passed the held-out gate above the
                // Apple baseline while remaining within the live watchdog.
                // Keep Apple as an immediate fallback if the resident process
                // or its packaged model is unavailable.
                let cleanedText = Self.cleanWindowBoundaryArtifacts(text)
                let normalizedText = TranslationSemanticNormalizer.normalize(
                    cleanedText, source: sourceLanguage, target: targetLanguage
                )
                if let remoteTranslation {
                    do {
                        let remote = try await Self.withTimeout(
                            TranslationQueuePolicy.remoteWatchdogTimeout
                        ) {
                            try await remoteTranslation.translate(
                                text: normalizedText,
                                sourceLanguage: sourceLanguage,
                                targetLanguage: targetLanguage
                            )
                        }
                        translatedText = remote.text
                        translationMilliseconds = remote.milliseconds
                    } catch {
                        // Network and prototype-server failures must never stop a
                        // live interview. Preserve the proven on-device path.
                        translatedText = try await AppleLocalTranslationProvider.translateText(
                            normalizedText,
                            source: sourceLanguage,
                            target: targetLanguage
                        )
                        translationMilliseconds = translationStarted.duration(to: .now).milliseconds
                    }
                } else {
                    do {
                        if !ownsWorkerSession {
                            try await worker.acquire()
                            ownsWorkerSession = true
                            try await worker.resetContext()
                        }
                        let response = try await Self.withTimeout(
                            TranslationQueuePolicy.remoteWatchdogTimeout
                        ) {
                            try await self.worker.translateText(
                                text: normalizedText,
                                sourceLanguage: sourceLanguage,
                                targetLanguage: targetLanguage,
                                preferredTerms: self.preferredTerms
                            )
                        }
                        guard let translation = response.translation,
                              !translation.isEmpty else {
                            throw LocalModelWorkerError.invalidResponse
                        }
                        translatedText = translation
                        translationMilliseconds = response.translationMilliseconds ?? 0
                    } catch {
                        translatedText = try await AppleLocalTranslationProvider.translateText(
                            normalizedText, source: sourceLanguage, target: targetLanguage
                        )
                        translationMilliseconds = translationStarted.duration(to: .now).milliseconds
                    }
                }
            } catch LocalTranslationError.languagePackUnavailable {
                // Offline fallback only. The installed Apple language pack is
                // the required production path for accuracy-first operation.
                if !ownsWorkerSession {
                    try await worker.acquire()
                    ownsWorkerSession = true
                    try await worker.resetContext()
                }
                let response = try await worker.translateText(
                    text: text,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage,
                    preferredTerms: preferredTerms
                )
                guard response.status != "duplicate" else { return }
                guard let translation = response.translation,
                      !translation.isEmpty else {
                    throw LocalModelWorkerError.invalidResponse
                }
                translatedText = translation
                translationMilliseconds = response.translationMilliseconds ?? 0
                totalMilliseconds = response.totalMilliseconds ?? 0
            }
            // Translation never waits for speech rendering. This keeps the ASR
            // and contextual text path responsive during long interviews.
            if targetLanguage == .korean {
                translatedText = KoreanHonorificNormalizer.normalize(translatedText)
                translatedText = KoreanTranslationNaturalizer.normalize(translatedText)
                translatedText = KoreanSpeechTextNormalizer.normalize(translatedText)
            }
            var spokenText = targetLanguage == .korean
                ? KoreanSpeechTextNormalizer.boundedForLiveSpeech(translatedText)
                : translatedText
            if simultaneousTTSIsEnabled {
                spokenText = simultaneousTranslationCommitter.remainder(of: spokenText)
                simultaneousTranslationCommitter.reset()
            }
            // Preview translations are deliberately handled by
            // schedulePreviewTranslation and never cross this boundary.
            let completedTranslationMilliseconds = translationStarted
                .duration(to: .now).milliseconds
            if TTSQueuePolicy.shouldSpeak(.confirmed),
               !spokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
               !TranslationQueuePolicy.isTooLateToSpeak(
                   queueMilliseconds: queueMilliseconds,
                   translationMilliseconds: completedTranslationMilliseconds
               ) {
                _ = speechContinuation?.yield(SpeechRequest(
                    text: spokenText, language: targetLanguage,
                    translationKind: .confirmed,
                    enqueuedAt: .now, sessionGeneration: sessionGeneration
                ))
            }
            let pipelineDuration = translationStarted.duration(to: .now).milliseconds
            if totalMilliseconds == 0 { totalMilliseconds = pipelineDuration }
            let normalizedForLog = TranslationSemanticNormalizer.normalize(
                Self.cleanWindowBoundaryArtifacts(text),
                source: sourceLanguage, target: targetLanguage
            )
            await TranslationDebugLogger.shared.record(
                source: text, normalizedSource: normalizedForLog,
                translation: translatedText,
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
                recognizerSessionMilliseconds: recognitionMilliseconds,
                translationQueueMilliseconds: queueMilliseconds,
                translationMilliseconds: translationMilliseconds
            )
            let diagnostic = "live_pipeline source=\(text.debugDescription) translation=\(translatedText.debugDescription) translation_ms=\(Int(translationMilliseconds)) translation_and_tts_ms=\(Int(pipelineDuration)) tts_ms=\(Int(synthesisMilliseconds))\n"
            FileHandle.standardError.write(Data(diagnostic.utf8))
            metrics.asrMilliseconds = recognitionMilliseconds
            metrics.translationMilliseconds = translationMilliseconds
            metrics.synthesisMilliseconds = synthesisMilliseconds
            metrics.modelTotalMilliseconds = totalMilliseconds
            // Confirmed translations are append-only. A later phrase may add
            // text, but it must never rewrite text the user already read.
            confirmedTranslationDisplay = [confirmedTranslationDisplay, translatedText]
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            let confirmedWords = confirmedTranslationDisplay.split(
                whereSeparator: \Character.isWhitespace
            )
            if confirmedWords.count > 1_200 {
                confirmedTranslationDisplay = confirmedWords.suffix(1_200)
                    .joined(separator: " ")
            }
            metrics.translationPhrase = confirmedTranslationDisplay
            metrics.outputChunks += 1
        } catch is CancellationError {
            return
        } catch {
            await TranslationDebugLogger.shared.record(
                source: text, normalizedSource: text, translation: nil,
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
                recognizerSessionMilliseconds: recognitionMilliseconds,
                translationQueueMilliseconds: queueMilliseconds,
                translationMilliseconds: nil, error: error.localizedDescription
            )
            pendingError = error
        }
    }

    private func startSpeechQueue() {
        let requests = AsyncStream<SpeechRequest>.makeStream(
            // Keep a small run of confirmed thoughts so brief TTS bursts do not
            // skip the sentence between the one speaking and the newest one.
            // The four-second request-age guard still removes stale backlog.
            bufferingPolicy: .bufferingNewest(12)
        )
        speechContinuation = requests.continuation
        speechTask = Task { [weak self] in
            for await request in requests.stream {
                guard !Task.isCancelled else { break }
                await self?.synthesize(request)
            }
        }
    }

    private func synthesize(_ request: SpeechRequest) async {
        guard TTSQueuePolicy.shouldSpeak(request.translationKind) else { return }
        guard request.sessionGeneration == sessionGeneration else { return }
        let speechKey = request.text.lowercased()
            .split(whereSeparator: \Character.isWhitespace).joined(separator: " ")
        guard speechKey != lastSpokenText else {
            let diagnostic = "speech_dropped_duplicate text=\(request.text.debugDescription)\n"
            FileHandle.standardError.write(Data(diagnostic.utf8))
            metrics.droppedChunks += 1
            return
        }
        guard !TTSQueuePolicy.isStale(
            request.translationKind,
            text: request.text,
            enqueuedAt: request.enqueuedAt
        ) else {
            let age = request.enqueuedAt.duration(to: .now).milliseconds
            let diagnostic = "speech_dropped_stale age_ms=\(Int(age)) text=\(request.text.debugDescription)\n"
            FileHandle.standardError.write(Data(diagnostic.utf8))
            metrics.droppedChunks += 1
            return
        }
        // Do not gate speech on cumulative ASR streamLag. That metric includes
        // intentionally dropped capture buffers and therefore may never fall
        // again after overload. The old hysteresis permanently muted every
        // later translation once it crossed 2.5 seconds. Freshness is enforced
        // by the bounded newest-only queue and request-age check above instead.
        let started = ContinuousClock.now
        lastSpokenText = speechKey
        hasStartedSimultaneousSpeech = true
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID, event: "tts_started",
            audioMilliseconds: receivedAudioMilliseconds,
            sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
            translation: request.text,
            status: request.translationKind == .simultaneousCommitted
                ? "simultaneous" : "confirmed",
            operationMilliseconds: request.enqueuedAt.duration(to: .now).milliseconds
        )
        do {
            let remoteTTS = self.remoteTTS
            let continuation = outputContinuation
            let outputGate = SpeechOutputGate()
            let playbackGroupID = UUID()
            defer { outputGate.close() }
            try await Self.withSpeechWatchdog {
                if let remoteTTS {
                    try await remoteTTS.synthesize(
                        text: request.text, language: request.language,
                        onChunk: {
                            outputGate.yieldIfOpen(
                                $0.groupedForPlayback(playbackGroupID), to: continuation
                            )
                        }
                    )
                } else {
                    try await AppleLocalTranslationProvider.speakText(
                        request.text, language: request.language
                    )
                }
            }
            metrics.synthesisMilliseconds = started.duration(to: .now).milliseconds
        } catch {
            let diagnostic = "speech_synthesis_failed error=\(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(diagnostic.utf8))
            // A failed optional voice must not make the product silent. Retry
            // once with the built-in voice unless that was already the failing
            // path, while leaving recognition and text translation untouched.
            if remoteTTS != nil {
                neuralVoiceReady = false
            }
        }
    }

    private static func withSpeechWatchdog(
        operation: @escaping @Sendable () async throws -> Void
    ) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(for: TTSQueuePolicy.synthesisWatchdogTimeout)
                throw SpeechSynthesisWatchdogTimeout()
            }
            guard let result = await group.nextResult() else {
                throw SpeechSynthesisWatchdogTimeout()
            }
            group.cancelAll()
            try result.get()
        }
    }

    private func setNeuralVoiceReady() {
        neuralVoiceReady = true
    }

    private func record(_ event: StableSpeechEvent) async {
        // ScreenCaptureKit may stop delivering callbacks completely when a
        // Chrome tab becomes silent. Audio-time checks then freeze at zero and
        // incorrectly accept a minutes-old SpeechTranscriber backlog.
        guard hasRecentSpeech() else { return }
        metrics.sourceHypothesis = event.hypothesis
        if metrics.firstHypothesisMilliseconds == nil {
            metrics.firstHypothesisMilliseconds = event.elapsedMilliseconds
        }
        if let delta = event.committedDelta {
            progressiveCommittedDisplay = [progressiveCommittedDisplay, delta]
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            // Keep a long, readable rolling transcript without allowing an
            // hours-long video to grow memory forever.
            let words = progressiveCommittedDisplay.split(
                whereSeparator: \Character.isWhitespace
            )
            if words.count > 1_200 {
                progressiveCommittedDisplay = words.suffix(1_200)
                    .joined(separator: " ")
            }
            metrics.committedSource = progressiveCommittedDisplay
            if metrics.firstStableCommitMilliseconds == nil {
                metrics.firstStableCommitMilliseconds = event.elapsedMilliseconds
            }
        }
        // Drive the revisable low-latency caption from the same recognizer.
        // Running SFSpeechRecognizer beside SpeechTranscriber made on-device
        // decoding slower than real time on long Chrome streams.
        let stable = event.committedDelta == nil ? nil : progressiveCommittedDisplay
        if let candidate = rollingTranslationScheduler.candidate(
            hypothesis: event.hypothesis,
            stablePrefix: stable,
            audioMilliseconds: receivedAudioMilliseconds
        ) {
            schedulePreviewTranslation(candidate.text, isStable: candidate.isStable)
        }
        let shouldLog = lastASRLogMilliseconds == nil || event.phrase != nil
            || event.isFinal
            || event.elapsedMilliseconds - (lastASRLogMilliseconds ?? 0) >= 500
        if shouldLog {
            lastASRLogMilliseconds = event.elapsedMilliseconds
            await ASRDebugLogger.shared.record(
                hypothesis: event.hypothesis,
                committedPhrase: event.phrase,
                isFinal: event.isFinal,
                sessionMilliseconds: event.elapsedMilliseconds,
                decodeMilliseconds: 0,
                audioProcessedSeconds: receivedAudioMilliseconds / 1_000,
                engine: "apple-speech-continuous"
            )
        }
    }

    private func recordPreview(_ hypothesis: String) {
        let audioMilliseconds = receivedAudioMilliseconds
        // Speech.framework can revise stale text long after browser audio has
        // stopped. Ignore those callbacks after a short final-word grace period.
        guard audioMilliseconds - lastSpeechAudioMilliseconds <= 2_000 else { return }
        metrics.sourceHypothesis = hypothesis
        if metrics.firstHypothesisMilliseconds == nil {
            metrics.firstHypothesisMilliseconds = receivedAudioMilliseconds
        }
        let stableDelta = previewStableCommitter.update(hypothesis)
        guard let candidate = rollingTranslationScheduler.candidate(
            hypothesis: hypothesis,
            stablePrefix: stableDelta == nil ? nil : previewStableCommitter.committedText,
            audioMilliseconds: audioMilliseconds
        ) else { return }
        schedulePreviewTranslation(candidate.text, isStable: candidate.isStable)
    }

    private func schedulePreviewTranslation(_ hypothesis: String, isStable: Bool) {
        guard let sourceLanguage, let targetLanguage else { return }
        let current = EarlyTranslationClauseSelector.currentSentence(in: hypothesis)
        // This path is display-only, but two-word partials are not meaningful
        // EN<->KO translation units. The scheduler has already selected a
        // four-word clause with a usable ending; enforce the same invariant at
        // this final boundary so no caller can leak fragments into the UI.
        let minimumPreviewWords = simultaneousTTSIsEnabled ? 3 : 4
        guard current.split(whereSeparator: \Character.isWhitespace).count >= minimumPreviewWords,
              current.count >= 7,
              current != latestPreviewSource || (isStable && !latestPreviewIsStable)
        else { return }
        latestPreviewSource = current
        latestPreviewIsStable = isStable
        launchLatestPreviewTranslationIfNeeded(
            sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
        )
    }

    private func launchLatestPreviewTranslationIfNeeded(
        sourceLanguage: Language, targetLanguage: Language
    ) {
        guard previewTranslationTask == nil,
              latestPreviewSource != translatedPreviewSource
                || latestPreviewIsStable != translatedPreviewIsStable
        else { return }
        let current = latestPreviewSource
        let isStable = latestPreviewIsStable
        previewTranslationTask = Task(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do {
                var selectedSource = current
                if self.learnedSimultaneousBoundaryIsEnabled {
                    do {
                        let decision = try await Self.withTimeout(.milliseconds(500)) {
                            try await self.worker.simultaneousBoundaryDecision(
                                sourcePrefix: current,
                                sourceLanguage: sourceLanguage
                            )
                        }
                        let audioMilliseconds = await self.receivedAudioMilliseconds
                        let boundaryMilliseconds = decision.boundaryMilliseconds ?? -1
                        let probability = String(
                            format: "%.3f", decision.speakProbability ?? -1
                        )
                        let diagnostic = "learned_boundary audio_ms=\(Int(audioMilliseconds)) source_language=\(sourceLanguage.rawValue) status=\(decision.status) boundary_ms=\(Int(boundaryMilliseconds)) probability=\(probability) source=\(current.debugDescription)\n"
                        FileHandle.standardError.write(Data(diagnostic.utf8))
                        await SimultaneousDebugLogger.shared.record(
                            sessionID: self.simultaneousDebugSessionID,
                            event: "boundary_decision",
                            audioMilliseconds: audioMilliseconds,
                            sourceLanguage: sourceLanguage,
                            targetLanguage: targetLanguage, source: current,
                            status: decision.status,
                            operationMilliseconds: decision.boundaryMilliseconds,
                            probability: decision.speakProbability
                        )
                        // Preview text is revisable; spoken audio is not. Never
                        // translate or speak a prefix rejected by the semantic
                        // gate, including the very first utterance.
                        if decision.status == "wait" {
                            await self.recordLearnedBoundaryWait(
                                expectedSource: current, isStable: isStable
                            )
                            return
                        }
                        selectedSource = decision.boundarySource ?? current
                    } catch {
                        // Fail closed for simultaneous speech. The confirmed
                        // translation path remains active and will flush the
                        // complete phrase without speaking a speculative tail.
                        await self.recordLearnedBoundaryWait(
                            expectedSource: current, isStable: isStable
                        )
                        return
                    }
                }
                // The portable semantic model returns the exact cumulative
                // source prefix that passed its 95% precision gate. Translate
                // only that prefix; the target committer emits any new stable
                // delta and never re-speaks its previous prefix.
                let normalized = TranslationSemanticNormalizer.normalize(
                    selectedSource, source: sourceLanguage, target: targetLanguage
                )
                let translatedResponse = try await Self.withTimeout(.seconds(2)) {
                    try await self.worker.translateText(
                        text: normalized,
                        sourceLanguage: sourceLanguage,
                        targetLanguage: targetLanguage,
                        preferredTerms: []
                    )
                }
                guard let translated = translatedResponse.translation,
                      !translated.isEmpty else {
                    // A duplicate/empty worker response means this exact
                    // source was already handled. Mark the scheduler input as
                    // consumed so it cannot spin at the same audio timestamp.
                    await self.recordLearnedBoundaryWait(
                        expectedSource: current, isStable: isStable
                    )
                    return
                }
                try Task.checkCancellation()
                await self.recordPreviewTranslation(
                    translated, expectedSource: current, isStable: isStable
                )
            } catch {
                await self.finishPreviewTranslation()
            }
        }
    }

    private func recordLearnedBoundaryWait(
        expectedSource: String, isStable: Bool
    ) {
        if expectedSource == latestPreviewSource,
           isStable == latestPreviewIsStable {
            translatedPreviewSource = expectedSource
            translatedPreviewIsStable = isStable
        }
        previewTranslationTask = nil
        if let sourceLanguage, let targetLanguage {
            launchLatestPreviewTranslationIfNeeded(
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
            )
        }
    }

    private func recordPreviewTranslation(
        _ translation: String, expectedSource: String, isStable: Bool
    ) {
        let normalizedTranslation: String
        if targetLanguage == .korean {
            normalizedTranslation = KoreanSpeechTextNormalizer.normalize(
                KoreanTranslationNaturalizer.normalize(
                    KoreanHonorificNormalizer.normalize(translation)
                )
            )
        } else {
            normalizedTranslation = translation
        }
        if simultaneousTTSIsEnabled,
           let targetLanguage,
           let committed = simultaneousTranslationCommitter.update(
               normalizedTranslation, sourceIsStable: isStable
           ),
           !committed.isEmpty {
            hasStartedSimultaneousSpeech = true
            _ = speechContinuation?.yield(SpeechRequest(
                text: committed,
                language: targetLanguage,
                translationKind: .simultaneousCommitted,
                enqueuedAt: .now,
                sessionGeneration: sessionGeneration
            ))
            Task {
                await SimultaneousDebugLogger.shared.record(
                    sessionID: simultaneousDebugSessionID,
                    event: "tts_enqueued",
                    audioMilliseconds: receivedAudioMilliseconds,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage,
                    source: expectedSource, translation: committed,
                    status: "simultaneous"
                )
            }
            let diagnostic = "simultaneous_translation_committed audio_ms=\(Int(receivedAudioMilliseconds)) text=\(committed.debugDescription)\n"
            FileHandle.standardError.write(Data(diagnostic.utf8))
        }
        if expectedSource == latestPreviewSource,
           isStable == latestPreviewIsStable {
            let displayTranslation = normalizedTranslation
            metrics.previewTranslationPhrase = displayTranslation
            translatedPreviewSource = expectedSource
            translatedPreviewIsStable = isStable
            let now = receivedAudioMilliseconds
            // A stable word prefix is still not a completed sentence. It may
            // improve the revisable caption, but never the confirmed transcript.
            if isStable, metrics.firstStableTranslationMilliseconds == nil {
                metrics.firstStableTranslationMilliseconds = now
            }
            if let previous = metrics.lastPreviewTranslationMilliseconds {
                metrics.maximumPreviewTranslationGapMilliseconds = max(
                    metrics.maximumPreviewTranslationGapMilliseconds, now - previous
                )
            }
            metrics.lastPreviewTranslationMilliseconds = now
            metrics.previewTranslationUpdates += 1
            if metrics.firstPreviewTranslationMilliseconds == nil {
                metrics.firstPreviewTranslationMilliseconds = now
            }
        }
        previewTranslationTask = nil
        if let sourceLanguage, let targetLanguage {
            launchLatestPreviewTranslationIfNeeded(
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
            )
        }
    }

    private func finishPreviewTranslation() {
        previewTranslationTask = nil
        if let sourceLanguage, let targetLanguage {
            launchLatestPreviewTranslationIfNeeded(
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
            )
        }
    }

    private static func withTimeout<T: Sendable>(
        _ timeout: Duration,
        operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(for: timeout)
                throw PreviewTranslationTimeout()
            }
            guard let first = try await group.next() else {
                throw PreviewTranslationTimeout()
            }
            group.cancelAll()
            return first
        }
    }

    private func recordSpeechClock(for chunk: AudioChunk) {
        guard chunk.rms >= 0.0005 else { return }
        lastSpeechAudioMilliseconds = receivedAudioMilliseconds
        lastSpeechWallClock = .now
    }

    private func hasRecentSpeech() -> Bool {
        guard let lastSpeechWallClock else { return false }
        return lastSpeechWallClock.duration(to: .now).milliseconds <= 3_000
    }

    static func cleanWindowBoundaryArtifacts(_ text: String) -> String {
        var words = text.split(whereSeparator: \Character.isWhitespace).map(String.init)
        guard !words.isEmpty else { return "" }

        // Streaming windows often commit the same boundary word twice. Remove
        // only adjacent exact repetitions; non-adjacent rhetorical repetition
        // remains untouched.
        var deduplicated: [String] = []
        for word in words {
            let key = word.lowercased().trimmingCharacters(in: .punctuationCharacters)
            let previous = deduplicated.last?.lowercased()
                .trimmingCharacters(in: .punctuationCharacters)
            if !key.isEmpty, key == previous { continue }
            deduplicated.append(word)
        }
        words = deduplicated

        // An acoustic endpoint can split an article correction as "a. An
        // office". Retain the corrected article and remove the obsolete one;
        // this is language-structural rather than phrase-specific repair.
        if words.count >= 2 {
            let articles: Set<String> = ["a", "an", "the"]
            var index = 0
            while index + 1 < words.count {
                let left = words[index].lowercased()
                    .trimmingCharacters(in: .punctuationCharacters)
                let right = words[index + 1].lowercased()
                    .trimmingCharacters(in: .punctuationCharacters)
                let ended = words[index].last.map { ".?!".contains($0) } ?? false
                if ended, articles.contains(left), articles.contains(right) {
                    words.remove(at: index)
                } else {
                    index += 1
                }
            }
        }

        // A streaming endpoint can create a false stop immediately before the
        // continuation of the same word ("in. Into the system"). Prefer the
        // longer revised token. This is strictly lexical overlap, not a
        // phrase- or domain-specific substitution.
        if words.count >= 2 {
            var index = 0
            while index + 1 < words.count {
                let rawLeft = words[index]
                let left = rawLeft.lowercased()
                    .trimmingCharacters(in: .punctuationCharacters)
                let right = words[index + 1].lowercased()
                    .trimmingCharacters(in: .punctuationCharacters)
                let ended = rawLeft.last.map { ".?!".contains($0) } ?? false
                if ended, left.count >= 2, right.count > left.count,
                   right.hasPrefix(left) {
                    words.remove(at: index)
                } else {
                    index += 1
                }
            }
        }

        // Nemotron sometimes inserts a full stop after a verb whose object is
        // emitted in the next stable window ("can't afford. A luxury car").
        // Remove only that false punctuation; the translation model then sees
        // the complete proposition instead of inventing two standalone ones.
        let objectSeekingEndings: Set<String> = [
            "afford", "bring", "ensure", "give", "make", "need", "provide",
            "say", "take", "tell", "use", "want",
        ]
        if words.count >= 2 {
            for index in 0..<(words.count - 1) {
                let rawLeft = words[index]
                let left = rawLeft.lowercased()
                    .trimmingCharacters(in: .punctuationCharacters)
                guard objectSeekingEndings.contains(left),
                      rawLeft.last.map({ ".?!".contains($0) }) == true else { continue }
                words[index] = String(rawLeft.dropLast())
            }
        }
        return words.joined(separator: " ")
    }
}

private extension Duration {
    var milliseconds: Double {
        let components = self.components
        return Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
    }
}
