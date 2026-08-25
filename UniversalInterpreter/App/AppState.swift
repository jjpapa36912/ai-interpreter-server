import Foundation

enum ComponentState: String, Sendable {
    case idle = "대기"
    case requestingPermission = "권한 요청 중"
    case starting = "시작 중"
    case running = "연결됨"
    case reconnecting = "재연결 중"
    case stopping = "중지 중"
    case failed = "오류"
}

@MainActor
@Observable
final class AppState {
    var microphoneState: ComponentState = .idle
    var microphoneLevel: Float = 0
    var microphoneFormat = "측정 전"
    var systemAudioState: ComponentState = .idle
    var systemAudioLevel: Float = 0
    var systemAudioFormat = "측정 전"
    var translationAState: ComponentState = .idle
    var translationBState: ComponentState = .idle
    var playbackState: ComponentState = .idle
    var virtualMicState: ComponentState = .idle
    var virtualMicDeviceState: ComponentState = .idle
    var virtualMicTestToneState: ComponentState = .idle
    var listeningMetrics = SessionPerformanceMetrics()
    var speakingMetrics = SessionPerformanceMetrics()
    var applications: [CapturableApplication] = []
    var selectedApplicationID: pid_t?
    var errorMessage: String?
    var translationActionInProgress = false
    var localModelWarmupState: ComponentState = .idle
    var listeningEnabled = true
    var speakingEnabled = false
    var translationEngineMode: TranslationEngineMode =
        CUDAStreamingTranslationProvider.configured() == nil ? .expressiveLocal : .gpuStreaming
    var localVocabularyText = UserDefaults.standard.string(forKey: "localVocabulary") ?? "" {
        didSet { UserDefaults.standard.set(localVocabularyText, forKey: "localVocabulary") }
    }
    var microphonePermission = PermissionManager.microphoneStatus
    var screenCapturePermission = PermissionManager.screenCaptureStatus
    var speechRecognitionPermission = PermissionManager.speechRecognitionStatus
    private var screenCaptureAccessConfirmed = false

    let microphone = MicrophoneCaptureService()
    let systemAudio = SystemAudioCaptureService()
    var translationA: any TranslationProvider = ResidentLocalTranslationProvider()
    var translationB: any TranslationProvider = ResidentLocalTranslationProvider()
    let playback = TranslationPlaybackService()
    let virtualMicOutput = VirtualMicOutputService()
    let virtualMicTestTone = VirtualMicTestToneService()
    let virtualMicDevice = VirtualMicDeviceService()
    private var meterTask: Task<Void, Never>?
    private var systemMeterTask: Task<Void, Never>?
    private var translationInputTask: Task<Void, Never>?
    private var translationOutputTask: Task<Void, Never>?
    private var listeningCaptureWatchdogTask: Task<Void, Never>?
    private var listeningReceivedAudio = false
    private var speakingInputTask: Task<Void, Never>?
    private var speakingOutputTask: Task<Void, Never>?

    var dualTranslationStatus: DualTranslationStatus {
        .resolve(listening: translationAState, speaking: translationBState)
    }

    var selectedApplication: CapturableApplication? {
        applications.first { $0.id == selectedApplicationID }
    }

    func refreshPermissions() {
        microphonePermission = PermissionManager.microphoneStatus
        speechRecognitionPermission = PermissionManager.speechRecognitionStatus
        let detectedScreenStatus = PermissionManager.screenCaptureStatus
        screenCapturePermission = screenCaptureAccessConfirmed ? .authorized : detectedScreenStatus
    }

    func prepareLocalModelInBackground() async {
        guard localModelWarmupState == .idle || localModelWarmupState == .failed else { return }
        localModelWarmupState = .starting
        do {
            try await LocalModelWorker.shared.prepareTranslation()
            localModelWarmupState = .running
        } catch {
            localModelWarmupState = .failed
            // Starting a capture session remains possible; the first real
            // utterance will retry and surface a user-facing processing error.
        }
    }

    func requestSpeechRecognitionPermission() async {
        _ = await PermissionManager.requestSpeechRecognitionAccess()
        refreshPermissions()
    }

    func requestScreenCapturePermission() {
        _ = PermissionManager.requestScreenCaptureAccess()
        refreshPermissions()
    }

    func refreshVirtualMicDevice() {
        virtualMicDeviceState = virtualMicDevice.isLoaded() ? .running : .failed
    }

    func toggleVirtualMicTestTone() async {
        if virtualMicTestToneState == .running {
            await virtualMicTestTone.stop()
            await virtualMicOutput.stop()
            virtualMicTestToneState = .idle
            return
        }
        do {
            virtualMicTestToneState = .starting
            try virtualMicDevice.requireLoaded()
            try await virtualMicOutput.start()
            await virtualMicTestTone.start(output: virtualMicOutput)
            virtualMicTestToneState = .running
        } catch {
            await virtualMicOutput.stop()
            virtualMicTestToneState = .failed
            errorMessage = error.localizedDescription
        }
    }

    func toggleDualTranslation() async {
        guard !translationActionInProgress else { return }
        translationActionInProgress = true
        defer { translationActionInProgress = false }

        if translationAState == .running || translationBState == .running ||
            translationAState == .starting || translationBState == .starting {
            async let listening: Void = stopListeningTranslation()
            async let speaking: Void = stopSpeakingTranslation()
            _ = await (listening, speaking)
            return
        }

        guard listeningEnabled || speakingEnabled else {
            errorMessage = "사용할 통역 방향을 하나 이상 선택해 주세요."
            return
        }

        // Each direction owns its errors and cleanup. Failure in one task does not
        // cancel the other direction.
        if listeningEnabled && speakingEnabled {
            async let listening: Void = toggleListeningTranslation()
            async let speaking: Void = toggleSpeakingTranslation()
            _ = await (listening, speaking)
        } else if listeningEnabled {
            await toggleListeningTranslation()
        } else {
            await toggleSpeakingTranslation()
        }
    }

    func toggleSpeakingTranslation() async {
        if translationBState == .running {
            await stopSpeakingTranslation()
            return
        }

        translationBState = .requestingPermission
        guard await PermissionManager.requestMicrophoneAccess() else {
            translationBState = .failed
            errorMessage = "말하기 통역을 사용하려면 마이크 권한이 필요합니다."
            return
        }

        do {
            translationBState = .starting
            virtualMicState = .starting
            translationB = makeTranslationProvider()
            try await translationB.startSession(sourceLanguage: .korean, targetLanguage: .english)
            speakingMetrics.start()
            try await virtualMicOutput.start()
            let input = try await microphone.startCapture()
            microphoneState = .running
            virtualMicState = .running
            translationBState = .running

            speakingInputTask = Task { [weak self] in
                guard let self else { return }
                for await chunk in input {
                    guard !Task.isCancelled else { break }
                    microphoneLevel = chunk.rms
                    microphoneFormat = "\(Int(chunk.format.sampleRate)) Hz · \(chunk.format.channelCount) ch · Float32"
                    speakingMetrics.recordInput(chunk)
                    do {
                        try await translationB.sendAudio(chunk)
                        speakingMetrics.updatePipeline(await translationB.currentPipelineMetrics())
                    } catch {
                        await handleSpeakingFailure(error)
                        break
                    }
                }
            }
            speakingOutputTask = Task { [weak self] in
                guard let self else { return }
                for await chunk in translationB.translatedAudio {
                    guard !Task.isCancelled else { break }
                    speakingMetrics.recordOutput(chunk)
                    speakingMetrics.updatePipeline(await translationB.currentPipelineMetrics())
                    await virtualMicOutput.writeTranslatedAudio(chunk)
                }
            }
        } catch {
            await microphone.stopCapture()
            let stoppedProvider = translationB
            await stoppedProvider.stopSession()
            translationB = makeTranslationProvider()
            await virtualMicOutput.stop()
            microphoneState = .idle
            virtualMicState = .idle
            translationBState = .failed
            errorMessage = error.localizedDescription
        }
    }

    private func stopSpeakingTranslation() async {
        if translationBState != .idle { translationBState = .stopping }
        speakingInputTask?.cancel()
        speakingOutputTask?.cancel()
        speakingInputTask = nil
        speakingOutputTask = nil
        await microphone.stopCapture()
        let stoppedProvider = translationB
        await stoppedProvider.stopSession()
        translationB = makeTranslationProvider()
        await virtualMicOutput.stop()
        microphoneState = .idle
        virtualMicState = .idle
        translationBState = .idle
    }

    private func handleSpeakingFailure(_ error: Error) async {
        speakingInputTask?.cancel()
        speakingOutputTask?.cancel()
        await microphone.stopCapture()
        let stoppedProvider = translationB
        await stoppedProvider.stopSession()
        translationB = makeTranslationProvider()
        await virtualMicOutput.stop()
        microphoneState = .idle
        virtualMicState = .idle
        translationBState = .failed
        errorMessage = "말하기 통역: \(error.localizedDescription)"
    }

    func toggleListeningTranslation() async {
        if translationAState == .running {
            await stopListeningTranslation()
            return
        }
        guard let app = applications.first(where: { $0.id == selectedApplicationID }) else {
            errorMessage = "먼저 번역할 앱을 선택해 주세요."
            return
        }
        do {
            translationAState = .starting
            playbackState = .starting
            translationA = makeTranslationProvider()
            try await translationA.startSession(sourceLanguage: .english, targetLanguage: .korean)
            listeningMetrics.start()
            try await playback.start()
            let input = try await systemAudio.startCapture(application: app)
            systemAudioState = .running
            playbackState = .running
            translationAState = .running
            listeningReceivedAudio = false
            listeningCaptureWatchdogTask?.cancel()
            listeningCaptureWatchdogTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(4))
                guard !Task.isCancelled, let self,
                      self.translationAState == .running,
                      !self.listeningReceivedAudio else { return }
                self.errorMessage = "Google Chrome 오디오가 들어오지 않습니다. Chrome에서 소리를 재생한 뒤 앱 목록을 새로고침하고 다시 선택해 주세요."
            }

            translationInputTask = Task { [weak self] in
                for await chunk in input {
                    guard !Task.isCancelled else { break }
                    self?.listeningReceivedAudio = true
                    self?.listeningCaptureWatchdogTask?.cancel()
                    self?.systemAudioLevel = chunk.rms
                    self?.systemAudioFormat = "\(Int(chunk.format.sampleRate)) Hz · \(chunk.format.channelCount) ch · \(chunk.format.bitsPerChannel) bit"
                    self?.listeningMetrics.recordInput(chunk)
                    do {
                        try await self?.translationA.sendAudio(chunk)
                        if let pipelineMetrics = await self?.translationA.currentPipelineMetrics() {
                            self?.listeningMetrics.updatePipeline(pipelineMetrics)
                        }
                    } catch {
                        await self?.handleListeningFailure(error)
                        break
                    }
                }
            }
            translationOutputTask = Task { [weak self] in
                guard let self else { return }
                for await chunk in translationA.translatedAudio {
                    guard !Task.isCancelled else { break }
                    listeningMetrics.recordOutput(chunk)
                    listeningMetrics.updatePipeline(await translationA.currentPipelineMetrics())
                    do { try await playback.enqueue(chunk) } catch {
                        await handleListeningFailure(error)
                        break
                    }
                }
            }
        } catch {
            await systemAudio.stopCapture()
            let stoppedProvider = translationA
            await stoppedProvider.stopSession()
            translationA = makeTranslationProvider()
            await playback.stop()
            systemAudioState = .idle
            playbackState = .idle
            translationAState = .failed
            errorMessage = error.localizedDescription
        }
    }

    private func stopListeningTranslation() async {
        if translationAState != .idle { translationAState = .stopping }
        translationInputTask?.cancel()
        translationOutputTask?.cancel()
        listeningCaptureWatchdogTask?.cancel()
        listeningCaptureWatchdogTask = nil
        listeningReceivedAudio = false
        translationInputTask = nil
        translationOutputTask = nil
        await systemAudio.stopCapture()
        let stoppedProvider = translationA
        await stoppedProvider.stopSession()
        translationA = makeTranslationProvider()
        await playback.stop()
        systemAudioState = .idle
        playbackState = .idle
        translationAState = .idle
    }

    private func handleListeningFailure(_ error: Error) async {
        translationInputTask?.cancel()
        translationOutputTask?.cancel()
        listeningCaptureWatchdogTask?.cancel()
        listeningCaptureWatchdogTask = nil
        listeningReceivedAudio = false
        await systemAudio.stopCapture()
        let stoppedProvider = translationA
        await stoppedProvider.stopSession()
        translationA = makeTranslationProvider()
        await playback.stop()
        systemAudioState = .idle
        playbackState = .idle
        translationAState = .failed
        errorMessage = "듣기 통역: \(error.localizedDescription)"
    }

    func refreshApplications() async {
        refreshPermissions()
        do {
            applications = try await systemAudio.availableApplications()
            screenCaptureAccessConfirmed = true
            screenCapturePermission = .authorized
            if !applications.contains(where: { $0.id == selectedApplicationID }) {
                // Chrome is the primary browser/conference capture target for
                // this prototype. Prefer its main process, not a helper, on
                // first launch while preserving an explicit valid user choice.
                selectedApplicationID = applications.first(where: {
                    $0.bundleIdentifier == "com.google.Chrome"
                        || $0.name.caseInsensitiveCompare("Google Chrome") == .orderedSame
                })?.id ?? applications.first?.id
            }
        } catch {
            screenCaptureAccessConfirmed = false
            refreshPermissions()
            applications = []
            selectedApplicationID = nil
            errorMessage = "실제 화면·시스템 오디오 접근이 거부되었습니다. 시스템 설정의 AI Interpreter 토글을 한 번 껐다 켠 뒤 앱을 다시 실행해 주세요. (\(error.localizedDescription))"
        }
    }

    func toggleSystemAudio() async {
        if systemAudioState == .running {
            systemAudioState = .stopping
            systemMeterTask?.cancel()
            await systemAudio.stopCapture()
            systemAudioState = .idle
            systemAudioLevel = 0
            return
        }
        guard let app = applications.first(where: { $0.id == selectedApplicationID }) else {
            errorMessage = "먼저 캡처할 앱을 선택해 주세요."
            return
        }
        do {
            systemAudioState = .starting
            let stream = try await systemAudio.startCapture(application: app)
            systemAudioState = .running
            systemMeterTask = Task { [weak self] in
                for await chunk in stream {
                    guard !Task.isCancelled else { break }
                    self?.systemAudioLevel = chunk.rms
                    self?.systemAudioFormat = "\(Int(chunk.format.sampleRate)) Hz · \(chunk.format.channelCount) ch · \(chunk.format.bitsPerChannel) bit"
                }
            }
        } catch {
            systemAudioState = .failed
            errorMessage = error.localizedDescription
        }
    }

    func toggleMicrophone() async {
        if microphoneState == .running {
            microphoneState = .stopping
            meterTask?.cancel()
            await microphone.stopCapture()
            microphoneState = .idle
            microphoneLevel = 0
            return
        }

        microphoneState = .requestingPermission
        guard await PermissionManager.requestMicrophoneAccess() else {
            microphoneState = .failed
            errorMessage = "마이크 권한이 필요합니다. 시스템 설정에서 허용해 주세요."
            return
        }

        do {
            microphoneState = .starting
            let stream = try await microphone.startCapture()
            microphoneState = .running
            meterTask = Task { [weak self] in
                for await chunk in stream {
                    guard !Task.isCancelled else { break }
                    self?.microphoneLevel = chunk.rms
                    self?.microphoneFormat = "\(Int(chunk.format.sampleRate)) Hz · \(chunk.format.channelCount) ch · Float32"
                }
            }
        } catch {
            microphoneState = .failed
            errorMessage = error.localizedDescription
        }
    }

    private func makeTranslationProvider() -> any TranslationProvider {
        switch translationEngineMode {
        case .economy:
            AppleLocalTranslationProvider(contextualTerms: LocalVocabulary.parse(localVocabularyText))
        case .expressiveLocal:
            ResidentLocalTranslationProvider(
                preferredTerms: LocalVocabulary.parse(localVocabularyText)
            )
        case .gpuStreaming:
            CUDAStreamingTranslationProvider.configured()
                ?? ResidentLocalTranslationProvider(
                    preferredTerms: LocalVocabulary.parse(localVocabularyText)
                )
        }
    }
}
