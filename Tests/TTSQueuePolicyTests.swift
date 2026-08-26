import Testing
@testable import UniversalInterpreter

@Suite("TTS queue policy")
struct TTSQueuePolicyTests {
    @Test func onlyConfirmedTranslationsCanReachTTS() {
        #expect(!TTSQueuePolicy.shouldSpeak(.preview))
        #expect(!TTSQueuePolicy.shouldSpeak(.simultaneousCommitted))
        #expect(TTSQueuePolicy.shouldSpeak(.confirmed))
    }

    @Test func oldSpeechRequestsAreDiscarded() {
        let now = ContinuousClock.now
        #expect(!TTSQueuePolicy.isStale(enqueuedAt: now, now: now))
        #expect(TTSQueuePolicy.isStale(
            enqueuedAt: now.advanced(by: .seconds(-13)), now: now
        ))
    }

    @Test func simultaneousSpeechExpiresBeforeItCanBecomeBacklog() {
        let now = ContinuousClock.now
        #expect(!TTSQueuePolicy.isStale(
            .simultaneousCommitted,
            text: "안녕하세요 여러분",
            enqueuedAt: now.advanced(by: .seconds(-1)),
            now: now
        ))
        #expect(TTSQueuePolicy.isStale(
            .simultaneousCommitted,
            text: "안녕하세요 여러분",
            enqueuedAt: now.advanced(by: .seconds(-2)),
            now: now
        ))
    }

    @Test func shortConfirmedBridgeSentenceSurvivesOneLongUtterance() {
        let now = ContinuousClock.now
        let enqueued = now.advanced(by: .seconds(-5))
        #expect(!TTSQueuePolicy.isStaleConfirmedSentence(
            "나만큼 운이 좋았습니다.", enqueuedAt: enqueued, now: now
        ))
        #expect(!TTSQueuePolicy.isStaleConfirmedSentence(
            "낸시 레이건은 미국 영부인의 역할을 재정의했습니다.",
            enqueuedAt: enqueued, now: now
        ))
        #expect(!TTSQueuePolicy.isStaleConfirmedSentence(
            "지난달 우리는 연구원들과 의사들 간의 더 많은 협력을 촉진하기 위한 새로운 조치를 취했습니다.",
            enqueuedAt: enqueued, now: now
        ))
        #expect(!TTSQueuePolicy.isStaleConfirmedSentence(
            String(repeating: "긴문장", count: 20), enqueuedAt: enqueued, now: now
        ))
        #expect(!TTSQueuePolicy.isStaleConfirmedSentence(
            "나만큼 운이 좋았습니다.",
            enqueuedAt: now.advanced(by: .seconds(-9)), now: now
        ))
        #expect(TTSQueuePolicy.isStaleConfirmedSentence(
            "나만큼 운이 좋았습니다.",
            enqueuedAt: now.advanced(by: .seconds(-13)), now: now
        ))
    }
}

@Suite("Translation queue policy")
struct TranslationQueuePolicyTests {
    @Test func staleTranslationsAreRejectedBeforeCallingTheServer() {
        #expect(!TranslationQueuePolicy.isStale(queueMilliseconds: 3_999))
        #expect(TranslationQueuePolicy.isStale(queueMilliseconds: 4_001))
    }

    @Test func lateTranslationsAreNotSpoken() {
        #expect(!TranslationQueuePolicy.isTooLateToSpeak(
            queueMilliseconds: 500, translationMilliseconds: 3_000
        ))
        #expect(!TranslationQueuePolicy.isTooLateToSpeak(
            queueMilliseconds: 3_500, translationMilliseconds: 3_000
        ))
        #expect(TranslationQueuePolicy.isTooLateToSpeak(
            queueMilliseconds: 2_000, translationMilliseconds: 6_100
        ))
    }
}
