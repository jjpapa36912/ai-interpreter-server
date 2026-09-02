import Testing
@testable import UniversalInterpreter

@Suite("TTS queue policy")
struct TTSQueuePolicyTests {
    @Test func onlyStarterAndConfirmedTranslationsCanReachTTS() {
        #expect(TTSQueuePolicy.shouldSpeak(.starter))
        #expect(!TTSQueuePolicy.shouldSpeak(.preview))
        #expect(TTSQueuePolicy.shouldSpeak(.simultaneousCommitted))
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

    @Test func lateStarterSpeechCannotBecomeBacklog() {
        let now = ContinuousClock.now
        #expect(!TTSQueuePolicy.isStale(
            .starter,
            text: "게다가,",
            enqueuedAt: now.advanced(by: .milliseconds(-900)),
            now: now
        ))
        #expect(TTSQueuePolicy.isStale(
            .starter,
            text: "게다가,",
            enqueuedAt: now.advanced(by: .milliseconds(-1_100)),
            now: now
        ))
    }

    @Test func confirmedSpeechNeverExpiresOrSkipsMeaning() {
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
        #expect(!TTSQueuePolicy.isStale(
            .confirmed,
            text: "나만큼 운이 좋았습니다.",
            enqueuedAt: now.advanced(by: .seconds(-30)),
            now: now
        ))
    }
}

@Suite("Confirmed translation microbatch")
struct ConfirmedTranslationMicrobatchPolicyTests {
    @Test func normalCadenceEmitsEachTranslationAsSoonAsItCompletes() {
        #expect(ConfirmedTranslationMicrobatchPolicy.maximumItems(
            waitingCount: 0
        ) == 1)
        #expect(ConfirmedTranslationMicrobatchPolicy.maximumItems(
            waitingCount: 2
        ) == 1)
    }

    @Test func pressureCannotReintroduceBurstCompletion() {
        #expect(ConfirmedTranslationMicrobatchPolicy.maximumItems(
            waitingCount: 3
        ) == 1)
        #expect(ConfirmedTranslationMicrobatchPolicy.maximumItems(
            waitingCount: 20
        ) == 1)
    }
}

@Suite("Remote TTS circuit breaker policy")
struct RemoteTTSCircuitBreakerPolicyTests {
    @Test func exhaustedRemoteOutageLocksTheSessionToLocalVoice() {
        #expect(RemoteTTSCircuitBreakerPolicy.shouldOpen(
            firstAudioWasEmitted: false, failureWasFirstAudioTimeout: true
        ))
        #expect(!RemoteTTSCircuitBreakerPolicy.shouldOpen(
            firstAudioWasEmitted: true, failureWasFirstAudioTimeout: true
        ))
        #expect(!RemoteTTSCircuitBreakerPolicy.shouldOpen(
            firstAudioWasEmitted: false, failureWasFirstAudioTimeout: false
        ))
        #expect(RemoteTTSCircuitBreakerPolicy.shouldOpen(
            firstAudioWasEmitted: false,
            failureWasFirstAudioTimeout: false,
            failureWasPermanentBeforeAudio: true
        ))
        #expect(RemoteTTSCircuitBreakerPolicy.shouldOpen(
            firstAudioWasEmitted: false,
            failureWasFirstAudioTimeout: false,
            failureWasRemoteOutageBeforeAudio: true
        ))
        #expect(!RemoteTTSCircuitBreakerPolicy.shouldOpen(
            firstAudioWasEmitted: true,
            failureWasFirstAudioTimeout: false,
            failureWasPermanentBeforeAudio: true
        ))
        #expect(!RemoteTTSCircuitBreakerPolicy.shouldOpen(
            firstAudioWasEmitted: false,
            failureWasFirstAudioTimeout: true,
            failureWasRemoteOutageBeforeAudio: true,
            sessionHasDeliveredRemoteSpeech: true
        ))
    }

    @Test func deadlineAllowsNormalLongClauseGpuLatency() {
        #expect(RemoteTTSCircuitBreakerPolicy.firstAudioDeadlineSeconds >= 3.0)
        #expect(RemoteTTSCircuitBreakerPolicy.firstAudioDeadlineSeconds <= 3.5)
    }
}

@Suite("Remote TTS startup availability policy")
struct RemoteTTSAvailabilityPolicyTests {
    @Test func onlySuccessfulReadyResponsesSelectGPUVoice() {
        #expect(RemoteTTSAvailabilityPolicy.isReady(statusCode: 200))
        #expect(RemoteTTSAvailabilityPolicy.isReady(statusCode: 204))
        #expect(!RemoteTTSAvailabilityPolicy.isReady(statusCode: 401))
        #expect(!RemoteTTSAvailabilityPolicy.isReady(statusCode: 503))
    }

    @Test func offlineProbeIsStrictlyBounded() {
        #expect(RemoteTTSAvailabilityPolicy.attemptCount == 2)
        #expect(RemoteTTSAvailabilityPolicy.requestTimeoutSeconds <= 1.0)
    }
}

@Suite("Confirmed contextual prefetch trust policy")
struct ConfirmedTranslationPrefetchPolicyTests {
    @Test func ordinaryClauseKeepsFastContextualLane() {
        #expect(ConfirmedTranslationPrefetchPolicy.canUseUnvalidatedPrefetch(
            source: "The atmosphere is extremely dense.",
            sourceLanguage: .english, targetLanguage: .korean
        ))
    }

    @Test func namedEntitiesRequireResidentValidation() {
        #expect(!ConfirmedTranslationPrefetchPolicy.canUseUnvalidatedPrefetch(
            source: "What turned Venus from paradise to pressure cooker?",
            sourceLanguage: .english, targetLanguage: .korean
        ))
        #expect(!ConfirmedTranslationPrefetchPolicy.canUseUnvalidatedPrefetch(
            source: "Mysterious Mercury appears.",
            sourceLanguage: .english, targetLanguage: .korean
        ))
    }
}

@Suite("Speech mailbox capacity policy")
struct SpeechMailboxCapacityPolicyTests {
    @Test func speculativeWorkIsEvictedBeforeAnyConfirmedSentence() {
        let decision = SpeechMailboxCapacityPolicy.overflowDecision(
            waitingKinds: [
                .confirmed, .simultaneousCommitted, .confirmed, .starter,
            ],
            incomingKind: .confirmed
        )

        #expect(decision == .evictWaiting(at: 1))
    }

    @Test func fullConfirmedMailboxNeverDiscardsAConfirmedSentence() {
        let waiting = Array(
            repeating: TTSQueuePolicy.TranslationKind.confirmed,
            count: SpeechFragmentCoalescingPolicy.maximumWaitingGenerations
        )

        #expect(SpeechMailboxCapacityPolicy.overflowDecision(
            waitingKinds: waiting,
            incomingKind: .confirmed
        ) == .appendWithoutEviction)
        #expect(SpeechMailboxCapacityPolicy.overflowDecision(
            waitingKinds: waiting,
            incomingKind: .simultaneousCommitted
        ) == .discardIncoming)
    }

    @Test func slowTTSStillAcceptsAContinuousRunOfConfirmedResults() {
        // Model an active synthesis that never drains during this burst. The
        // translation result loop must be able to hand off every confirmed
        // sentence even after the nominal waiting-generation target is full.
        var waitingIDs: [Int] = []
        let resultIDs = Array(0..<96)

        for id in resultIDs {
            if waitingIDs.count
                >= SpeechFragmentCoalescingPolicy.maximumWaitingGenerations {
                let decision = SpeechMailboxCapacityPolicy.overflowDecision(
                    waitingKinds: waitingIDs.map { _ in .confirmed },
                    incomingKind: .confirmed
                )
                switch decision {
                case let .evictWaiting(index):
                    Issue.record("confirmed result \(waitingIDs[index]) was selected for eviction")
                    waitingIDs.remove(at: index)
                case .discardIncoming:
                    Issue.record("confirmed result \(id) was discarded while TTS was busy")
                case .appendWithoutEviction:
                    break
                }
            }
            waitingIDs.append(id)
        }

        #expect(waitingIDs == resultIDs)
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

@Suite("Translation input coalescing")
struct TranslationInputCoalescingPolicyTests {
    @Test func alreadySizedConfirmedFragmentsKeepGeminiLikeCadence() {
        #expect(!TranslationInputCoalescingPolicy.canMerge(
            "The market has changed rapidly.",
            with: "Workers are feeling the pressure."
        ))
    }

    @Test func onlyTinyTailsAreRescuedIntoThePreviousInference() {
        #expect(TranslationInputCoalescingPolicy.canMerge(
            "The market changed rapidly.", with: "for everyone"
        ))
    }

    @Test func batchesRemainShortEnoughForLiveInterpretation() {
        let existing = Array(repeating: "word", count: 11).joined(separator: " ")
        #expect(!TranslationInputCoalescingPolicy.canMerge(
            existing, with: "two more words"
        ))
        #expect(!TranslationInputCoalescingPolicy.canMerge(
            String(repeating: "a", count: 90), with: "another fragment"
        ))
    }

    @Test func prolongedInputPressurePreservesEveryConfirmedWordInOrder() {
        let fragments = (0..<120).map { "source\($0) keeps moving." }
        var queuedBatches: [String] = []

        for fragment in fragments {
            if let last = queuedBatches.last,
               TranslationInputCoalescingPolicy.canMerge(last, with: fragment) {
                queuedBatches[queuedBatches.count - 1] = last + " " + fragment
            } else {
                queuedBatches.append(fragment)
            }
        }

        #expect(queuedBatches.joined(separator: " ") == fragments.joined(separator: " "))
        #expect(queuedBatches.allSatisfy {
            $0.count <= TranslationInputCoalescingPolicy.maximumCharacters
                && $0.split(whereSeparator: \Character.isWhitespace).count
                    <= TranslationInputCoalescingPolicy.maximumWords
        })
    }

    @Test func confirmedOpenFragmentsReceiveOneBoundedPieceOfRightContext() {
        #expect(ConfirmedTranslationContextPolicy.needsRightContext(
            "The smallest planet made out"
        ))
        #expect(ConfirmedTranslationContextPolicy.needsRightContext(
            "of water that is now"
        ))
        #expect(ConfirmedTranslationContextPolicy.needsRightContext(
            "There has never been"
        ))
        #expect(ConfirmedTranslationContextPolicy.needsRightContext(
            "There has never"
        ))
        #expect(ConfirmedTranslationContextPolicy.needsRightContext(
            "A mysterious signal appears"
        ))
        #expect(ConfirmedTranslationContextPolicy.needsRightContext(
            "a better time to boldly"
        ))
        #expect(ConfirmedTranslationContextPolicy.canMerge(
            "There has never been", with: "a better time to explore."
        ))
    }

    @Test func basecampSixContextGuardDoesNotRebuildContinuousSpeech() {
        #expect(ConfirmedTranslationContextPolicy.needsRightContext(
            "The proposal has larger effects"
        ))
        #expect(!ConfirmedTranslationContextPolicy.needsRightContext(
            "People across the country already understand why this proposal matters"
        ))
    }

    @Test func rightAttachingNounsAndSuperlativesReceiveAnExtraContextTick() {
        #expect(ConfirmedTranslationContextPolicy.needsExtendedRightContext(
            "hidden deep inside is a clue"
        ))
        #expect(ConfirmedTranslationContextPolicy.needsExtendedRightContext(
            "is possibly the greatest survival"
        ))
        #expect(ConfirmedTranslationContextPolicy.needsExtendedRightContext(
            "with the most lunar like"
        ))
        #expect(!ConfirmedTranslationContextPolicy.needsExtendedRightContext(
            "a clue to a different past."
        ))
        #expect(!ConfirmedTranslationContextPolicy.needsExtendedRightContext(
            "the greatest survival story of all."
        ))
        #expect(
            ConfirmedTranslationContextPolicy.idleHold(
                for: "hidden deep inside is a clue"
            ) == ConfirmedTranslationContextPolicy.structurallyAmbiguousIdleHold
        )
    }

    @Test func missingHeadOrObjectReceivesOnlyTheExtendedContextTick() {
        let fragments = [
            "The proposal has got significant",
            "The latest financial",
            "The advice I can give is treat",
            "The policy cannot convince",
            "We have to be very",
        ]
        for fragment in fragments {
            #expect(ConfirmedTranslationContextPolicy.needsExtendedRightContext(
                fragment
            ))
            #expect(
                ConfirmedTranslationContextPolicy.idleHold(for: fragment)
                    == ConfirmedTranslationContextPolicy
                        .structurallyAmbiguousIdleHold
            )
        }

        #expect(!ConfirmedTranslationContextPolicy.needsExtendedRightContext(
            "The service is really effective"
        ))
        #expect(ConfirmedTranslationContextPolicy.canMerge(
            "The proposal has got significant",
            with: "financial consequences."
        ))
    }

    @Test func anOpenMergedPairMayReceiveAThirdCadenceFragment() {
        let firstPair = "The smallest planet made out of the densest stuff with"
        #expect(ConfirmedTranslationContextPolicy.shouldContinueAccumulating(
            firstPair
        ))
        #expect(ConfirmedTranslationContextPolicy.canMerge(
            firstPair, with: "the most surprising history."
        ))

        let completed = firstPair + " the most surprising history."
        #expect(!ConfirmedTranslationContextPolicy.shouldContinueAccumulating(
            completed
        ))
    }

    @Test func contextualAccumulationRemainsStrictlyBounded() {
        let atLimit = Array(repeating: "unfinished", count: 16)
            .joined(separator: " ")
        #expect(!ConfirmedTranslationContextPolicy.shouldContinueAccumulating(
            atLimit
        ))
    }

    @Test func completeCadenceUnitsStillTranslateImmediately() {
        // Explicit sentence completion never pays the context hold.
        #expect(!ConfirmedTranslationContextPolicy.needsRightContext(
            "This approach works well."
        ))
        #expect(!ConfirmedTranslationContextPolicy.needsRightContext(
            "Mercury is the smallest planet."
        ))
        #expect(!ConfirmedTranslationContextPolicy.canMerge(
            "This approach works well.", with: "The next point matters."
        ))
    }

    @Test func grammaticalOpeningsReceiveOnlyOneBoundedContextPiece() {
        let opening = "Reviewing the report more than twice"
        let continuation = "can hide important changes."
        #expect(ConfirmedTranslationContextPolicy.needsRightContext(opening))
        #expect(ConfirmedTranslationContextPolicy.canMerge(
            opening, with: continuation
        ))
        #expect(!ConfirmedTranslationContextPolicy.needsRightContext(
            opening + " " + continuation
        ))
    }

    @Test func contextualMergeCannotRebuildAParagraph() {
        let longTail = Array(repeating: "unfinished", count: 12)
            .joined(separator: " ")
        #expect(!ConfirmedTranslationContextPolicy.canMerge(
            "There has never been", with: longTail
        ))
    }

    @Test func leadingSentenceClosureFinishesTheDeferredPhraseOnly() {
        #expect(ConfirmedTranslationContextPolicy.queueUnits(
            "launch. Teams inside the control center",
            hasDeferredOpenPhrase: true
        ) == ["launch.", "Teams inside the control center"])
    }

    @Test func tinyStandaloneTurnStaysWithItsFollowingSentence() {
        #expect(ConfirmedTranslationContextPolicy.queueUnits(
            "Yes. It is difficult.", hasDeferredOpenPhrase: false
        ) == ["Yes. It is difficult."])
    }

    @Test func completeSentencesBecomeIndependentTranslationUnits() {
        #expect(ConfirmedTranslationContextPolicy.queueUnits(
            "The mission launched. The team began its work.",
            hasDeferredOpenPhrase: false
        ) == ["The mission launched. The team began its work."])
    }

    @Test func queueUnitPartitionPreservesEverySourceWordInOrder() {
        let source = "The first phase ended. A second phase began and continued"
        let units = ConfirmedTranslationContextPolicy.queueUnits(
            source, hasDeferredOpenPhrase: false
        )
        #expect(units.joined(separator: " ") == source)
    }

    @Test func longConfirmedSentenceCannotHeadOfLineBlockSafeContinuation() {
        let source = "Reform is a party for working people and it is incredibly important that we lower household bills and raise wages for families."
        let units = ConfirmedTranslationContextPolicy.queueUnits(
            source, hasDeferredOpenPhrase: false
        )
        #expect(units == [
            "Reform is a party for working people",
            "and it is incredibly important that we lower household bills and raise wages for families.",
        ])
        #expect(units.joined(separator: " ") == source)
    }

    @Test func unsafeLongSubjectPhraseRemainsIntact() {
        let source = "The people in communities across the whole northern region who asked for meaningful changes are still waiting for a clear answer from the government."
        let units = ConfirmedTranslationContextPolicy.queueUnits(
            source, hasDeferredOpenPhrase: false
        )
        #expect(units == [source])
    }

    @Test func oversizedCommaPhrasesBatchWithoutLosingSourceWords() {
        let source = "Well, we are going to ensure that we bring in more people from the outside, from the private sector, not just recruiting from think tanks or the Treasury,"
        let units = ConfirmedTranslationContextPolicy.queueUnits(
            source, hasDeferredOpenPhrase: false
        )

        #expect(units.count >= 2)
        #expect(units.joined(separator: " ") == source)
        #expect(units.allSatisfy {
            $0.split(whereSeparator: \Character.isWhitespace).count <=
                ConfirmedTranslationContextPolicy.maximumDispatchWords
        })
    }

    @Test func nearbySafeBoundaryPreventsAConfirmedParagraphFromBlockingTheQueue() {
        let source = "oh actually probably not for a while, because well first of all reform's got to get elected, then we've got to get the public finances in shape, You have to say to them, we will lower taxes after that"
        let units = ConfirmedTranslationContextPolicy.queueUnits(
            source, hasDeferredOpenPhrase: false
        )

        #expect(units.count >= 2)
        #expect(units.first == "oh actually probably not for a while, because well first of all reform's got to get elected,")
        #expect(units.joined(separator: " ") == source)
        #expect(units.allSatisfy {
            $0.split(whereSeparator: \Character.isWhitespace).count <=
                ConfirmedTranslationContextPolicy.maximumBoundarySearchWords
        })
    }

    @Test func omittedCopulaPassiveClauseRemainsIntactWithoutPredicateProof() {
        // Basecamp 6 deliberately requires explicit predicate evidence before
        // making an irreversible split. Preserving an ambiguous ASR phrase is
        // safer than manufacturing a fluent but semantically wrong boundary.
        let source = "All the pressure inside amplified by the fact that monthly costs exceeded household income and families could no longer afford the essentials they needed."
        let units = ConfirmedTranslationContextPolicy.queueUnits(
            source, hasDeferredOpenPhrase: false
        )

        #expect(units == [source])
    }
}

@Suite("Confirmed translation backlog")
struct ConfirmedTranslationBacklogPolicyTests {
    @Test func normalCadenceDoesNotWaitForBatching() {
        #expect(!ConfirmedTranslationBacklogPolicy.canMerge(
            "The first phrase is ready.",
            with: "The next phrase follows.",
            initialWaitingCount: 1,
            mergedFragments: 1
        ))
    }

    @Test func pressureBatchIsLosslessAndStrictlyBounded() {
        let first = "The first phrase is ready."
        let second = "The next phrase follows."
        #expect(ConfirmedTranslationBacklogPolicy.canMerge(
            first,
            with: second,
            initialWaitingCount: 3,
            mergedFragments: 1
        ))
        let combined = first + " " + second
        #expect(combined.split(whereSeparator: \Character.isWhitespace).count
            <= ConfirmedTranslationBacklogPolicy.maximumWords)
        #expect(combined.count
            <= ConfirmedTranslationBacklogPolicy.maximumCharacters)
    }

    @Test func pressureBatchNeverRebuildsAParagraph() {
        let long = Array(repeating: "confirmed", count: 12)
            .joined(separator: " ")
        #expect(!ConfirmedTranslationBacklogPolicy.canMerge(
            long,
            with: "another complete phrase arrives",
            initialWaitingCount: 4,
            mergedFragments: 1
        ))
        #expect(!ConfirmedTranslationBacklogPolicy.canMerge(
            "one short phrase",
            with: "another short phrase",
            initialWaitingCount: 4,
            mergedFragments: ConfirmedTranslationBacklogPolicy.maximumFragments
        ))
    }
}

@Suite("Incremental preview exposure")
struct IncrementalPreviewExposurePolicyTests {
    @Test func rejectsRepeatedButUnprovenContextualDelta() {
        #expect(!IncrementalPreviewExposurePolicy.shouldExpose(
            sourceIsStable: false, semanticBoundaryApproved: false
        ))
        #expect(!IncrementalPreviewExposurePolicy.shouldExpose(
            sourceIsStable: true, semanticBoundaryApproved: false
        ))
    }

    @Test func exposesOnlyIndependentlyProvenIncrementalDelta() {
        #expect(IncrementalPreviewExposurePolicy.shouldExpose(
            sourceIsStable: true, semanticBoundaryApproved: true
        ))
    }
}

@Suite("Speech fragment coalescing")
struct SpeechFragmentCoalescingPolicyTests {
    @Test func keepsABoundedFIFOBehindTheActiveGeneration() {
        #expect(SpeechFragmentCoalescingPolicy.maximumWaitingGenerations == 6)
    }

    @Test func neverMergesPastOneShortBreath() {
        #expect(SpeechFragmentCoalescingPolicy.canMerge(
            "이것은 짧은 구절이고", with: "자연스럽게 이어집니다."
        ))
        #expect(!SpeechFragmentCoalescingPolicy.canMerge(
            String(repeating: "가", count: 40),
            with: String(repeating: "나", count: 10)
        ))
        let words = Array(repeating: "단어", count: 13).joined(separator: " ")
        #expect(!SpeechFragmentCoalescingPolicy.canMerge(words, with: "추가"))
    }

    @Test func keepsInterpreterSizedSourceUnitsAsSeparateTranslationCalls() {
        #expect(!TranslationInputCoalescingPolicy.canMerge(
            "As the planet closest to", with: "the sun Mercury is"
        ))
        #expect(TranslationInputCoalescingPolicy.canMerge(
            "in craters", with: "and valleys"
        ))
    }

    @Test func streamingNeuralVoiceAmortizesRequestStartupWithoutUnboundedBatches() {
        let streaming = SpeechFragmentCoalescingPolicy.generationLimits(
            usesStreamingNeuralVoice: true
        )
        let local = SpeechFragmentCoalescingPolicy.generationLimits(
            usesStreamingNeuralVoice: false
        )

        #expect(streaming.words == local.words)
        #expect(streaming.characters < local.characters)
        #expect(SpeechFragmentCoalescingPolicy.canMerge(
            "태양계가 형성된 초기 기간에 무슨 일이 일어났는가에 대한",
            with: "기록을 수성과 달이 보존합니다.",
            limits: streaming
        ))
        #expect(SpeechFragmentCoalescingPolicy.canMerge(
            "태양계가 형성된 초기 기간에 무슨 일이 일어났는가에 대한",
            with: "기록을 수성과 달이 보존합니다.",
            limits: local
        ))

        let oversized = Array(repeating: "확정번역", count: 13).joined(separator: " ")
        #expect(!SpeechFragmentCoalescingPolicy.canMerge(
            oversized, with: "추가 문장", limits: streaming
        ))
    }

    @Test func mergesQueuedConfirmedSpeechOnlyForTheLowStartupCostLocalVoice() {
        #expect(SpeechFragmentCoalescingPolicy.shouldMergeBackloggedConfirmed(
            usesStreamingNeuralVoice: false
        ))
        #expect(!SpeechFragmentCoalescingPolicy.shouldMergeBackloggedConfirmed(
            usesStreamingNeuralVoice: true
        ))
    }

    @Test func adaptsHoldToSourceRelativeInterpretationLag() {
        #expect(SpeechFragmentCoalescingPolicy.maximumHold(forEstimatedLagMilliseconds: 500) == .milliseconds(650))
        #expect(SpeechFragmentCoalescingPolicy.maximumHold(forEstimatedLagMilliseconds: 1_500) == .milliseconds(350))
        #expect(SpeechFragmentCoalescingPolicy.maximumHold(forEstimatedLagMilliseconds: 2_500) == .milliseconds(120))
    }

    @Test func holdsOnlyGrammaticallyOpenTinyFragments() {
        #expect(SpeechFragmentCoalescingPolicy.shouldHold("장은"))
        #expect(SpeechFragmentCoalescingPolicy.shouldHold("NASA의 다음"))
        #expect(!SpeechFragmentCoalescingPolicy.shouldHold("맞습니다."))
        #expect(SpeechFragmentCoalescingPolicy.shouldHold("여기서 먼저 해야 합니다"))
        #expect(!SpeechFragmentCoalescingPolicy.shouldHold(
            "여기서 먼저 반드시 해야 할 일입니다"
        ))
        #expect(!SpeechFragmentCoalescingPolicy.shouldHold(
            "지금까지발견된가장중요하고획기적인변화입니다"
        ))
    }

    @Test func confirmedTranslationNeverWaitsAfterItIsVisible() {
        #expect(!SpeechFragmentCoalescingPolicy.shouldHold(
            "우리는 그 이상을 넘어서",
            translationKind: .confirmed
        ))
        #expect(!SpeechFragmentCoalescingPolicy.shouldHold(
            "다음 문장입니다",
            translationKind: .confirmed
        ))
        #expect(SpeechFragmentCoalescingPolicy.shouldHold(
            "우리는 그 이상을 넘어서",
            translationKind: .simultaneousCommitted
        ))
    }
}
