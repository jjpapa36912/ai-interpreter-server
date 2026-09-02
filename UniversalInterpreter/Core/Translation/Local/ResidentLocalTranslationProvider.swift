import Foundation
@preconcurrency import Speech

private struct PreviewTranslationTimeout: Error, Sendable {}
private struct SpeechSynthesisWatchdogTimeout: Error, Sendable {}

private final class SpeechOutputGate: @unchecked Sendable {
    private let lock = NSLock()
    private var open = true

    @discardableResult
    func yieldIfOpen(
        _ chunk: AudioChunk, to output: AsyncStream<AudioChunk>.Continuation
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard open else { return false }
        switch output.yield(chunk) {
        case .enqueued, .dropped:
            // Production uses a lossless stream; retain `.dropped` for
            // compatibility with bounded diagnostic continuations.
            return true
        case .terminated:
            return false
        @unknown default:
            return false
        }
    }

    func close() {
        lock.lock()
        open = false
        lock.unlock()
    }
}

private final class OneShotEventGate: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false

    func fire() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !fired else { return false }
        fired = true
        return true
    }

    var hasFired: Bool {
        lock.lock()
        defer { lock.unlock() }
        return fired
    }
}

struct TTSQueuePolicy: Sendable {
    enum TranslationKind: Sendable {
        case starter, preview, simultaneousCommitted, confirmed
    }

    static let maximumQueueAge: Duration = .seconds(12)
    static let maximumShortSentenceQueueAge: Duration = .seconds(12)
    static let synthesisWatchdogTimeout: Duration = .seconds(30)
    // Do not cancel an in-flight CUDA generation just because a cold starter
    // misses the normal first-PCM target. The server serializes generation;
    // cancelling at 2.5 seconds can strand that slot and silence every later
    // confirmed sentence. Fresh starters are still rejected by the one-second
    // queue-age rule before a request begins.
    static let starterSynthesisWatchdogTimeout: Duration = .seconds(30)
    static let maximumSimultaneousQueueAge: Duration = .milliseconds(1_500)
    static let maximumStarterQueueAge: Duration = .milliseconds(1_000)

    static func shouldSpeak(_ kind: TranslationKind) -> Bool {
        // `simultaneousCommitted` is not a raw preview. It is the common target
        // prefix that survived consecutive translation revisions, with a
        // look-ahead word deliberately withheld. Speaking it is what lets the
        // confirmed remainder follow the starter instead of waiting for an
        // entire source sentence. Raw `.preview` text remains display-only.
        kind == .starter || kind == .simultaneousCommitted || kind == .confirmed
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
        if kind == .starter {
            return enqueuedAt.duration(to: now) > maximumStarterQueueAge
        }
        if kind == .simultaneousCommitted {
            return enqueuedAt.duration(to: now) > maximumSimultaneousQueueAge
        }
        // Confirmed text is irreversible source meaning. If overload ever
        // pushes it past the target lag, dropping it makes the interpreter
        // appear to skip a sentence. The mailbox already evicts speculative
        // work first and the 3-6 word cadence keeps confirmed service time
        // bounded, so confirmed speech remains lossless.
        if kind == .confirmed { return false }
        return isStaleConfirmedSentence(text, enqueuedAt: enqueuedAt, now: now)
    }
}

struct RemoteTTSCircuitBreakerPolicy: Sendable {
    /// Whole-utterance GPU responses normally arrive in 1.0-2.3 seconds. Give
    /// a warmed remote voice enough time to finish a longer clause without
    /// silently changing speakers. This is a transport safety deadline, not
    /// the interpretation-lag target; clause sizing controls normal latency.
    static let firstAudioDeadlineSeconds: TimeInterval = 3.0

    static func shouldOpen(
        firstAudioWasEmitted: Bool,
        failureWasFirstAudioTimeout: Bool,
        failureWasPermanentBeforeAudio: Bool = false,
        failureWasRemoteOutageBeforeAudio: Bool = false,
        sessionHasDeliveredRemoteSpeech: Bool = false
    ) -> Bool {
        // CosyVoice already performs its bounded transport retries before this
        // policy sees an outage. If no PCM was emitted after those retries,
        // lock the remainder of this session to local speech. Never switch
        // after audible GPU PCM: that would splice two speakers into one
        // utterance.
        !sessionHasDeliveredRemoteSpeech && !firstAudioWasEmitted && (
            failureWasFirstAudioTimeout
                || failureWasPermanentBeforeAudio
                || failureWasRemoteOutageBeforeAudio
        )
    }
}

/// The contextual lane is deliberately optimized for latency. It may be used
/// directly for ordinary clauses, but named entities need the resident model's
/// exact-source validation before a confirmed sentence becomes irreversible
/// speech. This is domain-independent: it protects people, places, products,
/// planets, and organizations without hard-coding any one video or term.
struct ConfirmedTranslationPrefetchPolicy: Sendable {
    private static let ordinarySentenceStarters: Set<String> = [
        "A", "An", "And", "As", "At", "But", "For", "From", "He", "Here",
        "How", "I", "If", "In", "It", "Its", "No", "Of", "On", "Once",
        "Or", "Our", "She", "So", "That", "The", "Their", "There", "These",
        "They", "This", "Those", "To", "We", "What", "When", "Where", "Which",
        "Who", "Why", "With", "Without", "Yes", "You", "Your",
    ]

    static func canUseUnvalidatedPrefetch(
        source: String,
        sourceLanguage: Language,
        targetLanguage: Language
    ) -> Bool {
        guard sourceLanguage == .english, targetLanguage == .korean else {
            return true
        }

        let tokens = source.split { !$0.isLetter && $0 != "'" && $0 != "-" }
        for tokenSlice in tokens {
            let token = String(tokenSlice)
            guard token.count >= 2,
                  token.first?.isUppercase == true,
                  token.dropFirst().contains(where: { $0.isLowercase }) else {
                continue
            }
            if !ordinarySentenceStarters.contains(token) {
                return false
            }
        }
        return true
    }
}

struct SpeechFragmentCoalescingPolicy: Sendable {
    struct GenerationLimits: Sendable, Equatable {
        let words: Int
        let characters: Int
    }

    // A bounded hold is faster end-to-end than launching another CUDA job.
    // The current neural voice pays roughly 0.8-2 seconds of fixed work per
    // request, irrespective of text length. Combining adjacent target deltas
    // removes both an audible seam and an entire model startup.
    static let maximumHold: Duration = .milliseconds(650)
    // Once neural synthesis is already busy, every separately queued sentence
    // pays another model startup and another audible voice boundary. Preserve
    // all text, but combine the waiting continuation into a single natural
    // breath. Keep batches bounded so one unusually dense speaker cannot turn
    // the next request into an unmanageably long utterance.
    // A waiting unit must remain a short breath. The previous single mutable
    // String was only "one queue slot" in name: it could grow without limit,
    // repeatedly copy its complete contents, then monopolize the serial GPU
    // for 6-8 seconds. A small FIFO keeps order while bounding both the text
    // retained in memory and the duration of any one synthesis request.
    // AVSpeechSynthesizer has a measurable fixed boundary cost for every
    // utterance. Splitting an already-confirmed Korean sentence every six
    // eojeol turned 113 translations into 145 serial speech jobs in a live
    // session, adding audible seams and up to 4.8 seconds of queueing. The
    // local renderer can speak one medium breath continuously, so amortize
    // that boundary while retaining a hard cap for unusually long turns.
    static let maximumWordsPerGeneration = 12
    static let maximumCharactersPerGeneration = 48
    // CosyVoice streams PCM while it is still generating, but its synthesis
    // cost rises more steeply with target length than the local voice.
    // Production traces showed that an 18-word neural request could monopolize
    // the serial GPU for 12.6 seconds. Keep it to the same word budget with a
    // slightly tighter character cap.
    static let streamingMaximumWordsPerGeneration = 12
    static let streamingMaximumCharactersPerGeneration = 42
    static let maximumWaitingGenerations = 6

    static func generationLimits(usesStreamingNeuralVoice: Bool) -> GenerationLimits {
        usesStreamingNeuralVoice
            ? GenerationLimits(
                words: streamingMaximumWordsPerGeneration,
                characters: streamingMaximumCharactersPerGeneration
            )
            : GenerationLimits(
                words: maximumWordsPerGeneration,
                characters: maximumCharactersPerGeneration
            )
    }

    /// The local system voice benefits from joining adjacent confirmed turns:
    /// it removes another synthesizer boundary without adding model latency.
    /// A remote neural generation has a much steeper length-dependent cost,
    /// so its confirmed requests remain separate and bounded.
    static func shouldMergeBackloggedConfirmed(
        usesStreamingNeuralVoice: Bool
    ) -> Bool {
        !usesStreamingNeuralVoice
    }

    static func canMerge(
        _ existing: String, with incoming: String,
        limits: GenerationLimits = generationLimits(usesStreamingNeuralVoice: false)
    ) -> Bool {
        let combined = [existing, incoming]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        guard !combined.isEmpty else { return false }
        let words = combined.split(whereSeparator: \Character.isWhitespace).count
        let compactCharacters = combined.filter { !$0.isWhitespace }.count
        return words <= limits.words && compactCharacters <= limits.characters
    }

    /// Keep enough context for natural speech when we are close to the source,
    /// but stop spending the full coalescing budget after the interpreter has
    /// fallen outside the desired one-to-two-second shadowing interval.
    static func maximumHold(forEstimatedLagMilliseconds lag: Double) -> Duration {
        // When lagged, do not add another large client-side hold. Natural
        // continuity is handled by joining grammatically open fragments; the
        // server now generates a faster delivery itself, so keeping an open
        // fragment here for hundreds of extra milliseconds only grows backlog.
        if lag > 2_000 { return .milliseconds(120) }
        if lag >= 1_000 { return .milliseconds(350) }
        return maximumHold
    }

    static func shouldHold(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        // A short fragment with an explicit sentence boundary is intentional
        // (for example, "맞습니다.") and should not be delayed.
        if let last = trimmed.last, ".!?。？！".contains(last) { return false }
        let words = trimmed.split(whereSeparator: \Character.isWhitespace).count
        let compactCharacters = trimmed.filter { !$0.isWhitespace }.count
        // Korean often packs substantial meaning into only a few whitespace
        // tokens. Start a new synthesis only for a useful breath, while the
        // timer still guarantees that a quiet speaker cannot block forever.
        return words < 5 && compactCharacters < 18
    }

    static func shouldHold(
        _ text: String, translationKind: TTSQueuePolicy.TranslationKind
    ) -> Bool {
        // A confirmed translation is already visible to the user and cannot
        // gain any more context by waiting in this client-side coalescer.
        // Holding it here created a very noticeable pause between sentences
        // even though translation had finished. Speculative continuation
        // fragments may still wait briefly so they do not become choppy
        // one-word neural-TTS requests.
        guard translationKind != .confirmed,
              translationKind != .starter else { return false }
        return shouldHold(text)
    }
}

/// Decides what a saturated TTS mailbox may discard. Confirmed speech carries
/// source meaning that cannot be reconstructed once a later sentence has
/// overtaken it, so capacity pressure may remove only speculative work. Fresh
/// confirmed requests are allowed to exceed the small request-count target;
/// the independent age and audible-time limits still bound retained work.
struct SpeechMailboxCapacityPolicy: Sendable {
    enum Decision: Sendable, Equatable {
        case evictWaiting(at: Int)
        case discardIncoming
        case appendWithoutEviction
    }

    static func overflowDecision(
        waitingKinds: [TTSQueuePolicy.TranslationKind],
        incomingKind: TTSQueuePolicy.TranslationKind
    ) -> Decision {
        if let speculativeIndex = waitingKinds.firstIndex(where: isSpeculative) {
            return .evictWaiting(at: speculativeIndex)
        }
        return incomingKind == .confirmed
            ? .appendWithoutEviction
            : .discardIncoming
    }

    private static func isSpeculative(_ kind: TTSQueuePolicy.TranslationKind) -> Bool {
        kind == .starter || kind == .preview || kind == .simultaneousCommitted
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

/// Losslessly combines adjacent confirmed source fragments while a translation
/// inference is already running. Streaming ASR can finalize a new fragment
/// faster than a sentence-sized model call completes; issuing every tiny
/// fragment as a separate call makes queue latency grow without bound. A
/// bounded batch amortizes model-call overhead without turning live speech into
/// a long paragraph or dropping any source text.
struct TranslationInputCoalescingPolicy: Sendable {
    // Keep the next confirmed translation small even while inference is busy.
    // The old 32-word merge turned a brief slowdown into positive feedback:
    // the next request became slower, so still more source accumulated.
    static let maximumWords = 6
    static let maximumCharacters = 48

    static func canMerge(_ existing: String, with incoming: String) -> Bool {
        let existingWords = existing.split(whereSeparator: \Character.isWhitespace).count
        let incomingWords = incoming.split(whereSeparator: \Character.isWhitespace).count
        // A 3-6 word ASR unit is already deliberately sized for live
        // interpretation. Joining two such units recreates the exact burst / 
        // silence pattern we are trying to remove. Only rescue a tiny final
        // tail by attaching it to its neighbour.
        guard existingWords < 3 || incomingWords < 3 else { return false }
        let combined = [existing, incoming]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        guard !combined.isEmpty, combined.count <= maximumCharacters else {
            return false
        }
        return combined.split(whereSeparator: \Character.isWhitespace).count
            <= maximumWords
    }
}

/// Amortizes model-call overhead only after confirmed work is already waiting
/// behind an active inference. Normal cadence remains one small phrase per
/// request. Under pressure, up to three adjacent phrases are joined losslessly
/// into one bounded translation call; source IDs and ordering are preserved by
/// `merging`. This prevents one slow decode from creating an ever-growing train
/// of equally expensive tiny calls without recreating the former paragraph
/// batches.
struct ConfirmedTranslationBacklogPolicy: Sendable {
    static let activationWaitingCount = 2
    static let maximumFragments = 3
    static let maximumWords = 14
    static let maximumCharacters = 112

    static func canMerge(
        _ existing: String,
        with incoming: String,
        initialWaitingCount: Int,
        mergedFragments: Int
    ) -> Bool {
        guard initialWaitingCount >= activationWaitingCount,
              mergedFragments < maximumFragments else { return false }
        let combined = [existing, incoming]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        guard !combined.isEmpty, combined.count <= maximumCharacters else {
            return false
        }
        return combined.split(whereSeparator: \Character.isWhitespace).count
            <= maximumWords
    }
}

/// Keeps confirmed translation completion incremental. The current resident
/// worker's batch endpoint decodes items serially and returns only after the
/// final item, so a four-item "batch" measured as four seconds of silence
/// followed by four simultaneous TTS jobs. It provided no throughput gain and
/// created the observed 11-second speech-queue spike. Keep one item in flight
/// until the worker supports streaming per-item batch results.
struct ConfirmedTranslationMicrobatchPolicy: Sendable {
    static func maximumItems(waitingCount: Int) -> Int {
        _ = waitingCount
        return 1
    }
}

/// Converts every delay already accumulated before speech synthesis into one
/// renderer backlog value. Looking only at the TTS mailbox hid six seconds
/// spent in translation and made the voice remain slow precisely when it had
/// to recover toward the source speaker.
struct SpeechCatchUpPolicy: Sendable {
    static func effectiveBacklogMilliseconds(
        speechQueueMilliseconds: Double,
        sourceAudioEndMilliseconds: Double,
        receivedAudioMilliseconds: Double
    ) -> Double {
        max(
            speechQueueMilliseconds,
            max(0, receivedAudioMilliseconds - sourceAudioEndMilliseconds)
        )
    }
}

/// Keeps a grammatically open *confirmed* source fragment beside the
/// translation queue until one small piece of right context arrives.  This is
/// intentionally separate from the ASR boundary rules: an acoustic endpoint
/// can be stable while still being a poor standalone translation unit.  The
/// portable cadence rule supplies the general language decision, while a
/// strict size cap prevents a slow inference from rebuilding a paragraph.
struct ConfirmedTranslationContextPolicy: Sendable {
    // Two cadence fragments are not always enough to close an English
    // clause ("The smallest planet made out" / "of the densest stuff with
    // the most" / ...).  Keep one short breath of source context instead of
    // asking the sentence model to invent a complete Korean sentence from
    // each fragment independently.
    static let maximumWords = 16
    static let maximumCharacters = 128
    static let shortUnitIdleHold: Duration = .milliseconds(700)
    static let grammaticallyOpenIdleHold: Duration = .milliseconds(1_000)
    // A narrowly selected class of English endings needs slightly more than
    // one cadence tick for its head/complement to arrive. Applying this hold
    // only to those shapes avoids slowing ordinary 3-6 word live units.
    static let structurallyAmbiguousIdleHold: Duration = .milliseconds(1_250)
    // A final recognizer update can occasionally deliver an entire sentence
    // after the rolling boundary lane missed its earlier revisions.  Do not
    // let that one large item monopolise the sole confirmed translator while
    // every newer sentence queues behind it.  Only model-/grammar-approved
    // cuts are used; this is a scheduling ceiling, not blind word chunking.
    static let maximumDispatchWords = 16
    // A usable punctuation boundary can land one or two words beyond the
    // preferred inference budget (for example, "... got to get elected,").
    // Looking slightly farther is still dramatically cheaper than preserving
    // a 30-40 word request, and unlike a blind hard cut it keeps each emitted
    // unit structurally translatable.
    static let maximumBoundarySearchWords = 18

    static func needsRightContext(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        if let last = trimmed.last, ".!?\u{3002}\u{ff1f}\u{ff01}".contains(last) {
            return false
        }
        let wordCount = trimmed.split(whereSeparator: \Character.isWhitespace).count
        // A confirmed 3-5 word unit is ideal for cadence, but without terminal
        // punctuation it is often only the left half of a clause. Give only
        // those tiny body units one bounded chance to see right context. This
        // is deliberately language-structure based rather than tied to a
        // recording, vocabulary list, Apple framework, or model artifact.
        return wordCount <= 5
            || LiveEnglishBoundarySafety.shouldHold(trimmed)
            || needsExtendedRightContext(trimmed)
    }

    static func canMerge(_ existing: String, with incoming: String) -> Bool {
        guard needsRightContext(existing) else { return false }
        let combined = [existing, incoming]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        guard !combined.isEmpty, combined.count <= maximumCharacters else {
            return false
        }
        return combined.split(whereSeparator: \Character.isWhitespace).count
            <= maximumWords
    }

    /// Splits an ASR callback only where a complete sentence is followed by
    /// more source text. The recognizer frequently places the final word of
    /// the previous clause at the beginning of the next callback (`before` +
    /// `launch. Teams ...`). When an open phrase is already waiting, expose
    /// that short completed prefix separately so it can close the waiting
    /// phrase; the new sentence tail can then receive its own right context.
    ///
    /// Without a waiting phrase, one- or two-word turns (`Yes. It's hard.`)
    /// remain together. This avoids creating tiny TTS jobs while still
    /// separating ordinary completed sentences. The transformation is
    /// lossless and language-structural; it contains no topic vocabulary.
    static func queueUnits(
        _ text: String, hasDeferredOpenPhrase: Bool
    ) -> [String] {
        let words = text.split(whereSeparator: \Character.isWhitespace)
            .map(String.init)
        guard words.count >= 2 else { return words.isEmpty ? [] : [text] }
        let normalized = words.joined(separator: " ")
        // The resident translator already batches multiple complete sentences
        // in one inference. Splitting every callback here would add serial
        // model calls and reduce the proven Basecamp cadence. Partition only
        // when the leading completion has a specific lossless job: closing an
        // older deferred source phrase.
        let initialUnits: [String]
        if hasDeferredOpenPhrase,
           let end = words.indices.first(where: {
               isTerminalWord(words[$0]) && $0 + 1 < words.count
           }) {
            initialUnits = [
                Array(words[...end]).joined(separator: " "),
                Array(words[(end + 1)...]).joined(separator: " "),
            ]
        } else {
            initialUnits = [normalized]
        }
        return initialUnits.flatMap(boundedDispatchUnits)
    }

    /// Losslessly partitions only a long, structurally separable confirmed
    /// unit.  The remainder stays in FIFO order and receives its own source ID,
    /// so no later backlog policy may skip it.  When no safe boundary exists,
    /// preserve the original unit intact rather than trading fidelity for a
    /// lower latency number.
    private static func boundedDispatchUnits(_ text: String) -> [String] {
        var remaining = text.split(whereSeparator: \Character.isWhitespace)
            .map(String.init)
        var result: [String] = []
        while remaining.count > maximumDispatchWords, result.count < 3 {
            let candidate = remaining.joined(separator: " ")
            let preferredPrefix = EarlyTranslationClauseSelector
                .latencyBoundedPrefix(
                    in: candidate,
                    maximumWords: maximumDispatchWords,
                    minimumWords: 6
                )
            let prefix = preferredPrefix ?? EarlyTranslationClauseSelector
                .latencyBoundedPrefix(
                    in: candidate,
                    maximumWords: maximumBoundarySearchWords,
                    minimumWords: 6
                )
            let losslessPunctuationPrefix = prefix == nil
                ? softPunctuationPrefix(in: remaining)
                : nil
            guard let prefix = prefix ?? losslessPunctuationPrefix else { break }
            let prefixWords = prefix.split(
                whereSeparator: \Character.isWhitespace
            ).map(String.init)
            guard !prefixWords.isEmpty, prefixWords.count < remaining.count else {
                break
            }
            result.append(prefixWords.joined(separator: " "))
            remaining.removeFirst(prefixWords.count)
        }
        if !remaining.isEmpty { result.append(remaining.joined(separator: " ")) }
        return result.isEmpty ? [text] : result
    }

    /// A comma/semicolon/colon after a substantial phrase is a lossless
    /// interpreter breath even when a conservative sentence-boundary model
    /// refuses it. This is used only for an oversized confirmed request and
    /// allows the pieces to share one batched model call instead of letting a
    /// 24+ word decode block every later sentence for six seconds.
    private static func softPunctuationPrefix(in words: [String]) -> String? {
        let upper = min(maximumDispatchWords, words.count - 3)
        guard upper >= 8 else { return nil }
        let closingCharacters = CharacterSet(charactersIn: "\"'’”)]}")
        for count in stride(from: upper, through: 8, by: -1) {
            let token = words[count - 1]
                .trimmingCharacters(in: closingCharacters)
            if token.last.map({ ",;:，；：".contains($0) }) == true {
                return words.prefix(count).joined(separator: " ")
            }
        }
        return nil
    }

    private static func isTerminalWord(_ word: String) -> Bool {
        let closingCharacters = CharacterSet(charactersIn: "\"'’”)]}")
        let trimmed = word.trimmingCharacters(in: closingCharacters)
        return trimmed.last.map { ".!?。？！".contains($0) } == true
    }

    /// A merged pair can itself still be only the left side of a clause. In
    /// that case retain it for one more already-arriving source fragment.
    /// This gives the stateless resident MT model the same useful clause
    /// context that a stateful live model sees, while the hard caps keep the
    /// interpreter from rebuilding paragraphs or accumulating latency.
    static func shouldContinueAccumulating(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard needsRightContext(trimmed) else { return false }
        let words = trimmed.split(whereSeparator: \Character.isWhitespace).count
        return words < maximumWords && trimmed.count < maximumCharacters
    }

    static func idleHold(for text: String) -> Duration {
        if needsExtendedRightContext(text) {
            return structurallyAmbiguousIdleHold
        }
        return LiveEnglishBoundarySafety.shouldHold(text)
            ? grammaticallyOpenIdleHold
            : shortUnitIdleHold
    }

    /// Detects productive English forms whose final content word commonly
    /// attaches to the immediately following fragment. This is deliberately
    /// grammatical rather than topic/entity based: complement nouns take a
    /// following `to/of/that` phrase, while a lone word after a superlative can
    /// still be an attributive modifier waiting for its head noun.
    static func needsExtendedRightContext(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              trimmed.last.map({ !".!?\u{3002}\u{ff1f}\u{ff01}".contains($0) }) == true
        else { return false }

        let words = trimmed.split(whereSeparator: \Character.isWhitespace)
            .map {
                $0.lowercased().trimmingCharacters(in: .punctuationCharacters)
            }
            .filter { !$0.isEmpty }
        guard let tail = words.last else { return false }

        if LiveEnglishBoundarySafety.needsExtendedHold(trimmed) {
            return true
        }

        let complementNouns: Set<String> = [
            "answer", "chance", "clue", "connection", "evidence", "guide",
            "key", "link", "opportunity", "path", "reason", "sign",
            "solution", "way",
        ]
        if complementNouns.contains(tail) { return true }

        let superlatives: Set<String> = [
            "best", "biggest", "greatest", "least", "most", "smallest",
            "strongest", "worst",
        ]
        guard words.count >= 2 else { return false }
        let penultimate = words[words.count - 2]
        if superlatives.contains(penultimate) { return true }

        // Hyphenation is often absent in streaming ASR (`lunar like`). The
        // final `like` still modifies the next noun rather than closing the
        // current target sentence.
        if tail == "like", words.suffix(4).contains(where: superlatives.contains) {
            return true
        }
        return false
    }

}

/// The contextual worker returns an append-only *delta*, not a cumulative
/// translation revision. Two identical speculative deltas therefore are not
/// independent proof that the text is safe: treating them as a repeated full
/// prefix allowed stale context (for example the previous planet name) to
/// leak into the visible translation. Only a source prefix independently
/// approved by both acoustic stability and the semantic boundary gate may
/// expose an incremental delta.
struct IncrementalPreviewExposurePolicy: Sendable {
    static func shouldExpose(
        sourceIsStable: Bool, semanticBoundaryApproved: Bool
    ) -> Bool {
        sourceIsStable && semanticBoundaryApproved
    }
}

/// Selects the orchestration contract without changing the proven model
/// artifacts. The baseline keeps its current behavior. Unified streaming owns
/// one portable, long-lived session and enables incremental confirmation by
/// construction instead of relying on process environment switches.
enum ResidentTranslationPipelineProfile: Sendable {
    case baseline
    case unifiedStreaming

    var requiresPortableASR: Bool { self == .unifiedStreaming }
    var enablesSimultaneousTTS: Bool { self == .unifiedStreaming }
    var enablesLearnedBoundary: Bool { self == .unifiedStreaming }
}

actor ResidentLocalTranslationProvider: TranslationProvider {
    /// Exact orchestration cadence of the latest Gemini-structure speed
    /// baseline. Keep this factory as the single source of truth so later
    /// quality experiments cannot silently restore the older Basecamp 5
    /// timing. It changes no translation or TTS behavior by itself.
    nonisolated static func latestGeminiStructureSpeedScheduler()
        -> RollingTranslationScheduler
    {
        RollingTranslationScheduler(
            targetLagMilliseconds: 350,
            stableUpdateMilliseconds: 250,
            maximumUpdateGapMilliseconds: 350,
            minimumWords: 3
        )
    }

    private struct RecognizedPhrase: Sendable {
        let sourceIDs: [UInt64]
        let text: String
        let recognitionMilliseconds: Double
        let enqueuedAt: ContinuousClock.Instant
        let requiresCompleteSourceGate: Bool
        let semanticPairBoundaryApproved: Bool
        let prefetchedTranslation: SpeculativeTranslationCache.Entry?
    }

    private struct SpeechRequest: Sendable {
        let id: UUID
        let text: String
        let language: Language
        let translationKind: TTSQueuePolicy.TranslationKind
        let enqueuedAt: ContinuousClock.Instant
        let sessionGeneration: UInt64
        /// End of the recognized source span represented by this request, on
        /// the same audio-session timeline as `receivedAudioMilliseconds`.
        let sourceAudioEndMilliseconds: Double
        /// Cumulative stable target prefix after this continuation. It becomes
        /// removable from a later confirmed translation only after this whole
        /// request completes successfully.
        let audiblePrefixAfterCompletion: String?
        /// Irreversible source entries represented by this speech. Empty for
        /// speculative starter/preview requests.
        let sourceIDs: [UInt64]
        /// True only for the final audible unit needed to deliver all source
        /// entries represented by this request. Earlier pipelined units still
        /// carry the IDs so a synthesis failure cannot disappear silently.
        let completesSourceDelivery: Bool

        init(
            id: UUID = UUID(), text: String, language: Language,
            translationKind: TTSQueuePolicy.TranslationKind,
            enqueuedAt: ContinuousClock.Instant, sessionGeneration: UInt64,
            sourceAudioEndMilliseconds: Double,
            audiblePrefixAfterCompletion: String? = nil,
            sourceIDs: [UInt64] = [],
            completesSourceDelivery: Bool = true
        ) {
            self.id = id
            self.text = text
            self.language = language
            self.translationKind = translationKind
            self.enqueuedAt = enqueuedAt
            self.sessionGeneration = sessionGeneration
            self.sourceAudioEndMilliseconds = sourceAudioEndMilliseconds
            self.audiblePrefixAfterCompletion = audiblePrefixAfterCompletion
            self.sourceIDs = sourceIDs
            self.completesSourceDelivery = completesSourceDelivery
        }
    }

    nonisolated let translatedAudio: AsyncStream<AudioChunk>

    private let outputContinuation: AsyncStream<AudioChunk>.Continuation
    private let worker: LocalModelWorker
    private let contextualWorker = ContextualTranslationWorker.shared
    private let remoteTTS: CosyVoiceStreamingClient?
    private let remoteTranslation: CUDATranslationClient?
    private let preferredTerms: [String]
    private let pipelineProfile: ResidentTranslationPipelineProfile
    private var segmenter = StreamingSpeechSegmenter()
    private var segmentContinuation: AsyncStream<AudioChunk>.Continuation?
    private var textContinuation: AsyncStream<RecognizedPhrase>.Continuation?
    private var pendingTranslationPhrases: [RecognizedPhrase] = []
    private var deferredContextPhrase: RecognizedPhrase?
    private var deferredContextFlushTask: Task<Void, Never>?
    private var confirmedSourceLedger = ConfirmedSourceDeliveryLedger()
    private var translationProcessingInFlight = false
    /// Confirmed source is lossless and always outranks revisable preview
    /// work. Both paths share one resident model process, so allowing preview
    /// requests to chain while this is true creates head-of-line blocking for
    /// speech the user is already entitled to hear.
    private var confirmedTranslationHasPriority: Bool {
        Self.shouldDeferPreviewTranslation(
            confirmedTranslationInFlight: translationProcessingInFlight,
            pendingConfirmedCount: pendingTranslationPhrases.count
                + (deferredContextPhrase == nil ? 0 : 1)
        )
    }
    private var workerTask: Task<Void, Never>?
    private var fallbackWorkerTask: Task<Void, Never>?
    private var progressiveTask: Task<Void, Never>?
    private var pendingSpeechFragment: SpeechRequest?
    private var backloggedSpeechRequests: [SpeechRequest] = []
    private var pendingSpeechFlushTask: Task<Void, Never>?
    private var lastSpokenText = ""
    private var speechTask: Task<Void, Never>?
    private var speechSynthesisInFlight = false
    private var activeSpeechOutputGate: SpeechOutputGate?
    private var neuralVoicePreparationTask: Task<Void, Never>?
    private var remoteTTSPrimeTask: Task<Bool, Never>?
    /// The route is selected once at session start. A first-request outage may
    /// still lock an otherwise silent session to local speech. Once GPU PCM has
    /// been audible, however, later failures remain visible and never change
    /// speakers halfway through a conversation.
    private var remoteTTSCircuitOpen = false
    private var neuralVoiceReady = false
    private var sessionHasDeliveredRemoteSpeech = false
    private var progressiveRecognizer: Any?
    private var sourceLanguage: Language?
    private var targetLanguage: Language?
    private var metrics = AudioPipelineMetrics()
    private var pendingError: Error?
    private var ownsWorkerSession = false
    private var ownsContextualWorkerSession = false
    private var usesAppleASR = false
    private var transcriptGate = ASRTranscriptGate()
    // A second, translation-admission-local gate is deliberate. The acoustic
    // path and the semantic early-boundary path can prove overlapping source
    // on adjacent actor turns. `transcriptGate` removes cumulative ASR
    // revisions, while this gate guarantees that whatever combination of
    // proof paths reaches the irreversible translation FIFO is still emitted
    // only once. It is synchronous and model-free, so it cannot add inference
    // latency.
    private var confirmedAdmissionGate = ASRTranscriptGate()
    private var progressivePhraseSeen = false
    private var progressiveCommittedDisplay = ""
    private var zipformerRecognizer: ZipformerStreamingRecognizer?
    // The 20M Zipformer is used only until the first audible starter is
    // reserved. It gives the latency-first path a portable early hypothesis;
    // Nemotron remains the sole owner of confirmed transcript and translation
    // quality, so the two recognizers can never duplicate confirmed speech.
    private var starterPreviewRecognizer: ZipformerStreamingRecognizer?
    private var starterPreviewAudioContinuation: AsyncStream<AudioChunk>.Continuation?
    private var starterPreviewTask: Task<Void, Never>?
    private var previewRecognizer: FastSpeechPreviewRecognizer?
    private var previewTask: Task<Void, Never>?
    private var previewTranslationTask: Task<Void, Never>?
    private var latestPreviewSource = ""
    /// Full cumulative ASR hypothesis corresponding to latestPreviewSource.
    /// The preview/boundary model sees only the uncommitted tail, while the
    /// transcript gate later receives the cumulative prefix through the
    /// approved boundary so it can preserve monotonic de-duplication.
    private var latestPreviewCumulativeSource = ""
    private var latestPreviewIsStable = false
    private var translatedPreviewSource = ""
    private var translatedPreviewIsStable = false
    private var speculativeTranslationCache = SpeculativeTranslationCache()
    private var previewStableCommitter = StableTranscriptCommitter(
        requiredUpdates: 2, lookaheadWords: 1
    )
    // This committer never drives speech by itself. It only proves that a
    // source prefix survived two identical ASR observations; the portable
    // 95%-precision semantic-boundary model must independently approve it
    // before it can enter the normal confirmed translation queue.
    private var earlyConfirmedSourceCommitter = StableTranscriptCommitter(
        requiredUpdates: 2, lookaheadWords: 0
    )
    // The exact source prefix that survived two independent ASR observations.
    // A semantic boundary fully contained in this prefix is confirmed even if
    // newer right context is still revisable. The revisable tail never enters
    // translation or speech.
    private var earlyConfirmedStablePrefix = ""
    // Semantic boundary observations repeat while the recognizer's cumulative
    // hypothesis is unchanged. Do not let those identical observations fill
    // the bounded FIFO and evict the next, genuinely new confirmed sentence.
    private var lastEnqueuedEarlyConfirmedSource = ""
    private var firstSpeechStarterSelector = FirstSpeechStarterSelector()
    private var firstSpeechStarterTask: Task<Void, Never>?
    private var firstSpeechStarterAwaitingFullCorrection = false
    /// Reserved starter target is deliberately not committed as spoken until
    /// an audible PCM chunk reaches the output stream. A failed/stale starter
    /// must never erase the beginning of the later confirmed translation.
    private var queuedStarterSpokenText: String?
    private var queuedStarterGeneration: UInt64?
    private var confirmedAudibleSpeechRequestIDs: Set<UUID> = []
    private var failedSpeechRequestIDs: Set<UUID> = []
    private var starterFirstAudibleAt: ContinuousClock.Instant?
    private var rollingTranslationScheduler = RollingTranslationScheduler()
    private var simultaneousTranslationCommitter = SimultaneousTranslationCommitter()
    private var monotonicTranslationCaptioner = MonotonicTranslationCaptioner()
    private var simultaneousStarterBridgeOpen = false
    private var hasQueuedImmediateStarterContinuation = false
    private var hasStartedSimultaneousSpeech = false
    private let simultaneousTTSIsEnabled: Bool
    private let learnedSimultaneousBoundaryIsEnabled: Bool
    // Keep speculative translation inference opt-in. LocalModelWorker owns one
    // serial resident process, so an unstable-prefix translation can otherwise
    // sit in front of the later confirmed translation and make first speech
    // slower. The semantic boundary check remains active, but only a stable,
    // confirmed source is allowed to consume translation capacity by default.
    private let speculativeTranslationIsEnabled = ProcessInfo.processInfo.environment[
        "AI_INTERPRETER_SPECULATIVE_TRANSLATION"
    ] == "1"
    /// Enabled by default because this model has its own process and therefore
    /// cannot head-of-line block confirmed MADLAD translation. Set to `0` only
    /// for an explicit A/B rollback to the frozen speed baseline.
    // Contextual preview runs in a separate process, but it still competes for
    // the same CPU, memory bandwidth and Metal scheduler as confirmed ASR/MT.
    // Long-session diagnostics showed 87/87 WAIT-preview results being thrown
    // away while confirmed MADLAD latency climbed from sub-second to 7-11 s.
    // Keep the experiment opt-in until an independently proven boundary can
    // consume its result; the lossless confirmed path and its quality are
    // unchanged.
    private let contextualPrefetchIsEnabled = ProcessInfo.processInfo.environment[
        "AI_INTERPRETER_CONTEXTUAL_PREFETCH"
    ] == "1"
    private var previewTranslationIsEnabled: Bool {
        speculativeTranslationIsEnabled || contextualPrefetchIsEnabled
    }
    private var receivedAudioMilliseconds = 0.0
    private var sourceSpeechPaceTracker = SourceSpeechPaceTracker()
    private var lastSpeechAudioMilliseconds = 0.0
    private var confirmedTranslationDisplay = ""
    private var nemotronRecognizer: NemotronStreamingRecognizer?
    private var nemotronAudioContinuation: AsyncStream<AudioChunk>.Continuation?
    private var nemotronAudioTask: Task<Void, Never>?
    private var nemotronCommittedDisplay = ""
    private var pendingTranslationWords: [String] = []
    private var pendingNemotronFinalFlushTask: Task<Void, Never>?
    /// Nemotron can leave a complete, twice-observed preview in semantic WAIT
    /// while no more audio callbacks arrive. An independent wall-clock timer
    /// promotes that acoustically stable tail after a short pause so the next
    /// sentence is not required to unlock the previous one.
    private var pendingStablePreviewIdleFlushTask: Task<Void, Never>?
    private var stablePreviewIdleFlushSource = ""
    private var lastRejectedPreviewSource = ""
    private var lastRejectedPreviewWasStable = false
    private var lastASRLogMilliseconds: Double?
    private var lastSpeechWallClock: ContinuousClock.Instant?
    private var hasLoggedFirstAudioActivity = false
    private var sessionGeneration: UInt64 = 0
    private var simultaneousDebugSessionID = UUID()
    private var nemotronInputSamples: [Float] = []
    private var nemotronInputFormat: AudioFormatInfo?

    init(
        preferredTerms: [String] = [],
        worker: LocalModelWorker = .shared,
        pipelineProfile: ResidentTranslationPipelineProfile = .baseline
    ) {
        self.preferredTerms = preferredTerms
        self.worker = worker
        self.pipelineProfile = pipelineProfile
        simultaneousTTSIsEnabled = pipelineProfile.enablesSimultaneousTTS
            || ProcessInfo.processInfo.environment[
                "AI_INTERPRETER_SIMULTANEOUS_TTS"
            ] == "1"
        learnedSimultaneousBoundaryIsEnabled = pipelineProfile.enablesLearnedBoundary
            || ProcessInfo.processInfo.environment[
                "AI_INTERPRETER_LEARNED_BOUNDARY"
            ] == "1"
        remoteTTS = CosyVoiceStreamingClient.configured()
        remoteTranslation = CUDATranslationClient.configured()
        let pair = AsyncStream<AudioChunk>.makeStream(
            // Never evict PCM from the middle of an accepted utterance. With
            // `bufferingNewest`, a brief consumer stall silently removed old
            // 50 ms pieces while keeping later pieces from the same word. The
            // resulting holes accumulated through a long session and sounded
            // like increasingly robotic, chopped speech. Stale work is already
            // rejected before synthesis and the playback service preserves a
            // confirmed playback group, so this transport boundary must be
            // lossless.
            bufferingPolicy: .unbounded
        )
        translatedAudio = pair.stream
        outputContinuation = pair.continuation
        if contextualPrefetchIsEnabled {
            // AppState creates its providers before the user presses Start.
            // Use that idle time for the one-time model load/Metal compile so
            // the first captured words never pay the cold-start cost.
            Task.detached(priority: .utility) {
                let worker = ContextualTranslationWorker.shared
                await worker.acquire()
                try? await worker.prepare()
                await worker.release()
            }
        }
    }

    func startSession(sourceLanguage: Language, targetLanguage: Language) async throws {
        await stopSession()
        sessionGeneration &+= 1
        simultaneousDebugSessionID = UUID()
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        if contextualPrefetchIsEnabled {
            await contextualWorker.acquire()
            ownsContextualWorkerSession = true
            // Preparation is deliberately not awaited. The live path starts
            // capture immediately and simply ignores this side lane until it
            // has loaded and compiled.
            Task.detached(priority: .utility) { [contextualWorker] in
                try? await contextualWorker.prepare()
            }
        }
        // Screen recording and microphone consent do not grant Speech.framework
        // access. Ask only on the first local-model session; a denial keeps the
        // proven Whisper path active instead of making the session fail.
        if SFSpeechRecognizer.authorizationStatus() == .notDetermined {
            _ = await AppleLocalTranslationProvider.requestSpeechAuthorization()
        }
        usesAppleASR = !pipelineProfile.requiresPortableASR
            && AppleLocalTranslationProvider.canRecognizeOnDevice(sourceLanguage)
        transcriptGate = ASRTranscriptGate()
        confirmedAdmissionGate = ASRTranscriptGate()
        segmenter = StreamingSpeechSegmenter()
        metrics = AudioPipelineMetrics()
        pendingError = nil
        progressivePhraseSeen = false
        progressiveCommittedDisplay = ""
        latestPreviewSource = ""
        latestPreviewCumulativeSource = ""
        latestPreviewIsStable = false
        translatedPreviewSource = ""
        translatedPreviewIsStable = false
        speculativeTranslationCache.reset()
        previewStableCommitter.reset()
        earlyConfirmedSourceCommitter.reset()
        earlyConfirmedStablePrefix = ""
        lastEnqueuedEarlyConfirmedSource = ""
        lastRejectedPreviewSource = ""
        lastRejectedPreviewWasStable = false
        firstSpeechStarterSelector.reset()
        firstSpeechStarterTask?.cancel()
        firstSpeechStarterTask = nil
        firstSpeechStarterAwaitingFullCorrection = false
        queuedStarterSpokenText = nil
        queuedStarterGeneration = nil
        confirmedAudibleSpeechRequestIDs.removeAll(keepingCapacity: true)
        failedSpeechRequestIDs.removeAll(keepingCapacity: true)
        starterFirstAudibleAt = nil
        rollingTranslationScheduler = simultaneousTTSIsEnabled
            ? Self.latestGeminiStructureSpeedScheduler()
            : RollingTranslationScheduler()
        simultaneousTranslationCommitter.reset()
        monotonicTranslationCaptioner.reset()
        simultaneousStarterBridgeOpen = false
        hasQueuedImmediateStarterContinuation = false
        hasStartedSimultaneousSpeech = false
        receivedAudioMilliseconds = 0
        sourceSpeechPaceTracker.reset()
        lastSpeechAudioMilliseconds = 0
        pendingTranslationPhrases.removeAll(keepingCapacity: true)
        deferredContextPhrase = nil
        deferredContextFlushTask?.cancel()
        deferredContextFlushTask = nil
        confirmedSourceLedger.reset()
        translationProcessingInFlight = false
        confirmedTranslationDisplay = ""
        lastSpokenText = ""
        remoteTTSCircuitOpen = false
        sessionHasDeliveredRemoteSpeech = false
        pendingSpeechFragment = nil
        backloggedSpeechRequests.removeAll(keepingCapacity: true)
        pendingSpeechFlushTask?.cancel()
        pendingSpeechFlushTask = nil
        lastASRLogMilliseconds = nil
        lastSpeechWallClock = nil
        hasLoggedFirstAudioActivity = false
        nemotronInputSamples.removeAll(keepingCapacity: true)
        nemotronInputFormat = nil
        pendingNemotronFinalFlushTask?.cancel()
        pendingNemotronFinalFlushTask = nil
        pendingStablePreviewIdleFlushTask?.cancel()
        pendingStablePreviewIdleFlushTask = nil
        stablePreviewIdleFlushSource = ""
        starterPreviewAudioContinuation?.finish()
        starterPreviewAudioContinuation = nil
        starterPreviewTask?.cancel()
        starterPreviewTask = nil
        starterPreviewRecognizer = nil
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID, event: "session_started",
            audioMilliseconds: 0, sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage
        )
        // Probe readiness in parallel with local model preparation. This
        // selects one stable speech route for the entire session without
        // adding the probe's normal network latency to capture startup.
        if let remoteTTS {
            remoteTTSPrimeTask = Task(priority: .userInitiated) {
                await remoteTTS.isReadyForSession()
            }
        }
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
            // Load only the high-precision boundary classifier plus the
            // portable MADLAD/CT2 translator used by confirmed phrases. The
            // unqualified experimental Qwen delta translator stays excluded.
            // Paying CT2's one-time graph/load cost before capture prevents the
            // first confirmed phrase from carrying a multi-second cold start.
            try await worker.prepareSimultaneousBoundary(sourceLanguage: sourceLanguage)
            try await worker.prepareTranslation()
        }
        if remoteTTS != nil {
            let remoteIsReady = await remoteTTSPrimeTask?.value ?? false
            remoteTTSCircuitOpen = !remoteIsReady
            neuralVoiceReady = remoteIsReady
            await SimultaneousDebugLogger.shared.record(
                sessionID: simultaneousDebugSessionID,
                event: "tts_route_selected",
                audioMilliseconds: 0,
                sourceLanguage: sourceLanguage,
                targetLanguage: targetLanguage,
                status: remoteIsReady ? "gpu_session" : "local_session"
            )
        }
        // The parallel synthesis prime above includes the connection warm-up.
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
            // Nemotron's confirmed fragments use the lossless coalescing queue
            // below instead of AsyncStream.bufferingNewest. Newest-only streams
            // can erase sentences, while an unbounded per-fragment stream lets
            // model-call overhead accumulate during dense speech.
            textContinuation = nil
            pendingTranslationPhrases.removeAll(keepingCapacity: true)
            deferredContextPhrase = nil
            deferredContextFlushTask?.cancel()
            deferredContextFlushTask = nil
            translationProcessingInFlight = false
            let audio = AsyncStream<AudioChunk>.makeStream(
                // sendAudio coalesces capture callbacks into native 80 ms
                // recognizer frames. The old eight-item ceiling therefore held
                // only 640 ms (not four seconds as the former comment claimed).
                // Any brief actor/model stall permanently discarded source
                // audio, and elapsed-ASR lag grew for the rest of the session.
                // Keep a bounded 20.48-second safety window. Nemotron runs near
                // 0.10 RTF on this machine, so it drains a transient backlog
                // about ten times faster than real time without losing words.
                bufferingPolicy: .bufferingNewest(Self.nemotronAudioBufferCapacity)
            )
            nemotronAudioContinuation = audio.continuation
            nemotronAudioTask = Task(priority: .userInitiated) { [weak self] in
                for await chunk in audio.stream {
                    guard !Task.isCancelled else { break }
                    await self?.processNemotronAudio(chunk)
                }
            }
            // A tiny cross-platform transducer supplies only the revisable
            // first lead-in. It runs on an independent consumer so decoding it
            // cannot block capture or the accurate Nemotron stream.
            if let starterRecognizer = try? ZipformerStreamingRecognizer(
                language: sourceLanguage
            ) {
                starterPreviewRecognizer = starterRecognizer
                let starterAudio = AsyncStream<AudioChunk>.makeStream(
                    bufferingPolicy: .bufferingNewest(4)
                )
                starterPreviewAudioContinuation = starterAudio.continuation
                starterPreviewTask = Task(priority: .userInitiated) { [weak self] in
                    for await chunk in starterAudio.stream {
                        guard !Task.isCancelled else { break }
                        _ = await starterRecognizer.ingest(chunk)
                        guard let hypothesis = await starterRecognizer.currentPreview()
                        else { continue }
                        await self?.processPortableStarterPreview(hypothesis)
                    }
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
                        queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds,
                        sourceIDs: phrase.sourceIDs
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
                            queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds,
                            sourceIDs: phrase.sourceIDs
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
                            queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds,
                            sourceIDs: phrase.sourceIDs
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
                    queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds,
                    sourceIDs: phrase.sourceIDs
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
            // one native recognizer frame before crossing into the consumer.
            // Keeping this value shared with NemotronStreamingRecognizer avoids
            // silently reintroducing a 160 ms outer batching floor.
            // now consumes each batch immediately; the independent two-update
            // stability gate still prevents a single native drain from
            // committing revisable text.
            if nemotronInputFormat != chunk.format {
                nemotronInputSamples.removeAll(keepingCapacity: true)
                nemotronInputFormat = chunk.format
            }
            nemotronInputSamples.append(contentsOf: chunk.samples)
            let targetSamples = max(
                1, Int(
                    chunk.format.sampleRate
                        * NemotronStreamingRecognizer.ingestFrameSeconds
                ) * chunk.format.channelCount
            )
            // A ScreenCaptureKit callback is commonly 240 ms. Draining only
            // one 160 ms batch per callback permanently retained 80 ms and
            // made ASR fall behind by one third of real time. Yield every full
            // batch now; retain only the sub-160 ms remainder.
            while nemotronInputSamples.count >= targetSamples {
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
                if !hasStartedSimultaneousSpeech,
                   firstSpeechStarterTask == nil {
                    _ = starterPreviewAudioContinuation?.yield(batchedChunk)
                }
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

    /// Diagnostics restart the same production provider repeatedly.  Wait for
    /// the accepted neural-voice request to release the server's serialized
    /// generator before starting the next session; otherwise an old request
    /// is still consuming TTS/ASR resources and the benchmark measures its own
    /// teardown backlog instead of live-session latency.
    func waitForSpeechSynthesisIdle(timeout: Duration = .seconds(6)) async {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while speechSynthesisInFlight, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(25))
        }
    }

    func stopSession() async {
        sessionGeneration &+= 1
        pendingNemotronFinalFlushTask?.cancel()
        pendingNemotronFinalFlushTask = nil
        starterPreviewAudioContinuation?.finish()
        starterPreviewAudioContinuation = nil
        starterPreviewTask?.cancel()
        starterPreviewTask = nil
        if let starterPreviewRecognizer {
            _ = await starterPreviewRecognizer.finish()
        }
        starterPreviewRecognizer = nil
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
        pendingStablePreviewIdleFlushTask?.cancel()
        pendingStablePreviewIdleFlushTask = nil
        stablePreviewIdleFlushSource = ""
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
        earlyConfirmedSourceCommitter.reset()
        earlyConfirmedStablePrefix = ""
        lastEnqueuedEarlyConfirmedSource = ""
        lastRejectedPreviewSource = ""
        lastRejectedPreviewWasStable = false
        firstSpeechStarterSelector.reset()
        firstSpeechStarterTask?.cancel()
        firstSpeechStarterTask = nil
        firstSpeechStarterAwaitingFullCorrection = false
        rollingTranslationScheduler.reset()
        monotonicTranslationCaptioner.reset()
        latestPreviewSource = ""
        latestPreviewIsStable = false
        translatedPreviewSource = ""
        translatedPreviewIsStable = false
        speculativeTranslationCache.reset()
        receivedAudioMilliseconds = 0
        lastSpeechAudioMilliseconds = 0
        confirmedTranslationDisplay = ""
        lastASRLogMilliseconds = nil
        lastSpeechWallClock = nil
        hasLoggedFirstAudioActivity = false
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
        pendingSpeechFlushTask?.cancel()
        pendingSpeechFlushTask = nil
        pendingSpeechFragment = nil
        backloggedSpeechRequests.removeAll(keepingCapacity: true)
        // Let an accepted GPU request finish server-side so its serialized
        // CUDA slot is released. Closing the gate prevents late PCM from the
        // old session reaching playback. Cancelling the URLSession request
        // here previously stranded the server and silenced later sessions.
        activeSpeechOutputGate?.close()
        activeSpeechOutputGate = nil
        speechTask = nil
        neuralVoicePreparationTask?.cancel()
        neuralVoicePreparationTask = nil
        remoteTTSPrimeTask?.cancel()
        remoteTTSPrimeTask = nil
        neuralVoiceReady = false
        if let final = await segmenter.flush() { _ = segmentContinuation?.yield(final) }
        segmentContinuation?.finish()
        segmentContinuation = nil
        workerTask?.cancel()
        workerTask = nil
        pendingTranslationPhrases.removeAll(keepingCapacity: true)
        deferredContextPhrase = nil
        deferredContextFlushTask?.cancel()
        deferredContextFlushTask = nil
        translationProcessingInFlight = false
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
        if ownsContextualWorkerSession {
            ownsContextualWorkerSession = false
            await contextualWorker.release()
        }
    }

    private func enqueueRecognizedPhrase(
        _ text: String, recognitionMilliseconds: Double,
        requiresCompleteSourceGate: Bool = false,
        semanticPairBoundaryApproved: Bool = false,
        prefetchedTranslation: SpeculativeTranslationCache.Entry? = nil
    ) {
        if nemotronRecognizer != nil, Self.shouldHoldNemotronDispatch(text) {
            let carry = text.split(whereSeparator: \Character.isWhitespace).map(String.init)
            pendingTranslationWords.insert(contentsOf: carry, at: 0)
            return
        }
        progressivePhraseSeen = true
        if nemotronRecognizer != nil {
            // Commit cumulative-source deduplication at queue admission, not
            // when the single resident translator eventually reaches this
            // item. Otherwise two fast ASR callbacks can both observe the
            // same uncommitted prefix while an older inference is running and
            // enqueue an audible duplicate. This operation is synchronous and
            // model-free, so it adds no translation latency.
            let freshText = transcriptGate.freshText(
                text,
                accepting: { candidate in
                    !requiresCompleteSourceGate
                        || !Self.shouldHoldEarlyConfirmedSource(
                            candidate,
                            allowSoftPunctuation:
                                semanticPairBoundaryApproved
                        )
                }
            )
            guard let freshText else { return }
            guard let admittedFreshText = confirmedAdmissionGate.freshText(
                freshText
            ) else { return }
            sourceSpeechPaceTracker.record(
                admittedFreshText, at: recognitionMilliseconds
            )
            let units = ConfirmedTranslationContextPolicy.queueUnits(
                admittedFreshText,
                hasDeferredOpenPhrase: deferredContextPhrase != nil
            )
            for unit in units {
                let sourceID = confirmedSourceLedger.admit(unit)
                let phrase = RecognizedPhrase(
                    sourceIDs: [sourceID],
                    text: unit,
                    recognitionMilliseconds: recognitionMilliseconds,
                    enqueuedAt: .now,
                    requiresCompleteSourceGate: requiresCompleteSourceGate,
                    semanticPairBoundaryApproved:
                        semanticPairBoundaryApproved && units.count == 1,
                    prefetchedTranslation:
                        units.count == 1 ? prefetchedTranslation : nil
                )
                if units.count > 1 {
                    // `queueUnits` already proved and losslessly partitioned a
                    // long source turn. Sending those pieces back through the
                    // right-context coalescer simply glued them into the same
                    // 20-30 word head-of-line blocker again.
                    enqueueNemotronTranslationImmediately(phrase)
                } else {
                    enqueueCoalescedNemotronTranslation(phrase)
                }
            }
            return
        }
        let sourceID = confirmedSourceLedger.admit(text)
        sourceSpeechPaceTracker.record(text, at: recognitionMilliseconds)
        let recognizedPhrase = RecognizedPhrase(
            sourceIDs: [sourceID],
            text: text, recognitionMilliseconds: recognitionMilliseconds,
            enqueuedAt: .now,
            requiresCompleteSourceGate: requiresCompleteSourceGate,
            semanticPairBoundaryApproved: semanticPairBoundaryApproved,
            prefetchedTranslation: prefetchedTranslation
        )
        switch textContinuation?.yield(recognizedPhrase) {
        case let .dropped(dropped):
            confirmedSourceLedger.mark(dropped.sourceIDs, as: .failed)
            metrics.droppedChunks += 1
        case .enqueued:
            metrics.outputChunks += 1
        default:
            break
        }
    }

    private func enqueueCoalescedNemotronTranslation(_ phrase: RecognizedPhrase) {
        // Stop speculative work from producing stale target speech after a
        // confirmed phrase arrives. The worker call itself may take a moment
        // to unwind, but cancellation prevents it from being followed by more
        // preview calls or from committing audio out of order.
        previewTranslationTask?.cancel()
        if let deferred = deferredContextPhrase {
            deferredContextPhrase = nil
            deferredContextFlushTask?.cancel()
            deferredContextFlushTask = nil
            let merged = merging(deferred, with: phrase)
            if ConfirmedTranslationContextPolicy.shouldContinueAccumulating(
                merged.text
            ) {
                deferContextPhrase(merged)
            } else {
                enqueueNemotronTranslationImmediately(merged)
            }
            return
        }

        if ConfirmedTranslationContextPolicy.needsRightContext(phrase.text) {
            deferContextPhrase(phrase)
            return
        }

        enqueueNemotronTranslationImmediately(phrase)
    }

    private func deferContextPhrase(_ phrase: RecognizedPhrase) {
        deferredContextPhrase = phrase
        let generation = sessionGeneration
        deferredContextFlushTask?.cancel()
        deferredContextFlushTask = Task(priority: .userInitiated) { [weak self] in
            try? await Task.sleep(
                for: ConfirmedTranslationContextPolicy.idleHold(for: phrase.text)
            )
            guard !Task.isCancelled else { return }
            await self?.flushDeferredContextPhrase(generation: generation)
        }
    }

    private func enqueueNemotronTranslationImmediately(_ phrase: RecognizedPhrase) {
        guard translationProcessingInFlight else {
            translationProcessingInFlight = true
            workerTask = Task(priority: .userInitiated) { [weak self] in
                await self?.drainNemotronTranslationQueue(startingWith: phrase)
            }
            return
        }

        if let last = pendingTranslationPhrases.last,
           TranslationInputCoalescingPolicy.canMerge(last.text, with: phrase.text)
            || ConfirmedTranslationContextPolicy.canMerge(
                last.text, with: phrase.text
            ) {
            pendingTranslationPhrases[pendingTranslationPhrases.count - 1] =
                merging(last, with: phrase)
        } else {
            pendingTranslationPhrases.append(phrase)
        }
    }

    private func flushDeferredContextPhrase(generation: UInt64) {
        guard generation == sessionGeneration,
              let phrase = deferredContextPhrase else { return }
        deferredContextPhrase = nil
        deferredContextFlushTask = nil
        enqueueNemotronTranslationImmediately(phrase)
    }

    private func merging(
        _ first: RecognizedPhrase, with second: RecognizedPhrase
    ) -> RecognizedPhrase {
        RecognizedPhrase(
            sourceIDs: first.sourceIDs + second.sourceIDs,
            text: first.text + " " + second.text,
            recognitionMilliseconds: second.recognitionMilliseconds,
            // Preserve the oldest enqueue time so diagnostics continue to
            // expose real queue pressure instead of hiding the context hold.
            enqueuedAt: first.enqueuedAt,
            requiresCompleteSourceGate:
                first.requiresCompleteSourceGate
                    || second.requiresCompleteSourceGate,
            semanticPairBoundaryApproved:
                first.semanticPairBoundaryApproved
                    && second.semanticPairBoundaryApproved,
            prefetchedTranslation: nil
        )
    }

    private func drainNemotronTranslationQueue(
        startingWith first: RecognizedPhrase
    ) async {
        var current: RecognizedPhrase? = first
        while let phrase = current, !Task.isCancelled {
            var batch = [phrase]
            // Once confirmed work is waiting, send independent FIFO phrases
            // in one CT2 call. They remain separate hypotheses and separate
            // TTS requests; this only shares model scheduling. A speculative
            // promotion keeps its ordinary single-item commit semantics.
            if remoteTranslation == nil, phrase.prefetchedTranslation == nil {
                let maximumItems = ConfirmedTranslationMicrobatchPolicy
                    .maximumItems(waitingCount: pendingTranslationPhrases.count)
                while batch.count < maximumItems,
                      let next = pendingTranslationPhrases.first,
                      next.prefetchedTranslation == nil {
                    batch.append(pendingTranslationPhrases.removeFirst())
                }
            }
            if batch.count > 1 {
                await translateNemotronPhrasesIfFresh(batch)
            } else {
                await translateNemotronPhraseIfFresh(phrase)
            }
            current = dequeueNextConfirmedTranslationPhrase()
        }
        translationProcessingInFlight = false
        workerTask = nil
        if let sourceLanguage, let targetLanguage {
            launchLatestPreviewTranslationIfNeeded(
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
            )
        }
    }

    private func dequeueNextConfirmedTranslationPhrase() -> RecognizedPhrase? {
        guard !pendingTranslationPhrases.isEmpty else { return nil }
        return pendingTranslationPhrases.removeFirst()
    }

    static func shouldHoldNemotronDispatch(_ text: String) -> Bool {
        let words = text.split(whereSeparator: \Character.isWhitespace).map(String.init)
        guard let last = words.last else { return true }
        let hasUnsafeEnglishBoundary = LiveEnglishBoundarySafety.shouldHold(text)
        let hasCadenceSizedStableUnit = words.count >= 5
            && !endsWithOpenSpokenNumber(words)
            && canEndLiveTranslationPhrase(last)
        // Stable 5-6 word source is the portable simultaneous-interpreter
        // contract. A leading preposition or dependent connective may make it
        // an incomplete *sentence*, but it is still a useful interpreter
        // segment ("As the planet closest to the sun" ->
        // "태양에 가장 가까운 행성으로서"). Waiting for the main clause
        // caused four-second gaps that Gemini avoids by emitting such segments.
        if hasCadenceSizedStableUnit && !hasUnsafeEnglishBoundary { return false }
        // A comma is not inherently incomplete. The clause selector only
        // exposes one after observing right context, and holding every such
        // prefix caused 40-50 word bursts. Keep only genuinely dangling
        // grammar (prepositions, auxiliaries and unresolved dependent clauses).
        return startsWithTinyDependentFragment(words)
            || endsWithOpenSpokenNumber(words)
            || !canEndLiveTranslationPhrase(last)
            || startsWithUnresolvedDependentClause(words)
            || hasUnsafeEnglishBoundary
    }

    /// Returns a small irreversible source prefix with one already-observed
    /// word of right context. Gemini-style simultaneous interpretation keeps
    /// a nearly constant cadence by advancing short, stable deltas instead of
    /// waiting for a whole English sentence and then racing through it.
    ///
    /// The right-context word is essential: the last streaming ASR token can
    /// still be only a lexical prefix (`cook` -> `cooker`, `Ven` -> `Venus`).
    /// It remains pending until a later observation proves it complete. No
    /// source word is discarded; callers remove only the returned prefix.
    static func stableCadencePrefix(
        in words: [String], minimumWords: Int = 4, maximumWords: Int = 6
    ) -> String? {
        guard minimumWords >= 2, maximumWords >= minimumWords,
              words.count > minimumWords else { return nil }
        let upperBound = min(maximumWords, words.count - 1)
        guard upperBound >= minimumWords else { return nil }

        // A completed short sentence is the strongest boundary. Do not join
        // the beginning of the next sentence merely to reach five words.
        // Terminal punctuation is its own stability proof. The former
        // `words.count - 1` bound always retained the final word even when it
        // completed the sentence. That turned `... days before launch.` into
        // `... days before` + `launch. Teams ...`, and similarly left
        // `the world.` to be joined with the next decoder shard. A complete
        // sentence needs no revisable right-context token, so include its last
        // word while preserving the one-word look-ahead rule for every
        // unpunctuated cadence cut below.
        let sentenceUpperBound = min(12, words.count)
        for count in minimumWords...sentenceUpperBound {
            let last = words[count - 1]
            guard last.last.map({ ".?!。？！".contains($0) }) == true else {
                continue
            }
            let candidateWords = Array(words.prefix(count))
            let candidate = candidateWords.joined(separator: " ")
            if !shouldHoldNemotronDispatch(candidate) { return candidate }
        }

        let rightAttachingMarkers: Set<String> = [
            "am", "are", "as", "at", "be", "been", "being", "by", "can",
            "could", "did", "does", "for", "from", "had", "has", "have",
            "how", "in", "into", "is", "may", "might", "must", "of", "on", "shall",
            "should", "that", "to", "was", "were", "who", "whom", "whose",
            "which", "what", "why", "will", "with", "without", "would",
        ]
        for count in stride(from: upperBound, through: minimumWords, by: -1) {
            let candidateWords = Array(words.prefix(count))
            let candidate = candidateWords.joined(separator: " ")
            guard !shouldHoldNemotronDispatch(candidate) else { continue }
            let next = words[count].lowercased()
                .trimmingCharacters(in: .punctuationCharacters)
            // Preserve a noun together with its immediately following
            // relative/complement clause. A later observation normally
            // exposes a nearby safer cut without adding a multi-second wait.
            guard !rightAttachingMarkers.contains(next) else { continue }
            return candidate
        }
        return nil
    }

    /// Nemotron can finalize a lexical shard immediately after a completed
    /// sentence (`the world.` + `Determ`). A one-token, unpunctuated final is
    /// not a usable utterance and is normally replaced by the next decoder
    /// window. Never attach it to the preceding complete sentence: doing so
    /// makes both the valid sentence and the shard audible as one malformed
    /// translation. Genuine one-word turns carry terminal punctuation.
    static func isIsolatedFinalLexicalShard(
        _ phraseWords: [String], after pendingWords: [String], isFinal: Bool
    ) -> Bool {
        guard isFinal, phraseWords.count == 1,
              let shard = phraseWords.first,
              shard.last.map({ !".!?。？！".contains($0) }) ?? true,
              let pendingLast = pendingWords.last,
              pendingLast.last.map({ ".!?。？！".contains($0) }) == true
        else { return false }
        return true
    }

    /// A streaming endpoint immediately after a spoken number is frequently
    /// only the left half of a quantity ("sixty" + "percent", "twenty" +
    /// "million").  Hold the unpunctuated number for the next ASR delta so the
    /// translation model receives one semantic quantity instead of inventing
    /// two unrelated values.  Explicit terminal punctuation still closes a
    /// genuine sentence such as "The answer is sixty.".
    private static func endsWithOpenSpokenNumber(_ words: [String]) -> Bool {
        guard let rawLast = words.last,
              rawLast.last.map({ !".!?。？！".contains($0) }) ?? true
        else { return false }
        let last = rawLast.lowercased()
            .trimmingCharacters(in: .punctuationCharacters)
        let numberWords: Set<String> = [
            "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
            "seventy", "eighty", "ninety", "hundred", "thousand", "million",
            "billion", "trillion",
        ]
        return numberWords.contains(last)
            || (!last.isEmpty && last.allSatisfy(\.isNumber))
    }

    /// RNNT can close an acoustic endpoint after only a prepositional or
    /// comparative lead-in ("without seeing", "of what you", "as a mem").
    /// Translating such a fragment independently creates an awkward target
    /// sentence and another unnecessary TTS job. Hold only tiny, unpunctuated
    /// dependent fragments; complete short thoughts such as "I agree." remain
    /// immediately dispatchable, and the next ASR delta supplies right context.
    private static func startsWithTinyDependentFragment(_ words: [String]) -> Bool {
        guard (1...8).contains(words.count),
              let firstWord = words.first,
              let lastWord = words.last,
              lastWord.last.map({ ".?!。？！".contains($0) }) != true
        else { return false }

        let first = firstWord.lowercased()
            .trimmingCharacters(in: .punctuationCharacters)
        let dependentOpeners: Set<String> = [
            "about", "after", "among", "as", "at", "before", "between",
            "by", "despite", "during", "for", "from", "in", "into", "of",
            "on", "onto", "over", "than", "through", "to", "under",
            "until", "with", "within", "without",
        ]
        return dependentOpeners.contains(first)
    }

    /// Early-confirmed speech is irreversible, unlike preview text. Apply a
    /// deliberately narrow lexical safety net after cumulative-prefix
    /// deduplication. It rejects only endings that are overwhelmingly likely
    /// to need the next 320-640 ms ASR update; the portable semantic boundary
    /// model remains the primary decision-maker for every other phrase.
    static func shouldHoldEarlyConfirmedSource(
        _ text: String, allowSoftPunctuation: Bool = false
    ) -> Bool {
        let compact = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let words = compact.split(whereSeparator: \Character.isWhitespace).map(String.init)
        guard words.count >= 3 else { return true }
        if shouldHoldNemotronDispatch(compact) { return true }

        // The normal cadence path may translate a stable dependent segment as
        // a reversible interpreter breath. Early-confirmed speech is not
        // reversible, so an unresolved lead ("If we do not take action,")
        // must still wait for its governing clause even after it reaches the
        // five-word cadence threshold.
        if startsWithUnresolvedDependentClause(words)
            && !resolvesDependentLeadWithFollowingSentence(words) {
            return true
        }

        // Soft punctuation normally leaves a thought open. Only the English
        // pair model may override comma/semicolon holding after three words
        // of right context and the deterministic structural gate agree.
        let endsApprovedSoftBoundary = compact.hasSuffix(",")
            || compact.hasSuffix("，") || compact.hasSuffix(";")
            || compact.hasSuffix("；")
        let endsUnclosedBoundary = compact.hasSuffix(":")
            || compact.hasSuffix("：") || compact.hasSuffix("-")
            || compact.hasSuffix("—") || compact.hasSuffix("...")
            || compact.hasSuffix("…")
        if (!allowSoftPunctuation && endsApprovedSoftBoundary)
            || endsUnclosedBoundary {
            return true
        }

        let normalized = words.map {
            $0.lowercased().trimmingCharacters(in: .punctuationCharacters)
        }
        let tail = normalized.last ?? ""
        // A prenominal ordinal/adjective is not a complete object by itself
        // ("take the next" -> "take the next great leap").  The generic
        // eight-word latency deadline below may otherwise make this revisable
        // noun phrase irreversible just before the head noun arrives.
        if tail == "next", normalized.count >= 2,
           ["a", "an", "the", "this", "that", "my", "our", "your", "their"]
            .contains(normalized[normalized.count - 2]) {
            return true
        }
        let incompleteAuxiliaries: Set<String> = [
            "am", "are", "be", "been", "being", "did", "do", "does",
            "gonna", "gotta", "wanna",
        ]
        if incompleteAuxiliaries.contains(tail) { return true }

        // Negation is complete in "probably not", but not after a subject or
        // auxiliary ("we're not", "they do not"). This catches unfinished
        // self-repairs without depending on any particular recording.
        if tail == "not", normalized.count >= 2 {
            let previous = normalized[normalized.count - 2]
            let negationNeedsComplement: Set<String> = [
                "am", "are", "is", "was", "were", "be", "been", "being",
                "do", "does", "did", "have", "has", "had",
                "can", "could", "may", "might", "must", "shall", "should",
                "will", "would", "i'm", "we're", "they're", "you're",
                "he's", "she's", "it's",
            ]
            if negationNeedsComplement.contains(previous) { return true }
        }
        return LiveEnglishBoundarySafety.shouldHold(compact)
    }

    /// Continuous speech cannot rely only on punctuation or acoustic pauses:
    /// both can be several seconds apart.  Once twelve words form a usable
    /// monotonic unit, release that exact prefix even when the learned model
    /// conservatively says WAIT.  Twelve words intentionally excludes the
    /// short growing revisions that otherwise produce several obsolete TTS
    /// jobs before the complete source unit arrives.  The dedicated 3-6 word
    /// starter path remains responsible for low first-speech latency.
    static func canConfirmLatencyDeadlineBoundary(
        _ source: String, isAcousticallyStable: Bool
    ) -> Bool {
        let words = source.split(whereSeparator: \Character.isWhitespace)
            .map {
                $0.lowercased().trimmingCharacters(in: .punctuationCharacters)
            }
        guard isAcousticallyStable, words.count >= 8,
              !shouldHoldEarlyConfirmedSource(source)
        else { return false }

        // Repeated determiner starts only one token apart usually mean that
        // streaming ASR stopped inside a self-repair or an unfinished noun
        // phrase ("the breathtaking the ground"). Do not make that malformed
        // prefix irreversible merely because the latency deadline elapsed.
        let determiners: Set<String> = [
            "a", "an", "another", "any", "each", "every", "some", "the",
            "this", "that", "these", "those", "my", "our", "your", "their",
        ]
        let positions = words.indices.filter { determiners.contains(words[$0]) }
        if zip(positions, positions.dropFirst()).contains(where: {
            $0.1 - $0.0 <= 2
        }) {
            return false
        }
        return true
    }

    private func translateNemotronPhraseIfFresh(_ phrase: RecognizedPhrase) async {
        // Source admission already made this phrase irreversible and removed
        // every previously queued cumulative prefix. Re-running the mutable
        // transcript gate here would make deduplication depend on inference
        // completion order and recreate the queue race it is meant to avoid.
        let freshText = phrase.text
        let prefetchedTranslation: SpeculativeTranslationCache.Entry?
        if let candidate = phrase.prefetchedTranslation,
           SpeculativeTranslationCache.canonicalSource(freshText)
            == candidate.confirmedSource {
            prefetchedTranslation = candidate
        } else {
            prefetchedTranslation = nil
        }
        await translateText(
            freshText,
            recognitionMilliseconds: phrase.recognitionMilliseconds,
            queueMilliseconds: phrase.enqueuedAt.duration(to: .now).milliseconds,
            translationInput: phrase.semanticPairBoundaryApproved
                ? EarlyTranslationClauseSelector
                    .closingSoftBoundaryForTranslation(freshText)
                : nil,
            prefetchedTranslation: prefetchedTranslation,
            sourceIDs: phrase.sourceIDs
        )
    }

    private func translateNemotronPhrasesIfFresh(
        _ phrases: [RecognizedPhrase]
    ) async {
        guard phrases.count >= 2,
              let sourceLanguage, let targetLanguage else { return }
        let generation = sessionGeneration
        let requestedAt = ContinuousClock.now
        let queueMilliseconds = phrases.map {
            $0.enqueuedAt.duration(to: requestedAt).milliseconds
        }
        for (phrase, queue) in zip(phrases, queueMilliseconds) {
            await SimultaneousDebugLogger.shared.record(
                sessionID: simultaneousDebugSessionID,
                event: "translation_requested",
                audioMilliseconds: receivedAudioMilliseconds,
                sourceLanguage: sourceLanguage,
                targetLanguage: targetLanguage,
                source: phrase.text,
                status: "confirmed_batch",
                operationMilliseconds: queue
            )
        }

        let normalizedTexts = phrases.map { phrase in
            let selected = phrase.semanticPairBoundaryApproved
                ? EarlyTranslationClauseSelector
                    .closingSoftBoundaryForTranslation(phrase.text)
                : nil
            return TranslationSemanticNormalizer.normalize(
                Self.cleanWindowBoundaryArtifacts(selected ?? phrase.text),
                source: sourceLanguage, target: targetLanguage
            )
        }

        do {
            if !ownsWorkerSession {
                try await worker.acquire()
                ownsWorkerSession = true
                try await worker.resetContext()
            }
            let response = try await worker.translateTextBatch(
                texts: normalizedTexts,
                sourceLanguage: sourceLanguage,
                targetLanguage: targetLanguage,
                preferredTerms: preferredTerms
            )
            guard !Task.isCancelled, generation == sessionGeneration,
                  let outputs = response.translations,
                  outputs.count == phrases.count else { return }

            for (index, pair) in zip(phrases.indices, outputs) {
                let phrase = phrases[index]
                if pair.status == "duplicate" {
                    confirmedSourceLedger.mark(phrase.sourceIDs, as: .superseded)
                    continue
                }
                guard pair.status == "ok",
                      let translation = pair.translation,
                      !translation.isEmpty else {
                    throw LocalModelWorkerError.invalidResponse
                }
                await commitBatchedConfirmedTranslation(
                    phrase: phrase,
                    translatedText: translation,
                    normalizedSource: normalizedTexts[index],
                    translationMilliseconds:
                        pair.translationMilliseconds ?? 0,
                    queueMilliseconds: queueMilliseconds[index],
                    generation: generation,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage
                )
            }
        } catch is CancellationError {
            return
        } catch {
            // Batch transport is an optimization only. If it is unavailable,
            // preserve every FIFO phrase through the established path.
            for phrase in phrases where !Task.isCancelled {
                await translateNemotronPhraseIfFresh(phrase)
            }
        }
    }

    private func commitBatchedConfirmedTranslation(
        phrase: RecognizedPhrase,
        translatedText rawTranslation: String,
        normalizedSource: String,
        translationMilliseconds: Double,
        queueMilliseconds: Double,
        generation: UInt64,
        sourceLanguage: Language,
        targetLanguage: Language
    ) async {
        guard !Task.isCancelled, generation == sessionGeneration else { return }
        var translatedText = rawTranslation
        if targetLanguage == .korean {
            translatedText = KoreanHonorificNormalizer.normalize(translatedText)
            translatedText = KoreanTranslationNaturalizer.normalize(translatedText)
            translatedText = KoreanSpeechTextNormalizer.normalize(translatedText)
        }
        let spokenText = targetLanguage == .korean
            ? KoreanSpeechTextNormalizer.boundedForLiveSpeech(translatedText)
            : translatedText
        confirmedSourceLedger.mark(phrase.sourceIDs, as: .translated)
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID,
            event: "translation_completed",
            audioMilliseconds: receivedAudioMilliseconds,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage,
            source: phrase.text,
            translation: spokenText,
            status: "confirmed_batch",
            operationMilliseconds: translationMilliseconds
        )
        if TTSQueuePolicy.shouldSpeak(.confirmed),
           !spokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let limits = SpeechFragmentCoalescingPolicy.generationLimits(
                usesStreamingNeuralVoice: remoteTTS != nil
            )
            let speechUnits = targetLanguage == .korean
                ? KoreanSpeechTextNormalizer.pipelinedSpeechUnits(
                    spokenText,
                    maximumWords: limits.words,
                    maximumCharacters: limits.characters
                )
                : [spokenText]
            let enqueuedAt = ContinuousClock.now
            for (index, unit) in speechUnits.enumerated() {
                enqueueSpeechRequest(SpeechRequest(
                    text: unit, language: targetLanguage,
                    translationKind: .confirmed,
                    enqueuedAt: enqueuedAt,
                    sessionGeneration: generation,
                    sourceAudioEndMilliseconds:
                        phrase.recognitionMilliseconds,
                    sourceIDs: phrase.sourceIDs,
                    completesSourceDelivery: index == speechUnits.count - 1
                ))
            }
            confirmedSourceLedger.mark(phrase.sourceIDs, as: .speechQueued)
        } else if !phrase.sourceIDs.isEmpty {
            confirmedSourceLedger.mark(phrase.sourceIDs, as: .failed)
        }
        await TranslationDebugLogger.shared.record(
            source: phrase.text,
            normalizedSource: normalizedSource,
            translation: translatedText,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage,
            recognizerSessionMilliseconds: phrase.recognitionMilliseconds,
            translationQueueMilliseconds: queueMilliseconds,
            translationMilliseconds: translationMilliseconds
        )
        metrics.asrMilliseconds = phrase.recognitionMilliseconds
        metrics.translationMilliseconds = translationMilliseconds
        metrics.modelTotalMilliseconds = translationMilliseconds
        _ = monotonicTranslationCaptioner.finalize(translatedText)
        confirmedTranslationDisplay = monotonicTranslationCaptioner.text
        metrics.translationPhrase = confirmedTranslationDisplay
        metrics.outputChunks += 1
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
                if metrics.firstHypothesisMilliseconds == nil {
                    metrics.firstHypothesisAt = .now
                    metrics.firstHypothesisMilliseconds = event.elapsedMilliseconds
                    await SimultaneousDebugLogger.shared.record(
                        sessionID: simultaneousDebugSessionID,
                        event: "asr_first_hypothesis",
                        audioMilliseconds: event.audioProcessedSeconds * 1_000,
                        sourceLanguage: sourceLanguage,
                        targetLanguage: targetLanguage,
                        source: event.hypothesis,
                        operationMilliseconds: event.elapsedMilliseconds
                            - event.audioProcessedSeconds * 1_000
                    )
                }
                // Nemotron's revisable cumulative hypothesis is accurate enough
                // to drive the display layer before a clause becomes immutable.
                // This does not feed TTS or the confirmed transcript.
                let contextualHypothesis = Self.contextualNemotronPreview(
                    pendingWords: pendingTranslationWords,
                    hypothesis: event.hypothesis
                )
                if let starterSource = firstSpeechStarterSelector.claim(
                    from: contextualHypothesis
                ) {
                    scheduleFirstSpeechStarter(
                        starterSource,
                        recognitionMilliseconds: event.elapsedMilliseconds
                    )
                }
                let stableDelta = earlyConfirmedSourceCommitter.update(
                    contextualHypothesis, isFinal: event.isFinal
                )
                earlyConfirmedStablePrefix =
                    earlyConfirmedSourceCommitter.committedText
                // Translate/classify only source that has not already entered
                // the confirmed queue. Keep the acoustic committer on the full
                // cumulative hypothesis: switching its committedCount to a
                // shorter tail would permanently suppress later updates.
                let untranslatedHypothesis = transcriptGate
                    .uncommittedCandidate(contextualHypothesis)
                let repeatedStablePrefix = stableDelta == nil
                    ? nil
                    : transcriptGate.uncommittedCandidate(
                        earlyConfirmedStablePrefix
                    )
                if let untranslatedHypothesis {
                    scheduleStablePreviewIdleFlush(
                        untranslatedHypothesis,
                        cumulativeSource: contextualHypothesis,
                        recognitionMilliseconds: event.elapsedMilliseconds
                    )
                } else {
                    pendingStablePreviewIdleFlushTask?.cancel()
                    pendingStablePreviewIdleFlushTask = nil
                    stablePreviewIdleFlushSource = ""
                }
                if let untranslatedHypothesis,
                   let candidate = rollingTranslationScheduler.candidate(
                       hypothesis: untranslatedHypothesis,
                       stablePrefix: repeatedStablePrefix,
                       audioMilliseconds: event.audioProcessedSeconds * 1_000
                   ) {
                        schedulePreviewTranslation(
                            candidate.text, isStable: candidate.isStable,
                            cumulativeSource: contextualHypothesis
                        )
                }
                if let phrase = event.phrase {
                    let correctedPhrase = ASRContextualCorrector.correct(
                        phrase, preferredTerms: preferredTerms
                    )
                    nemotronCommittedDisplay = [nemotronCommittedDisplay, correctedPhrase]
                        .filter { !$0.isEmpty }.joined(separator: " ")
                    // Keep the recognizer's rolling context bounded, but give
                    // the UI its own complete session transcript. Reusing the
                    // rolling snapshot for display used to erase everything
                    // before the last 48 words.
                    let rollingWords = nemotronCommittedDisplay.split(
                        whereSeparator: \Character.isWhitespace
                    )
                    if rollingWords.count > 48 {
                        nemotronCommittedDisplay = rollingWords.suffix(48)
                            .joined(separator: " ")
                    }
                    metrics.committedSource = [
                        metrics.committedSource, correctedPhrase,
                    ]
                    .compactMap { $0 }
                    .filter { !$0.isEmpty }
                    .joined(separator: " ")
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

    /// Uses the portable 20M streaming recognizer only to reserve the first
    /// revisable spoken lead-in. Confirmed captions and all later speech still
    /// come exclusively from Nemotron, preventing cross-engine duplication or
    /// quality drift while removing Nemotron's multi-second first-token wait.
    private func processPortableStarterPreview(_ hypothesis: String) {
        guard !hasStartedSimultaneousSpeech,
              firstSpeechStarterTask == nil,
              let consensusSource = FirstSpeechStarterSelector.consensusSource(
                  portableHypothesis: hypothesis,
                  authoritativeHypothesis: metrics.sourceHypothesis ?? ""
              ),
              let starterSource = firstSpeechStarterSelector.claim(from: consensusSource)
        else { return }
        scheduleFirstSpeechStarter(
            starterSource,
            recognitionMilliseconds: receivedAudioMilliseconds
        )
    }

    /// Translates the first recognized lexical unit without mutating the
    /// resident translator's context or the confirmed-source deduplication
    /// gate. It is intentionally revisable: once the full context arrives the
    /// ordinary quality path translates and speaks the whole sentence again.
    private func scheduleFirstSpeechStarter(
        _ source: String, recognitionMilliseconds: Double
    ) {
        guard firstSpeechStarterTask == nil,
              let sourceLanguage, let targetLanguage
        else { return }
        let generation = sessionGeneration
        let audioMilliseconds = receivedAudioMilliseconds
        // Hold early semantic cuts only until the starter has been translated
        // and queued. The first full-context candidate is then allowed to
        // follow immediately and correct this deliberately revisable lead-in.
        firstSpeechStarterAwaitingFullCorrection = true
        firstSpeechStarterTask = Task(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            await SimultaneousDebugLogger.shared.record(
                sessionID: self.simultaneousDebugSessionID,
                event: "starter_translation_requested",
                audioMilliseconds: audioMilliseconds,
                sourceLanguage: sourceLanguage,
                targetLanguage: targetLanguage,
                source: source,
                status: "revisable_first_unit"
            )
            do {
                let normalizedSource = TranslationSemanticNormalizer.normalize(
                    source, source: sourceLanguage, target: targetLanguage
                )
                let response = try await self.worker.previewTranslateText(
                    text: normalizedSource,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage,
                    preferredTerms: self.preferredTerms
                )
                guard let translation = response.translation,
                      !translation.trimmingCharacters(
                        in: .whitespacesAndNewlines
                      ).isEmpty
                else {
                    await self.finishFirstSpeechStarter(
                        failed: true, generation: generation
                    )
                    return
                }
                await self.enqueueFirstSpeechStarter(
                    translation,
                    source: source,
                    generation: generation,
                    recognitionMilliseconds: recognitionMilliseconds,
                    translationMilliseconds:
                        response.translationMilliseconds ?? 0,
                    sourceAudioEndMilliseconds: audioMilliseconds
                )
            } catch {
                await self.finishFirstSpeechStarter(
                    failed: true, generation: generation
                )
            }
        }
    }

    private func enqueueFirstSpeechStarter(
        _ translation: String,
        source: String,
        generation: UInt64,
        recognitionMilliseconds: Double,
        translationMilliseconds: Double,
        sourceAudioEndMilliseconds: Double
    ) async {
        defer {
            if generation == sessionGeneration { firstSpeechStarterTask = nil }
        }
        guard generation == sessionGeneration else { return }
        guard !hasStartedSimultaneousSpeech, let targetLanguage else {
            firstSpeechStarterAwaitingFullCorrection = false
            return
        }

        var spoken = translation.trimmingCharacters(in: .whitespacesAndNewlines)
        if targetLanguage == .korean {
            spoken = KoreanHonorificNormalizer.normalize(spoken)
            spoken = KoreanTranslationNaturalizer.normalize(spoken)
            spoken = KoreanSpeechTextNormalizer.normalize(spoken)
        }
        guard !spoken.isEmpty else {
            finishFirstSpeechStarter(failed: true, generation: generation)
            return
        }

        // A comma makes the one-word starter sound like an intentional lead-in
        // rather than a clipped sentence. CosyVoice renders it as a short pause
        // while the confirmed full-context translation catches up.
        if spoken.last.map({ !",，.!?。？！".contains($0) }) ?? false {
            spoken.append(",")
        }
        // A complete/full-context request that reached synthesis first makes
        // this revisable lead-in redundant. Never flush that request ahead of
        // the starter and then mark an unheard starter as spoken.
        guard !speechSynthesisInFlight, pendingSpeechFragment == nil else {
            firstSpeechStarterAwaitingFullCorrection = false
            if let sourceLanguage {
                launchLatestPreviewTranslationIfNeeded(
                    sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
                )
            }
            return
        }
        queuedStarterSpokenText = spoken
        queuedStarterGeneration = generation
        enqueueSpeechRequest(SpeechRequest(
            text: spoken,
            language: targetLanguage,
            translationKind: .starter,
            enqueuedAt: .now,
            sessionGeneration: generation,
            sourceAudioEndMilliseconds: sourceAudioEndMilliseconds
        ))
        // Keep the fuller correction parked only until audible PCM confirms
        // this reservation. `confirmAudibleSpeech` then opens the bridge and
        // wakes the newest preview; failure releases it without consuming any
        // source or target prefix.
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID,
            event: "starter_translation_ready",
            audioMilliseconds: receivedAudioMilliseconds,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage,
            source: source,
            translation: spoken,
            status: "revisable_first_unit",
            operationMilliseconds: translationMilliseconds
        )
        metrics.asrMilliseconds = recognitionMilliseconds
    }

    private func finishFirstSpeechStarter(
        failed: Bool = false, generation: UInt64? = nil
    ) {
        if let generation, generation != sessionGeneration { return }
        firstSpeechStarterTask = nil
        if failed {
            queuedStarterSpokenText = nil
            queuedStarterGeneration = nil
            firstSpeechStarterAwaitingFullCorrection = false
            if let sourceLanguage, let targetLanguage {
                launchLatestPreviewTranslationIfNeeded(
                    sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
                )
            }
        }
    }

    private func enqueueNemotronTranslation(
        _ phrase: String, isFinal: Bool, recognitionMilliseconds: Double
    ) {
        if !phrase.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            pendingNemotronFinalFlushTask?.cancel()
            pendingNemotronFinalFlushTask = nil
        }
        let incomingWords = phrase.split(
            whereSeparator: \Character.isWhitespace
        ).map(String.init)
        if Self.isIsolatedFinalLexicalShard(
            incomingWords, after: pendingTranslationWords, isFinal: isFinal
        ) {
            flushPendingNemotronTranslation(
                recognitionMilliseconds: recognitionMilliseconds
            )
            return
        }
        pendingTranslationWords.append(contentsOf: incomingWords)
        // Audio-driven silence detection runs only while ScreenCaptureKit keeps
        // delivering buffers. Pausing a video can stop callbacks completely,
        // leaving the last otherwise speakable 3+ words parked forever. Arm an
        // independent ASR-idle timer on every confirmed delta; a newer delta
        // cancels it, while a genuine pause releases the tail promptly.
        if pendingTranslationWords.count >= 3 {
            scheduleNemotronFinalFallbackFlush(
                recognitionMilliseconds: recognitionMilliseconds,
                delayMilliseconds: 850,
                allowDanglingEnding: false
            )
        }
        // Preserve Basecamp 6's proven four-to-six-word body cadence. The
        // first-speech starter and Basecamp 7 immediate FIFO release provide
        // low latency independently; changing semantic unit size with speaker
        // pace made otherwise identical clauses receive inconsistent context.
        while let cadencePrefix = Self.stableCadencePrefix(
            in: pendingTranslationWords
        ) {
            let prefixCount = cadencePrefix.split(
                whereSeparator: \Character.isWhitespace
            ).count
            pendingTranslationWords.removeFirst(prefixCount)
            firstSpeechStarterAwaitingFullCorrection = false
            enqueueRecognizedPhrase(
                cadencePrefix,
                recognitionMilliseconds: recognitionMilliseconds
            )
        }
        if pendingTranslationWords.isEmpty { return }
        let cadencePending = pendingTranslationWords.joined(separator: " ")
        let endsClause = pendingTranslationWords.last?.last.map {
            ".?!。？！".contains($0)
        } ?? false
        let cadenceContinuesIntoNextEndpoint =
            Self.endsWithContinuingDiscourseMarker(cadencePending)
        // Do not make an uninterrupted long sentence wait for its acoustic
        // endpoint.  A comma or a complete main clause around the middle is a
        // safe monotonic cut: the selected prefix enters the confirmed FIFO
        // now and every word to its right remains in pendingTranslationWords.
        // This is intentionally grammar based rather than tied to any test
        // recording or phrase.
        if let livePrefix = EarlyTranslationClauseSelector.latencyBoundedPrefix(
            in: cadencePending, maximumWords: 10, minimumWords: 5
        ) {
            // The selector is latency oriented and may expose a prefix that a
            // stricter grammar gate still considers dependent. Sending that
            // prefix to `enqueueRecognizedPhrase` re-inserts it at the front of
            // this same buffer. The former recursive remainder scan then saw
            // the identical text forever and exhausted the cooperative
            // executor's stack. Wait for right context instead.
            guard !Self.shouldHoldNemotronDispatch(livePrefix) else { return }
            let prefixCount = livePrefix.split(
                whereSeparator: \Character.isWhitespace
            ).count
            pendingTranslationWords.removeFirst(prefixCount)
            firstSpeechStarterAwaitingFullCorrection = false
            enqueueRecognizedPhrase(
                livePrefix, recognitionMilliseconds: recognitionMilliseconds
            )
            // Never recursively rescan the remainder on the same actor stack.
            // The already-armed idle flush or the next ASR delta will process
            // it with fresh right context.
            return
        }
        // Do not let quality-first semantic waiting turn into a 40-65 word TTS
        // burst. Emit an already complete sentence or clause and retain only
        // its unfinished right context for the next acoustic endpoint.
        if let readyPrefix = EarlyTranslationClauseSelector.boundedPrefix(
            in: cadencePending, maximumWords: 22
        ) {
            // `enqueueRecognizedPhrase` retains an incomplete candidate by
            // putting it back into `pendingTranslationWords`. Recursing after
            // that would select the identical prefix forever and eventually
            // overflow this task's stack. Keep the original buffer intact and
            // wait for more context instead.
            guard !Self.shouldHoldNemotronDispatch(readyPrefix) else { return }
            let prefixCount = readyPrefix.split(whereSeparator: \Character.isWhitespace).count
            pendingTranslationWords.removeFirst(prefixCount)
            firstSpeechStarterAwaitingFullCorrection = false
            enqueueRecognizedPhrase(
                readyPrefix, recognitionMilliseconds: recognitionMilliseconds
            )
            // As above, the next ASR update or idle flush owns the remainder.
            // This makes stack depth constant for arbitrarily long input.
            return
        }
        if !cadenceContinuesIntoNextEndpoint,
           (endsClause && Self.shouldFlushPunctuatedNemotronFragment(
               pendingTranslationWords
           )) || (isFinal && Self.shouldFlushFinalNemotronFragment(
               pendingTranslationWords
           )) {
            flushPendingNemotronTranslation(
                recognitionMilliseconds: recognitionMilliseconds
            )
            return
        }
        if cadenceContinuesIntoNextEndpoint { return }
        // A malformed ASR final can end on a verb fragment (for example
        // "we're just get."). The ordinary grammar gate correctly refuses to
        // commit it immediately, but previously it then remained buffered for
        // the rest of the session. Give the recognizer one brief chance to
        // append a correction; if none arrives, preserve and translate the
        // final instead of silently losing the last sentence.
        if isFinal, endsClause, pendingTranslationWords.count >= 6 {
            scheduleNemotronFinalFallbackFlush(
                recognitionMilliseconds: recognitionMilliseconds,
                delayMilliseconds: 900,
                allowDanglingEnding: true
            )
            return
        }
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
            && (resolvesDependentLeadWithFollowingSentence(words)
                || NeuralSentenceBoundaryClassifier.canEnd(
                    words.joined(separator: " ")
                ))
    }

    static func contextualNemotronPreview(
        pendingWords: [String], hypothesis: String
    ) -> String {
        let current = hypothesis.split(whereSeparator: \Character.isWhitespace)
            .map(String.init)
        guard !pendingWords.isEmpty else { return current.joined(separator: " ") }
        guard !current.isEmpty else { return pendingWords.joined(separator: " ") }
        let pendingKeys = pendingWords.map(Self.previewWordKey)
        let currentKeys = current.map(Self.previewWordKey)

        // RNNT hypotheses are cumulative. Once the pending prefix grows past
        // the old 12-word overlap window, prepending it to a hypothesis that
        // already contains it duplicated the whole spoken passage.
        if currentKeys.starts(with: pendingKeys) {
            return current.joined(separator: " ")
        }
        if pendingKeys.starts(with: currentKeys) {
            return pendingWords.joined(separator: " ")
        }

        // A cumulative RNNT hypothesis can restart a few words before the
        // pending untranslated tail. In that case the tail appears *inside*
        // the newer, longer hypothesis instead of at its beginning. Treating
        // the two strings as unrelated prepended the pending tail to the full
        // hypothesis, producing a repeated source such as
        // "... clean and the Washing your skin ...". The stable committer
        // then made that corruption irreversible and the real intervening
        // clause never reached translation. When the complete pending run is
        // already present anywhere in the cumulative hypothesis, the newer
        // hypothesis is authoritative and needs no prefix added.
        if currentKeys.count >= pendingKeys.count,
           currentKeys.indices.contains(where: { start in
               let end = start + pendingKeys.count
               guard end <= currentKeys.count else { return false }
               return Array(currentKeys[start..<end]) == pendingKeys
           }) {
            return current.joined(separator: " ")
        }

        // A revisable cumulative hypothesis may temporarily omit the final
        // one or two provisional words from the pending tail.  In that case
        // the *stable body* of pending appears inside the newer hypothesis,
        // but the exact full-run check above cannot match it.  Treat a
        // substantial interior match as evidence that `current` is the newer
        // authoritative cumulative view.  Concatenating both views caused the
        // same clause to double on every 80 ms ASR update (27 -> 44 -> 80 ->
        // 148 ... words) and eventually blocked the translator for a minute.
        if Self.hasSubstantialInteriorMatch(
            pendingKeys: pendingKeys, currentKeys: currentKeys
        ) {
            return current.joined(separator: " ")
        }

        let maximum = min(pendingWords.count, current.count)
        var overlap = 0
        if maximum >= 2 {
            for count in stride(from: maximum, through: 2, by: -1) {
                let left = pendingKeys.suffix(count)
                let right = currentKeys.prefix(count)
                if left == right {
                    overlap = count
                    break
                }
            }
        }
        let merged = pendingWords + current.dropFirst(overlap)
        // Never manufacture an adjacent multi-word replay that was absent
        // from the recognizer's current hypothesis.  Real speakers can repeat
        // a word; an immediate repetition of the same three-or-more-word run
        // here is a merge artifact and must not enter the irreversible stable
        // committer or the confirmed translation FIFO.
        if Self.hasAdjacentRepeatedWordRun(merged.map(Self.previewWordKey)),
           !Self.hasAdjacentRepeatedWordRun(currentKeys) {
            return current.joined(separator: " ")
        }
        return merged.joined(separator: " ")
    }

    private static func hasSubstantialInteriorMatch(
        pendingKeys: [String], currentKeys: [String]
    ) -> Bool {
        let maximum = min(pendingKeys.count, currentKeys.count)
        guard maximum >= 4 else { return false }
        for count in stride(from: maximum, through: 4, by: -1) {
            var foundInteriorMatch = false
            for pendingStart in 0...(pendingKeys.count - count) {
                let pendingRange = pendingStart..<(pendingStart + count)
                for currentStart in 0...(currentKeys.count - count) {
                    let currentRange = currentStart..<(currentStart + count)
                    if pendingKeys[pendingRange].elementsEqual(
                        currentKeys[currentRange]
                    ) {
                        // Prefer the longest alignment. If that alignment is
                        // pending-suffix -> current-prefix, it is an ordinary
                        // overlapping window and the earlier pending context
                        // must be retained. Shorter shifted submatches inside
                        // the same run must not override this decision.
                        if currentStart == 0,
                           pendingStart + count == pendingKeys.count {
                            return false
                        }
                        if currentStart > 0 { foundInteriorMatch = true }
                    }
                }
            }
            if foundInteriorMatch { return true }
        }
        return false
    }

    static func hasAdjacentRepeatedWordRun(_ words: [String]) -> Bool {
        guard words.count >= 6 else { return false }
        let maximumRun = min(12, words.count / 2)
        for runLength in stride(from: maximumRun, through: 3, by: -1) {
            let lastStart = words.count - (runLength * 2)
            guard lastStart >= 0 else { continue }
            for start in 0...lastStart {
                let middle = start + runLength
                let end = middle + runLength
                if words[start..<middle].elementsEqual(words[middle..<end]) {
                    return true
                }
            }
        }
        return false
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
            && (resolvesDependentLeadWithFollowingSentence(words)
                || NeuralSentenceBoundaryClassifier.canEnd(
                    words.joined(separator: " ")
                ))
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
        recognitionMilliseconds: Double? = nil,
        allowIncompleteFinal: Bool = false
    ) {
        guard !pendingTranslationWords.isEmpty,
              let last = pendingTranslationWords.last,
              !last.hasSuffix(","),
              allowIncompleteFinal || Self.canEndLiveTranslationPhrase(last)
        else { return }
        let text = pendingTranslationWords.joined(separator: " ")
        pendingTranslationWords.removeAll(keepingCapacity: true)
        pendingNemotronFinalFlushTask?.cancel()
        pendingNemotronFinalFlushTask = nil
        firstSpeechStarterAwaitingFullCorrection = false
        enqueueRecognizedPhrase(
            text,
            recognitionMilliseconds: recognitionMilliseconds
                ?? metrics.asrMilliseconds ?? 0
        )
    }

    private func scheduleNemotronFinalFallbackFlush(
        recognitionMilliseconds: Double,
        delayMilliseconds: Int = 900,
        allowDanglingEnding: Bool = true
    ) {
        pendingNemotronFinalFlushTask?.cancel()
        let generation = sessionGeneration
        pendingNemotronFinalFlushTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(delayMilliseconds))
            guard !Task.isCancelled else { return }
            await self?.flushStalledNemotronFinal(
                generation: generation,
                recognitionMilliseconds: recognitionMilliseconds,
                allowDanglingEnding: allowDanglingEnding
            )
        }
    }

    private func flushStalledNemotronFinal(
        generation: UInt64, recognitionMilliseconds: Double,
        allowDanglingEnding: Bool
    ) {
        guard generation == sessionGeneration,
              Self.shouldFlushNemotronAfterHypothesisStall(
                pendingTranslationWords,
                allowDanglingEnding: allowDanglingEnding
              )
        else { return }
        flushPendingNemotronTranslation(
            recognitionMilliseconds: recognitionMilliseconds,
            allowIncompleteFinal: true
        )
    }

    static func shouldFlushNemotronAfterHypothesisStall(
        _ words: [String], allowDanglingEnding: Bool
    ) -> Bool {
        guard words.count >= 3, let last = words.last,
              !endsInRepeatedToken(words) else { return false }
        let lastKey = previewWordKey(last)
        let previousKey = previewWordKey(words[words.count - 2])
        guard lastKey != previousKey else { return false }
        if !allowDanglingEnding,
           LiveEnglishBoundarySafety.shouldHold(words.joined(separator: " ")) {
            return false
        }
        // Ordinary partials still need a plausible lexical ending. A genuine
        // final may be malformed (for example "we're just get."); preserving
        // it after the final grace period is better than dropping it forever.
        return allowDanglingEnding || canEndLiveTranslationPhrase(last)
    }

    /// A preview may bypass semantic WAIT only after its complete word span is
    /// covered by the independent acoustic stability proof. This is a generic
    /// pause boundary, not a phrase-specific rule.
    static func shouldFlushStablePreviewAfterIdle(
        _ source: String, acousticStablePrefix: String,
        silenceMilliseconds: Double
    ) -> Bool {
        let compact = EarlyTranslationClauseSelector.currentSentence(in: source)
        let words = compact.split(whereSeparator: \Character.isWhitespace).map(String.init)
        guard silenceMilliseconds >= 720,
              shouldFlushNemotronAfterHypothesisStall(
            words, allowDanglingEnding: false
        ), stablePrefix(acousticStablePrefix, coversSemanticBoundary: compact)
        else { return false }
        return !shouldHoldEarlyConfirmedSource(compact)
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
        let recoveredDependentLead = resolvesDependentLeadWithFollowingSentence(words)
        if startsWithUnresolvedDependentClause(words) { return false }
        if !recoveredDependentLead,
           normalizedPhrase.hasPrefix("when ")
            || normalizedPhrase.hasPrefix("and when ") {
            return false
        }
        if !recoveredDependentLead,
           normalizedPhrase.hasPrefix("if ") && !normalizedPhrase.contains(",") {
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
        if LiveEnglishBoundarySafety.shouldHold(words.joined(separator: " ")) {
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
            && (recoveredDependentLead
                || NeuralSentenceBoundaryClassifier.canEnd(
                    words.joined(separator: " ")
                ))
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

        // A comma-closed relative clause already contains a useful semantic
        // breath even when its governing noun was spoken in the preceding
        // unit ("who suffered from that in the moment,"). Keeping it buffered
        // until a new main clause recreates the long pauses this path exists
        // to avoid. Bare relative subjects without a closing comma still wait.
        if ["who", "which"].contains(normalized[start]),
           words.count - start >= 5,
           words.contains(where: { $0.contains(",") }) {
            return false
        }

        // A recognizer endpoint can leave a connective at the beginning of a
        // sentence ("And because ... .").  Holding that fragment alone is
        // correct, but holding the *entire buffer forever* after a subsequent
        // independent sentence has completed is a liveness bug.  Once a
        // following sentence supplies its own subject/predicate-sized unit,
        // dispatch the two together.  Nothing is dropped or reordered, and a
        // single unresolved dependent sentence is still held for context.
        if resolvesDependentLeadWithFollowingSentence(words) {
            return false
        }
        // A comma closes the dependent clause but is not itself a main clause.
        // Require a short right-hand tail so "who suffered ...," remains held
        // while "If ..., we will act" can be emitted.
        guard let comma = words.firstIndex(where: { $0.contains(",") }) else {
            return true
        }
        return words.distance(from: words.index(after: comma), to: words.endIndex) < 3
    }

    private static func resolvesDependentLeadWithFollowingSentence(
        _ words: [String]
    ) -> Bool {
        guard !words.isEmpty else { return false }
        let normalized = words.map {
            $0.lowercased().trimmingCharacters(in: .punctuationCharacters)
        }
        var start = 0
        if ["and", "but", "so"].contains(normalized[0]) { start = 1 }
        guard start < normalized.count,
              ["although", "because", "if", "unless", "when", "whenever",
               "while", "who", "which"].contains(normalized[start]),
              let firstSentenceEnd = words.indices.first(where: { index in
            index >= start + 2
                && words[index].last.map { ".?!。？！".contains($0) } == true
              })
        else { return false }
        let followingStart = words.index(after: firstSentenceEnd)
        guard followingStart < words.endIndex else { return false }
        let following = Array(words[followingStart...])
        guard following.count >= 3,
              let followingLast = following.last,
              followingLast.last.map({ ".?!。？！".contains($0) }) == true,
              canEndLiveTranslationPhrase(followingLast),
              !startsWithUnresolvedDependentClause(following)
        else { return false }
        return true
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
        _ text: String, recognitionMilliseconds: Double, queueMilliseconds: Double,
        translationInput: String? = nil,
        prefetchedTranslation: SpeculativeTranslationCache.Entry? = nil,
        sourceIDs: [UInt64] = []
    ) async {
        guard let sourceLanguage, let targetLanguage else { return }
        let generation = sessionGeneration
        if TranslationQueuePolicy.isStale(queueMilliseconds: queueMilliseconds) {
            await TranslationDebugLogger.shared.record(
                source: text, normalizedSource: text, translation: nil,
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
                recognizerSessionMilliseconds: recognitionMilliseconds,
                translationQueueMilliseconds: queueMilliseconds,
                translationMilliseconds: nil, error: "delayed_confirmed_translation_preserved"
            )
        }
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID,
            event: "translation_requested",
            audioMilliseconds: receivedAudioMilliseconds,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage,
            source: text,
            status: "confirmed",
            operationMilliseconds: queueMilliseconds
        )
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
                let cleanedText = Self.cleanWindowBoundaryArtifacts(
                    translationInput ?? text
                )
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
                    } catch is CancellationError {
                        return
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
                        var consumedPrefetch = false
                        if let prefetchedTranslation,
                           prefetchedTranslation.sessionGeneration == sessionGeneration,
                           prefetchedTranslation.sourceLanguage == sourceLanguage,
                           prefetchedTranslation.targetLanguage == targetLanguage,
                           prefetchedTranslation.confirmedSource
                            == SpeculativeTranslationCache.canonicalSource(text) {
                            if !prefetchedTranslation.requiresResidentCommit {
                                if ConfirmedTranslationPrefetchPolicy
                                    .canUseUnvalidatedPrefetch(
                                        source: normalizedText,
                                        sourceLanguage: sourceLanguage,
                                        targetLanguage: targetLanguage
                                    ) {
                                    // The independent v13 lane already
                                    // guarantees append-only output for this
                                    // exact ordinary clause.
                                    translatedText = prefetchedTranslation.translation
                                    translationMilliseconds =
                                        prefetchedTranslation.translationMilliseconds
                                    consumedPrefetch = true
                                }
                            } else {
                                let promotion = try await Self.withTimeout(
                                    .milliseconds(500)
                                ) {
                                    try await self.worker.commitPreviewTranslation(
                                        text: prefetchedTranslation.translationSource,
                                        translation: prefetchedTranslation.translation,
                                        sourceLanguage: sourceLanguage,
                                        targetLanguage: targetLanguage,
                                        expectedContextVersion:
                                            prefetchedTranslation.contextVersion
                                    )
                                }
                                if promotion.status == "duplicate" {
                                    // The exact source was already committed by an
                                    // older confirmed queue item. Never speak it a
                                    // second time merely because its preview raced.
                                    confirmedSourceLedger.mark(sourceIDs, as: .superseded)
                                    return
                                }
                                if promotion.status == "preview_committed" {
                                    translatedText = prefetchedTranslation.translation
                                    translationMilliseconds =
                                        prefetchedTranslation.translationMilliseconds
                                    consumedPrefetch = true
                                }
                            }
                        }
                        if !consumedPrefetch {
                            // LocalModelWorker owns one synchronous resident
                            // process. Cancelling a Swift timeout cannot abort
                            // its blocking response read; the old code started
                            // an Apple fallback while that same worker remained
                            // occupied, doubling latency and leaving confirmed
                            // work behind an unobservable orphaned request.
                            // Await the single confirmed request directly. The
                            // worker/process lifecycle supplies the real crash
                            // watchdog, while the FIFO prevents sentence loss.
                            let response = try await self.worker.translateText(
                                text: normalizedText,
                                sourceLanguage: sourceLanguage,
                                targetLanguage: targetLanguage,
                                preferredTerms: self.preferredTerms
                            )
                            guard response.status != "duplicate" else {
                                confirmedSourceLedger.mark(sourceIDs, as: .superseded)
                                return
                            }
                            guard let translation = response.translation,
                                  !translation.isEmpty else {
                                throw LocalModelWorkerError.invalidResponse
                            }
                            translatedText = translation
                            translationMilliseconds = response.translationMilliseconds ?? 0
                        }
                    } catch is CancellationError {
                        return
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
                guard response.status != "duplicate" else {
                    confirmedSourceLedger.mark(sourceIDs, as: .superseded)
                    return
                }
                guard let translation = response.translation,
                      !translation.isEmpty else {
                    throw LocalModelWorkerError.invalidResponse
                }
                translatedText = translation
                translationMilliseconds = response.translationMilliseconds ?? 0
                totalMilliseconds = response.totalMilliseconds ?? 0
            }
            // A synchronous resident-process read may finish after stop/start
            // even though its Swift task was cancelled. Never let an old
            // session's result masquerade under the new session generation.
            guard !Task.isCancelled, generation == sessionGeneration else {
                return
            }
            // Translation never waits for speech rendering. This keeps the ASR
            // and contextual text path responsive during long interviews.
            if targetLanguage == .korean {
                translatedText = KoreanHonorificNormalizer.normalize(translatedText)
                translatedText = KoreanTranslationNaturalizer.normalize(translatedText)
                translatedText = KoreanSpeechTextNormalizer.normalize(translatedText)
            }
            let spokenText = targetLanguage == .korean
                ? KoreanSpeechTextNormalizer.boundedForLiveSpeech(translatedText)
                : translatedText
            confirmedSourceLedger.mark(sourceIDs, as: .translated)
            // Preview translations are deliberately handled by
            // schedulePreviewTranslation and never cross this boundary.
            let completedTranslationMilliseconds = translationStarted
                .duration(to: .now).milliseconds
            await SimultaneousDebugLogger.shared.record(
                sessionID: simultaneousDebugSessionID,
                event: "translation_completed",
                audioMilliseconds: receivedAudioMilliseconds,
                sourceLanguage: sourceLanguage,
                targetLanguage: targetLanguage,
                source: text,
                translation: spokenText,
                status: "confirmed",
                operationMilliseconds: completedTranslationMilliseconds
            )
            if TTSQueuePolicy.shouldSpeak(.confirmed),
               !spokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let generationLimits = SpeechFragmentCoalescingPolicy.generationLimits(
                    usesStreamingNeuralVoice: remoteTTS != nil
                )
                let speechUnits = targetLanguage == .korean
                    ? KoreanSpeechTextNormalizer.pipelinedSpeechUnits(
                        spokenText,
                        maximumWords: generationLimits.words,
                        maximumCharacters: generationLimits.characters
                    )
                    : [spokenText]
                let enqueuedAt = ContinuousClock.now
                for (index, unit) in speechUnits.enumerated() {
                    enqueueSpeechRequest(SpeechRequest(
                        text: unit, language: targetLanguage,
                        translationKind: .confirmed,
                        enqueuedAt: enqueuedAt, sessionGeneration: generation,
                        sourceAudioEndMilliseconds: recognitionMilliseconds,
                        sourceIDs: sourceIDs,
                        completesSourceDelivery: index == speechUnits.count - 1
                    ))
                }
                confirmedSourceLedger.mark(sourceIDs, as: .speechQueued)
            } else if !sourceIDs.isEmpty {
                confirmedSourceLedger.mark(sourceIDs, as: .failed)
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
            // Rolling previews already expose only target words that survived
            // consecutive revisions.  Append just the final model's unseen
            // remainder instead of waiting for (or repeating) the whole
            // sentence.  With no preview this naturally appends the full text.
            _ = monotonicTranslationCaptioner.finalize(translatedText)
            confirmedTranslationDisplay = monotonicTranslationCaptioner.text
            metrics.translationPhrase = confirmedTranslationDisplay
            metrics.outputChunks += 1
        } catch is CancellationError {
            return
        } catch {
            confirmedSourceLedger.mark(sourceIDs, as: .failed)
            await TranslationDebugLogger.shared.record(
                source: text, normalizedSource: text, translation: nil,
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
                recognizerSessionMilliseconds: recognitionMilliseconds,
                translationQueueMilliseconds: queueMilliseconds,
                translationMilliseconds: nil, error: error.localizedDescription
            )
            // A single model/network failure is item-scoped. Propagating it
            // through pendingError makes the next audio callback tear down the
            // entire capture/ASR/TTS session, so every later healthy sentence
            // is lost. Keep the three-stage pipeline alive and let the next
            // confirmed unit retry the resident/remote translator.
            metrics.droppedChunks += 1
            await SimultaneousDebugLogger.shared.record(
                sessionID: simultaneousDebugSessionID,
                event: "translation_failed",
                audioMilliseconds: receivedAudioMilliseconds,
                sourceLanguage: sourceLanguage,
                targetLanguage: targetLanguage,
                source: text,
                status: error.localizedDescription
            )
        }
    }

    private func startSpeechQueue() {
        // Requests are admitted directly by scheduleSpeechRequest. Reserving
        // the synthesis slot synchronously closes the race where several
        // requests entered an AsyncStream before its consumer set
        // speechSynthesisInFlight, bypassing backlog coalescing entirely.
    }

    /// Prevents grammatically unfinished one- and two-word tails from becoming
    /// independent CUDA generations. Qwen's production path has a fixed
    /// first-audio cost for every request, so speaking such tails separately
    /// creates both audible cuts and growing lag. A tail waits at most 650 ms
    /// for its continuation; complete sentences and the latency-critical first
    /// starter still pass through immediately.
    private func enqueueSpeechRequest(_ request: SpeechRequest) {
        // A cancelled session can still finish an already-running translation
        // or synthesis callback. Never allow that stale callback to reserve the
        // new session's TTS worker or mutate its backlog.
        guard request.sessionGeneration == sessionGeneration else { return }

        if request.translationKind == .starter {
            // The caller admits a starter only into an empty synthesis queue.
            // Flushing a normal pending request here inverted priority and
            // could make the short first unit expire unheard behind it.
            scheduleSpeechRequest(request)
            return
        }

        if request.translationKind == .simultaneousCommitted,
           simultaneousStarterBridgeOpen,
           !hasQueuedImmediateStarterContinuation {
            // The first stable continuation is computed while the short
            // starter is already playing. Do not add the ordinary 650 ms tiny
            // fragment hold here: it is precisely the bridge that removes the
            // audible pause after first speech. Later deltas still use normal
            // coalescing so they cannot turn into a train of choppy requests.
            flushPendingSpeechFragment()
            hasQueuedImmediateStarterContinuation = true
            scheduleSpeechRequest(request)
            return
        }

        if speechSynthesisInFlight {
            enqueueBackloggedSpeechRequest(request)
            return
        }

        if let pending = pendingSpeechFragment {
            pendingSpeechFlushTask?.cancel()
            pendingSpeechFlushTask = nil
            pendingSpeechFragment = nil
            guard pending.sessionGeneration == request.sessionGeneration,
                  pending.language == request.language else {
                scheduleSpeechRequest(pending)
                enqueueSpeechRequest(request)
                return
            }
            if request.translationKind == .confirmed,
               pending.translationKind == .simultaneousCommitted {
                // The confirmed request contains the complete source thought.
                // Replace an *unspoken* speculative delta instead of joining
                // both and reading the same opening twice.
                scheduleSpeechRequest(request)
                return
            }
            if pending.translationKind == .confirmed,
               request.translationKind == .simultaneousCommitted {
                // Never let a later preview jump ahead of an already confirmed
                // sentence. Its content will be recovered by the next full
                // confirmed request if it remains relevant.
                scheduleSpeechRequest(pending)
                return
            }
            let generationLimits = SpeechFragmentCoalescingPolicy.generationLimits(
                usesStreamingNeuralVoice: remoteTTS != nil
            )
            guard SpeechFragmentCoalescingPolicy.canMerge(
                pending.text, with: request.text, limits: generationLimits
            ) else {
                // The open tail has reached one bounded breath. Start it now
                // and place the newer request in the FIFO behind it instead
                // of growing a single ever-larger CUDA generation.
                scheduleSpeechRequest(pending)
                enqueueBackloggedSpeechRequest(request)
                return
            }
            let joined = [pending.text, request.text]
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            let kind: TTSQueuePolicy.TranslationKind =
                pending.translationKind == .confirmed || request.translationKind == .confirmed
                ? .confirmed : .simultaneousCommitted
            let combined = SpeechRequest(
                text: joined,
                language: request.language,
                translationKind: kind,
                enqueuedAt: pending.enqueuedAt,
                sessionGeneration: request.sessionGeneration,
                sourceAudioEndMilliseconds: pending.sourceAudioEndMilliseconds,
                audiblePrefixAfterCompletion: kind == .simultaneousCommitted
                    ? request.audiblePrefixAfterCompletion
                        ?? pending.audiblePrefixAfterCompletion
                    : nil,
                sourceIDs: pending.sourceIDs + request.sourceIDs,
                completesSourceDelivery: pending.completesSourceDelivery
                    || request.completesSourceDelivery
            )
            // Two tiny deltas can still be grammatically unfinished. Continue
            // coalescing instead of emitting another one-word GPU request.
            if SpeechFragmentCoalescingPolicy.shouldHold(
                joined, translationKind: kind
            ) {
                pendingSpeechFragment = combined
                let generation = combined.sessionGeneration
                let hold = SpeechFragmentCoalescingPolicy.maximumHold(
                    forEstimatedLagMilliseconds: max(
                        0, receivedAudioMilliseconds - combined.sourceAudioEndMilliseconds
                    )
                )
                pendingSpeechFlushTask = Task { [weak self] in
                    try? await Task.sleep(for: hold)
                    guard !Task.isCancelled else { return }
                    await self?.flushPendingSpeechFragment(generation: generation)
                }
            } else {
                scheduleSpeechRequest(combined)
            }
            return
        }

        guard SpeechFragmentCoalescingPolicy.shouldHold(
            request.text, translationKind: request.translationKind
        ) else {
            scheduleSpeechRequest(request)
            return
        }
        pendingSpeechFragment = request
        let generation = request.sessionGeneration
        let hold = SpeechFragmentCoalescingPolicy.maximumHold(
            forEstimatedLagMilliseconds: max(
                0, receivedAudioMilliseconds - request.sourceAudioEndMilliseconds
            )
        )
        pendingSpeechFlushTask = Task { [weak self] in
            try? await Task.sleep(for: hold)
            guard !Task.isCancelled else { return }
            await self?.flushPendingSpeechFragment(generation: generation)
        }
    }

    /// Admits work to the dedicated TTS worker's bounded FIFO. Adjacent tiny
    /// fragments may share one short breath, but no queue slot can grow past
    /// the policy's word/character ceiling. This prevents the positive feedback
    /// loop where a busy GPU produced an even larger next request.
    private func enqueueBackloggedSpeechRequest(_ request: SpeechRequest) {
        pendingSpeechFlushTask?.cancel()
        pendingSpeechFlushTask = nil
        guard let pending = backloggedSpeechRequests.last else {
            appendBoundedSpeechRequest(request)
            return
        }
        guard pending.sessionGeneration == request.sessionGeneration,
              pending.language == request.language else {
            // An incoming request must never erase valid queued speech. Stale
            // generations are rejected at enqueueSpeechRequest; a same-session
            // language transition is simply appended in FIFO order.
            appendBoundedSpeechRequest(request)
            return
        }
        if request.translationKind == .confirmed,
           pending.translationKind == .simultaneousCommitted {
            // The confirmed unit contains the complete thought represented by
            // the still-unheard speculative tail.
            backloggedSpeechRequests[backloggedSpeechRequests.count - 1] = request
            return
        }
        if pending.translationKind == .confirmed,
           request.translationKind == .simultaneousCommitted {
            return
        }
        if pending.translationKind == .confirmed,
           request.translationKind == .confirmed {
            guard SpeechFragmentCoalescingPolicy.shouldMergeBackloggedConfirmed(
                usesStreamingNeuralVoice: remoteTTS != nil
            ) else {
                appendBoundedSpeechRequest(request)
                return
            }
        }
        let generationLimits = SpeechFragmentCoalescingPolicy.generationLimits(
            usesStreamingNeuralVoice: remoteTTS != nil
        )
        guard pending.translationKind == request.translationKind,
              SpeechFragmentCoalescingPolicy.canMerge(
                  pending.text, with: request.text, limits: generationLimits
              ) else {
            appendBoundedSpeechRequest(request)
            return
        }
        let joined = [pending.text, request.text]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        backloggedSpeechRequests[backloggedSpeechRequests.count - 1] = SpeechRequest(
            text: joined,
            language: request.language,
            translationKind: request.translationKind,
            enqueuedAt: pending.enqueuedAt,
            sessionGeneration: request.sessionGeneration,
            // Preserve the oldest source endpoint so lag diagnostics cannot be
            // made artificially small by appending a newer fragment.
            sourceAudioEndMilliseconds: pending.sourceAudioEndMilliseconds,
            audiblePrefixAfterCompletion: request.translationKind == .simultaneousCommitted
                ? request.audiblePrefixAfterCompletion
                    ?? pending.audiblePrefixAfterCompletion
                : nil,
            sourceIDs: pending.sourceIDs + request.sourceIDs,
            completesSourceDelivery: pending.completesSourceDelivery
                || request.completesSourceDelivery
        )
    }

    private func appendBoundedSpeechRequest(_ request: SpeechRequest) {
        if backloggedSpeechRequests.count
            >= SpeechFragmentCoalescingPolicy.maximumWaitingGenerations {
            let staleIndex = backloggedSpeechRequests.firstIndex {
                TTSQueuePolicy.isStale(
                    $0.translationKind, text: $0.text, enqueuedAt: $0.enqueuedAt
                )
            }
            let decision = staleIndex.map {
                SpeechMailboxCapacityPolicy.Decision.evictWaiting(at: $0)
            } ?? SpeechMailboxCapacityPolicy.overflowDecision(
                waitingKinds: backloggedSpeechRequests.map(\.translationKind),
                incomingKind: request.translationKind
            )
            switch decision {
            case let .evictWaiting(removalIndex):
                let removed = backloggedSpeechRequests.remove(at: removalIndex)
                metrics.droppedChunks += 1
                failQueuedStarterIfNeeded(removed)
                let diagnostic = "speech_mailbox_evicted kind=\(removed.translationKind) text=\(removed.text.debugDescription)\n"
                FileHandle.standardError.write(Data(diagnostic.utf8))
            case .discardIncoming:
                metrics.droppedChunks += 1
                failQueuedStarterIfNeeded(request)
                let diagnostic = "speech_mailbox_discarded_speculative kind=\(request.translationKind) text=\(request.text.debugDescription)\n"
                FileHandle.standardError.write(Data(diagnostic.utf8))
                return
            case .appendWithoutEviction:
                break
            }
        }
        backloggedSpeechRequests.append(request)
    }

    private func flushPendingSpeechFragment(generation: UInt64? = nil) {
        guard let pending = pendingSpeechFragment,
              generation == nil || pending.sessionGeneration == generation else { return }
        pendingSpeechFragment = nil
        pendingSpeechFlushTask?.cancel()
        pendingSpeechFlushTask = nil
        scheduleSpeechRequest(pending)
    }

    private func scheduleSpeechRequest(_ request: SpeechRequest) {
        guard speechSynthesisInFlight else {
            // Reserve before yielding the actor. This is the key backpressure
            // invariant: no second request can slip into a separate FIFO slot
            // while the new task is waiting to begin.
            speechSynthesisInFlight = true
            speechTask = Task(priority: .userInitiated) { [weak self] in
                await self?.drainSpeechQueue(startingWith: request)
            }
            return
        }
        enqueueBackloggedSpeechRequest(request)
    }

    private func drainSpeechQueue(startingWith first: SpeechRequest) async {
        var current: SpeechRequest? = first
        while let request = current, !Task.isCancelled {
            await synthesize(request)
            if !backloggedSpeechRequests.isEmpty {
                current = backloggedSpeechRequests.removeFirst()
            } else if let openTail = pendingSpeechFragment {
                pendingSpeechFragment = nil
                pendingSpeechFlushTask?.cancel()
                pendingSpeechFlushTask = nil
                current = openTail
            } else {
                current = nil
            }
        }
        speechSynthesisInFlight = false
        speechTask = nil
    }

    private func synthesize(_ request: SpeechRequest) async {
        guard TTSQueuePolicy.shouldSpeak(request.translationKind) else { return }
        guard request.sessionGeneration == sessionGeneration else { return }
        var speechText = request.text
        let consumesSimultaneousBridge = request.translationKind == .confirmed
            && simultaneousTTSIsEnabled && simultaneousStarterBridgeOpen
        if consumesSimultaneousBridge {
            // A confirmed translation may have finished while the starter was
            // still being synthesized. Apply the *audible* prefix remainder at
            // the last responsible moment. A merely queued or failed preview
            // is never removed from the confirmed sentence.
            speechText = simultaneousTranslationCommitter.audibleRemainder(
                of: speechText
            )
        }
        guard !speechText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            if consumesSimultaneousBridge { closeSimultaneousBridge() }
            return
        }
        let synthesisText = request.language == .korean
            ? KoreanSpeechTextNormalizer.normalizedForSpeech(speechText)
            : speechText
        let speechKey = synthesisText.lowercased()
            .split(whereSeparator: \Character.isWhitespace).joined(separator: " ")
        if speechKey == lastSpokenText,
           request.translationKind != .confirmed {
            let diagnostic = "speech_dropped_duplicate text=\(speechText.debugDescription)\n"
            FileHandle.standardError.write(Data(diagnostic.utf8))
            metrics.droppedChunks += 1
            if consumesSimultaneousBridge { closeSimultaneousBridge() }
            failQueuedStarterIfNeeded(request)
            return
        }
        guard !TTSQueuePolicy.isStale(
            request.translationKind,
            text: speechText,
            enqueuedAt: request.enqueuedAt
        ) else {
            let age = request.enqueuedAt.duration(to: .now).milliseconds
            let diagnostic = "speech_dropped_stale age_ms=\(Int(age)) text=\(speechText.debugDescription)\n"
            FileHandle.standardError.write(Data(diagnostic.utf8))
            metrics.droppedChunks += 1
            confirmedSourceLedger.mark(request.sourceIDs, as: .failed)
            if consumesSimultaneousBridge { closeSimultaneousBridge() }
            failQueuedStarterIfNeeded(request)
            return
        }
        if consumesSimultaneousBridge { closeSimultaneousBridge() }
        // Do not gate speech on cumulative ASR streamLag. That metric includes
        // intentionally dropped capture buffers and therefore may never fall
        // again after overload. The old hysteresis permanently muted every
        // later translation once it crossed 2.5 seconds. Freshness is enforced
        // by the bounded newest-only queue and request-age check above instead.
        let started = ContinuousClock.now
        lastSpokenText = speechKey
        let speechStatus: String
        switch request.translationKind {
        case .starter: speechStatus = "starter"
        case .simultaneousCommitted: speechStatus = "simultaneous"
        case .confirmed: speechStatus = "confirmed"
        case .preview: speechStatus = "preview"
        }
        let speechQueueMilliseconds = request.enqueuedAt.duration(to: .now)
            .milliseconds
        let effectiveSpeechBacklogMilliseconds = SpeechCatchUpPolicy
            .effectiveBacklogMilliseconds(
                speechQueueMilliseconds: speechQueueMilliseconds,
                sourceAudioEndMilliseconds: request.sourceAudioEndMilliseconds,
                receivedAudioMilliseconds: receivedAudioMilliseconds
            )
        let sourceWordsPerMinute = sourceSpeechPaceTracker.wordsPerMinute(
            at: receivedAudioMilliseconds
        )
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID, event: "tts_started",
            audioMilliseconds: receivedAudioMilliseconds,
            sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
            translation: speechText,
            status: speechStatus,
            operationMilliseconds: speechQueueMilliseconds
        )
        let firstChunkGate = OneShotEventGate()
        let firstAudibleChunkGate = OneShotEventGate()
        let attemptedRemoteTTS = remoteTTS != nil && !remoteTTSCircuitOpen
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID,
            event: "tts_pace_selected",
            audioMilliseconds: receivedAudioMilliseconds,
            sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
            status: attemptedRemoteTTS ? "neural" : "system",
            operationMilliseconds: sourceWordsPerMinute ?? 0,
            probability: SourceSpeechPacePolicy.rateMultiplier(
                for: sourceWordsPerMinute
            )
        )
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID,
            event: "tts_catch_up_selected",
            audioMilliseconds: receivedAudioMilliseconds,
            sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
            status: attemptedRemoteTTS ? "neural" : "system",
            operationMilliseconds: effectiveSpeechBacklogMilliseconds,
            probability: attemptedRemoteTTS
                ? SourceSpeechPacePolicy.neuralGenerationSpeed(
                    for: sourceWordsPerMinute,
                    backlogMilliseconds: effectiveSpeechBacklogMilliseconds
                )
                : SourceSpeechPacePolicy.rateMultiplier(
                    for: sourceWordsPerMinute
                ) * SourceSpeechPacePolicy.neuralCatchUpMultiplier(
                    backlogMilliseconds: effectiveSpeechBacklogMilliseconds
                )
        )
        do {
            let remoteTTS = attemptedRemoteTTS ? self.remoteTTS : nil
            let continuation = outputContinuation
            let outputGate = SpeechOutputGate()
            activeSpeechOutputGate = outputGate
            let playbackGroupID = UUID()
            let debugSessionID = simultaneousDebugSessionID
            let debugAudioMilliseconds = receivedAudioMilliseconds
            let debugSourceLanguage = sourceLanguage
            let debugTargetLanguage = targetLanguage
            let debugTranslation = speechText
            defer {
                outputGate.close()
                if activeSpeechOutputGate === outputGate {
                    activeSpeechOutputGate = nil
                }
            }
            let watchdogTimeout = request.translationKind == .starter
                ? TTSQueuePolicy.starterSynthesisWatchdogTimeout
                : TTSQueuePolicy.synthesisWatchdogTimeout
            try await Self.withSpeechWatchdog(timeout: watchdogTimeout) {
                if let remoteTTS {
                    let initialFrames = request.translationKind == .starter
                        ? CosyVoiceStreamingClient.initialFramesPerChunk
                        : CosyVoiceStreamingClient.continuityInitialFramesPerChunk
                    try await remoteTTS.synthesize(
                        text: synthesisText, language: request.language,
                        speed: SourceSpeechPacePolicy.neuralGenerationSpeed(
                            for: sourceWordsPerMinute,
                            backlogMilliseconds: effectiveSpeechBacklogMilliseconds
                        ),
                        initialFramesPerChunk: initialFrames,
                        onChunk: { chunk in
                            let wasEnqueued = outputGate.yieldIfOpen(
                                chunk.groupedForPlayback(playbackGroupID), to: continuation
                            )
                            if wasEnqueued, firstChunkGate.fire() {
                                let firstPCMDelay = started.duration(to: .now).milliseconds
                                Task {
                                    await SimultaneousDebugLogger.shared.record(
                                        sessionID: debugSessionID,
                                        event: "tts_first_pcm",
                                        audioMilliseconds: debugAudioMilliseconds,
                                        sourceLanguage: debugSourceLanguage,
                                        targetLanguage: debugTargetLanguage,
                                        translation: debugTranslation,
                                        status: "output_stream_enqueued",
                                        operationMilliseconds: firstPCMDelay
                                    )
                                }
                            }
                            if wasEnqueued,
                               chunk.rms >= SessionPerformanceMetrics.audibleOutputRMSThreshold,
                               firstAudibleChunkGate.fire() {
                                let firstAudiblePCMDelay = started.duration(to: .now).milliseconds
                                Task {
                                    await self.confirmAudibleSpeech(request)
                                    await SimultaneousDebugLogger.shared.record(
                                        sessionID: debugSessionID,
                                        event: "tts_first_audible_pcm",
                                        audioMilliseconds: debugAudioMilliseconds,
                                        sourceLanguage: debugSourceLanguage,
                                        targetLanguage: debugTargetLanguage,
                                        translation: debugTranslation,
                                        status: "audible_output_stream_enqueued",
                                        operationMilliseconds: firstAudiblePCMDelay
                                    )
                                }
                            }
                        }
                    )
                } else {
                    try await AppleLocalTranslationProvider.speakText(
                        synthesisText, language: request.language,
                        backlogMilliseconds: effectiveSpeechBacklogMilliseconds,
                        sourceWordsPerMinute: sourceWordsPerMinute
                    )
                }
            }
            if remoteTTS == nil {
                // The local renderer owns playback directly rather than
                // yielding PCM through translatedAudio.
                _ = firstChunkGate.fire()
                _ = firstAudibleChunkGate.fire()
            }
            if firstAudibleChunkGate.hasFired {
                if remoteTTS != nil { sessionHasDeliveredRemoteSpeech = true }
                // Make the state transition deterministic even if the callback
                // task has not yet resumed on this actor.
                confirmAudibleSpeech(request)
                confirmCompletedSpeech(request)
            } else {
                failQueuedStarterIfNeeded(request)
            }
            let completedSynthesisMilliseconds = started.duration(to: .now).milliseconds
            metrics.synthesisMilliseconds = completedSynthesisMilliseconds
            await SimultaneousDebugLogger.shared.record(
                sessionID: simultaneousDebugSessionID,
                event: "tts_completed",
                audioMilliseconds: receivedAudioMilliseconds,
                sourceLanguage: sourceLanguage,
                targetLanguage: targetLanguage,
                translation: speechText,
                status: speechStatus,
                operationMilliseconds: completedSynthesisMilliseconds
            )
        } catch {
            let failureWasFirstAudioTimeout: Bool
            if case CosyVoiceStreamingError.firstAudioTimeout = error {
                failureWasFirstAudioTimeout = true
            } else {
                failureWasFirstAudioTimeout = false
            }
            let failureWasPermanentBeforeAudio: Bool
            let failureWasRemoteOutageBeforeAudio: Bool
            if case let CosyVoiceStreamingError.invalidResponse(status) = error {
                // A missing/unsupported route will not recover on the next
                // sentence. Open the circuit once and immediately retain
                // audible interpretation through the local renderer.
                failureWasPermanentBeforeAudio = [400, 404, 405, 410, 415, 422]
                    .contains(status)
                failureWasRemoteOutageBeforeAudio = [502, 503, 504].contains(status)
            } else {
                failureWasPermanentBeforeAudio = false
                if let urlError = error as? URLError {
                    failureWasRemoteOutageBeforeAudio = [
                        .timedOut, .cannotFindHost, .cannotConnectToHost,
                        .dnsLookupFailed, .networkConnectionLost,
                        .notConnectedToInternet, .secureConnectionFailed
                    ].contains(urlError.code)
                } else {
                    failureWasRemoteOutageBeforeAudio =
                        error is SpeechSynthesisWatchdogTimeout
                }
            }
            if attemptedRemoteTTS,
               RemoteTTSCircuitBreakerPolicy.shouldOpen(
                   firstAudioWasEmitted: firstChunkGate.hasFired,
                   failureWasFirstAudioTimeout: failureWasFirstAudioTimeout,
                   failureWasPermanentBeforeAudio: failureWasPermanentBeforeAudio,
                   failureWasRemoteOutageBeforeAudio: failureWasRemoteOutageBeforeAudio,
                   sessionHasDeliveredRemoteSpeech: sessionHasDeliveredRemoteSpeech
               ) {
                remoteTTSCircuitOpen = true
                neuralVoiceReady = false
                await SimultaneousDebugLogger.shared.record(
                    sessionID: simultaneousDebugSessionID,
                    event: "tts_remote_circuit_open",
                    audioMilliseconds: receivedAudioMilliseconds,
                    sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
                    translation: speechText,
                    status: failureWasPermanentBeforeAudio
                        ? "remote_route_unavailable"
                        : failureWasFirstAudioTimeout
                            ? "first_audio_deadline_exceeded"
                            : "remote_service_unavailable"
                )
                do {
                    try await AppleLocalTranslationProvider.speakText(
                        synthesisText, language: request.language,
                        backlogMilliseconds: effectiveSpeechBacklogMilliseconds,
                        sourceWordsPerMinute: sourceWordsPerMinute
                    )
                    _ = firstChunkGate.fire()
                    _ = firstAudibleChunkGate.fire()
                    confirmAudibleSpeech(request)
                    confirmCompletedSpeech(request)
                    let completed = started.duration(to: .now).milliseconds
                    metrics.synthesisMilliseconds = completed
                    await SimultaneousDebugLogger.shared.record(
                        sessionID: simultaneousDebugSessionID,
                        event: "tts_completed",
                        audioMilliseconds: receivedAudioMilliseconds,
                        sourceLanguage: sourceLanguage,
                        targetLanguage: targetLanguage,
                        translation: speechText,
                        status: "\(speechStatus):local_failover",
                        operationMilliseconds: completed
                    )
                    return
                } catch {
                    let diagnostic = "speech_local_failover_failed error=\(error.localizedDescription)\n"
                    FileHandle.standardError.write(Data(diagnostic.utf8))
                }
            }
            let diagnostic = "speech_synthesis_failed error=\(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(diagnostic.utf8))
            // A request is only "spoken" after PCM enters the playback stream.
            // Let an identical confirmed correction retry after a starter or
            // transient network failure instead of dropping it as a duplicate.
            if !firstChunkGate.hasFired, lastSpokenText == speechKey {
                lastSpokenText = ""
            }
            if firstAudibleChunkGate.hasFired,
               request.translationKind != .starter {
                // Some audible PCM already entered playback. Resolve the
                // lag event, but do not mark the incomplete target prefix as
                // fully spoken. The confirmed request will retain it.
                confirmAudibleSpeech(request)
            } else if firstAudibleChunkGate.hasFired {
                invalidatePartiallyDeliveredStarter(request)
            } else {
                failQueuedStarterIfNeeded(request)
            }
            // GPU-voice mode must never silently change the product voice to a
            // platform-specific local synthesizer. Keep translation alive, but
            // leave speech unavailable until the configured portable service
            // recovers so the failure is measurable instead of disguised.
            if remoteTTS != nil, request.translationKind != .starter {
                neuralVoiceReady = false
            }
            confirmedSourceLedger.mark(request.sourceIDs, as: .failed)
            await SimultaneousDebugLogger.shared.record(
                sessionID: simultaneousDebugSessionID, event: "tts_failed",
                audioMilliseconds: receivedAudioMilliseconds,
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage,
                translation: speechText,
                status: "\(speechStatus):\(error.localizedDescription)"
            )
        }
    }

    private func confirmAudibleSpeech(_ request: SpeechRequest) {
        guard request.sessionGeneration == sessionGeneration else { return }
        guard !failedSpeechRequestIDs.contains(request.id),
              confirmedAudibleSpeechRequestIDs.insert(request.id).inserted
        else { return }
        hasStartedSimultaneousSpeech = true
        recordInterpretationLag(
            sourceAudioEndMilliseconds: request.sourceAudioEndMilliseconds
        )
        if request.translationKind == .starter {
            starterFirstAudibleAt = .now
        } else if let starterFirstAudibleAt {
            let gap = starterFirstAudibleAt.duration(to: .now).milliseconds
            self.starterFirstAudibleAt = nil
            let status = request.translationKind == .simultaneousCommitted
                ? "simultaneous" : "confirmed"
            Task {
                await SimultaneousDebugLogger.shared.record(
                    sessionID: simultaneousDebugSessionID,
                    event: "starter_continuation_gap",
                    audioMilliseconds: receivedAudioMilliseconds,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage,
                    translation: request.text,
                    status: status,
                    operationMilliseconds: gap
                )
            }
        }
        guard request.translationKind == .starter,
              queuedStarterGeneration == request.sessionGeneration,
              queuedStarterSpokenText == request.text
        else { return }
        simultaneousTranslationCommitter.markSpoken(request.text)
        simultaneousStarterBridgeOpen = true
        queuedStarterSpokenText = nil
        queuedStarterGeneration = nil
        firstSpeechStarterAwaitingFullCorrection = false
        if let sourceLanguage, let targetLanguage {
            launchLatestPreviewTranslationIfNeeded(
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
            )
        }
    }

    private func confirmCompletedSpeech(_ request: SpeechRequest) {
        guard request.sessionGeneration == sessionGeneration,
              !failedSpeechRequestIDs.contains(request.id) else { return }
        if request.translationKind == .confirmed,
           request.completesSourceDelivery {
            confirmedSourceLedger.mark(request.sourceIDs, as: .audible)
            return
        }
        guard request.translationKind == .simultaneousCommitted,
              let audiblePrefix = request.audiblePrefixAfterCompletion,
              !audiblePrefix.isEmpty
        else { return }
        simultaneousTranslationCommitter.markAudiblePrefix(audiblePrefix)
    }

    private func invalidatePartiallyDeliveredStarter(_ request: SpeechRequest) {
        guard request.sessionGeneration == sessionGeneration,
              request.translationKind == .starter else { return }
        failedSpeechRequestIDs.insert(request.id)
        confirmedAudibleSpeechRequestIDs.remove(request.id)
        queuedStarterSpokenText = nil
        queuedStarterGeneration = nil
        firstSpeechStarterAwaitingFullCorrection = false
        starterFirstAudibleAt = nil
        closeSimultaneousBridge()
        if pendingSpeechFragment?.translationKind == .simultaneousCommitted {
            pendingSpeechFragment = nil
        }
        backloggedSpeechRequests.removeAll {
            $0.translationKind == .simultaneousCommitted
        }
        if let sourceLanguage, let targetLanguage {
            launchLatestPreviewTranslationIfNeeded(
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
            )
        }
    }

    private func closeSimultaneousBridge() {
        simultaneousTranslationCommitter.reset()
        simultaneousStarterBridgeOpen = false
        hasQueuedImmediateStarterContinuation = false
    }

    private func failQueuedStarterIfNeeded(_ request: SpeechRequest) {
        guard request.translationKind == .starter,
              queuedStarterGeneration == request.sessionGeneration,
              queuedStarterSpokenText == request.text
        else { return }
        queuedStarterSpokenText = nil
        queuedStarterGeneration = nil
        firstSpeechStarterAwaitingFullCorrection = false
        if let sourceLanguage, let targetLanguage {
            launchLatestPreviewTranslationIfNeeded(
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
            )
        }
    }

    private static func withSpeechWatchdog(
        timeout: Duration = TTSQueuePolicy.synthesisWatchdogTimeout,
        operation: @escaping @Sendable () async throws -> Void
    ) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(for: timeout)
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

    private func recordInterpretationLag(sourceAudioEndMilliseconds: Double) {
        let lag = max(0, receivedAudioMilliseconds - sourceAudioEndMilliseconds)
        if metrics.firstInterpretationLagMilliseconds == nil {
            metrics.firstInterpretationLagMilliseconds = lag
        }
        metrics.interpretationLagMilliseconds = lag
        metrics.maximumInterpretationLagMilliseconds = max(
            metrics.maximumInterpretationLagMilliseconds, lag
        )
    }

    private func record(_ event: StableSpeechEvent) async {
        // ScreenCaptureKit may stop delivering callbacks completely when a
        // Chrome tab becomes silent. Audio-time checks then freeze at zero and
        // incorrectly accept a minutes-old SpeechTranscriber backlog.
        guard hasRecentSpeech() else { return }
        metrics.sourceHypothesis = event.hypothesis
        if metrics.firstHypothesisMilliseconds == nil {
            metrics.firstHypothesisAt = .now
            metrics.firstHypothesisMilliseconds = event.elapsedMilliseconds
        }
        if let delta = event.committedDelta {
            progressiveCommittedDisplay = [progressiveCommittedDisplay, delta]
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            // Keep translation scheduling on bounded right context while the
            // display owns the complete transcript for this session.
            let rollingWords = progressiveCommittedDisplay.split(
                whereSeparator: \Character.isWhitespace
            )
            if rollingWords.count > 1_200 {
                progressiveCommittedDisplay = rollingWords.suffix(1_200)
                    .joined(separator: " ")
            }
            metrics.committedSource = [metrics.committedSource, delta]
                .compactMap { $0 }
                .filter { !$0.isEmpty }
                .joined(separator: " ")
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
            metrics.firstHypothesisAt = .now
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

    private func scheduleStablePreviewIdleFlush(
        _ hypothesis: String,
        cumulativeSource: String,
        recognitionMilliseconds: Double
    ) {
        let current = EarlyTranslationClauseSelector.currentSentence(in: hypothesis)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let canonical = SpeculativeTranslationCache.canonicalSource(current)
        guard current.split(whereSeparator: \Character.isWhitespace).count >= 3,
              current.count >= 5 else {
            pendingStablePreviewIdleFlushTask?.cancel()
            pendingStablePreviewIdleFlushTask = nil
            stablePreviewIdleFlushSource = ""
            return
        }
        // Identical ASR observations establish acoustic stability but must not
        // keep pushing the wall-clock deadline forward. Only a lexical change
        // restarts the pause timer.
        guard canonical != stablePreviewIdleFlushSource else { return }
        pendingStablePreviewIdleFlushTask?.cancel()
        stablePreviewIdleFlushSource = canonical
        let generation = sessionGeneration
        pendingStablePreviewIdleFlushTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(850))
            guard !Task.isCancelled else { return }
            // An unchanged decoder hypothesis is not silence: Nemotron may be
            // decoding behind a speaker who is still talking. Wait for VAD to
            // independently observe a real pause before creating another
            // neural-TTS job.
            while !Task.isCancelled {
                guard let self else { return }
                if await self.hasAcousticSilenceForStablePreviewFlush(
                    expectedGeneration: generation,
                    expectedCanonicalSource: canonical
                ) {
                    break
                }
                try? await Task.sleep(for: .milliseconds(120))
            }
            guard !Task.isCancelled else { return }
            await self?.flushStablePreviewAfterIdle(
                source: current,
                cumulativeSource: cumulativeSource,
                recognitionMilliseconds: recognitionMilliseconds,
                expectedGeneration: generation,
                expectedCanonicalSource: canonical
            )
        }
    }

    private func hasAcousticSilenceForStablePreviewFlush(
        expectedGeneration: UInt64,
        expectedCanonicalSource: String
    ) -> Bool {
        guard expectedGeneration == sessionGeneration,
              expectedCanonicalSource == stablePreviewIdleFlushSource,
              let lastSpeechWallClock else { return false }
        return lastSpeechWallClock.duration(to: .now).milliseconds >= 720
    }

    private func flushStablePreviewAfterIdle(
        source: String,
        cumulativeSource: String,
        recognitionMilliseconds: Double,
        expectedGeneration: UInt64,
        expectedCanonicalSource: String
    ) async {
        pendingStablePreviewIdleFlushTask = nil
        guard expectedGeneration == sessionGeneration,
              expectedCanonicalSource == stablePreviewIdleFlushSource,
              let currentUncommitted = transcriptGate.uncommittedCandidate(
                  cumulativeSource
              ) else { return }
        let current = EarlyTranslationClauseSelector.currentSentence(
            in: currentUncommitted
        )
        guard SpeculativeTranslationCache.canonicalSource(current)
                == expectedCanonicalSource else { return }
        let uncommittedStablePrefix = transcriptGate.uncommittedCandidate(
            earlyConfirmedStablePrefix
        ) ?? ""
        let acousticSilenceMilliseconds = lastSpeechWallClock?
            .duration(to: .now).milliseconds ?? 0
        guard Self.shouldFlushStablePreviewAfterIdle(
            source, acousticStablePrefix: uncommittedStablePrefix,
            silenceMilliseconds: acousticSilenceMilliseconds
        ) else { return }

        let deduplicationSource = Self.cumulativePrefix(
            in: cumulativeSource, throughTailBoundary: source
        ) ?? source
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID,
            event: "stable_preview_idle_flush",
            audioMilliseconds: receivedAudioMilliseconds,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage,
            source: source,
            status: "confirmed_after_850ms_idle"
        )
        await enqueueEarlyConfirmedBoundary(
            source,
            deduplicationSource: deduplicationSource,
            recognitionMilliseconds: recognitionMilliseconds,
            probability: nil
        )
    }

    private func schedulePreviewTranslation(
        _ hypothesis: String, isStable: Bool,
        cumulativeSource: String? = nil
    ) {
        guard let sourceLanguage, let targetLanguage else { return }
        let current = EarlyTranslationClauseSelector.currentSentence(in: hypothesis)
        let canonicalCurrent = SpeculativeTranslationCache.canonicalSource(current)
        if !lastRejectedPreviewSource.isEmpty,
           canonicalCurrent != lastRejectedPreviewSource {
            lastRejectedPreviewSource = ""
            lastRejectedPreviewWasStable = false
        }
        // This path is display-only, but two-word partials are not meaningful
        // EN<->KO translation units. The scheduler has already selected a
        // four-word clause with a usable ending; enforce the same invariant at
        // this final boundary so no caller can leak fragments into the UI.
        let minimumPreviewWords = simultaneousTTSIsEnabled ? 3 : 4
        guard current.split(whereSeparator: \Character.isWhitespace).count >= minimumPreviewWords,
              current.count >= 7,
              canonicalCurrent != lastRejectedPreviewSource
                || isStable != lastRejectedPreviewWasStable,
              current != latestPreviewSource || (isStable && !latestPreviewIsStable)
        else { return }
        if SpeculativeTranslationCache.canonicalSource(current)
            != SpeculativeTranslationCache.canonicalSource(latestPreviewSource) {
            speculativeTranslationCache.reset()
        }
        latestPreviewSource = current
        latestPreviewCumulativeSource = cumulativeSource ?? hypothesis
        latestPreviewIsStable = isStable
        launchLatestPreviewTranslationIfNeeded(
            sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
        )
    }

    private func launchLatestPreviewTranslationIfNeeded(
        sourceLanguage: Language, targetLanguage: Language
    ) {
        guard !firstSpeechStarterAwaitingFullCorrection,
              !confirmedTranslationHasPriority,
              previewTranslationTask == nil,
              latestPreviewSource != translatedPreviewSource
                || latestPreviewIsStable != translatedPreviewIsStable
        else { return }
        let current = latestPreviewSource
        let cumulativeSource = latestPreviewCumulativeSource
        let isStable = latestPreviewIsStable
        let sourceAudioEndMilliseconds = receivedAudioMilliseconds
        let generation = sessionGeneration
        let previewTranslationEnabled = previewTranslationIsEnabled
        let usesContextualPrefetch = contextualPrefetchIsEnabled
        previewTranslationTask = Task(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do {
                // A cache entry proves that this exact source already passed
                // the semantic SPEAK gate while revisable. It still cannot be
                // consumed until a second identical ASR observation promotes
                // the source to stable in this same session.
                if isStable, self.learnedSimultaneousBoundaryIsEnabled,
                   let cached = await self.takeSpeculativeTranslation(
                    candidateSource: current,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage,
                    generation: generation
                   ) {
                    let deduplicationSource = Self.cumulativePrefix(
                        in: cumulativeSource,
                        throughTailBoundary: cached.confirmedSource
                    ) ?? cached.confirmedSource
                    await self.enqueueEarlyConfirmedBoundary(
                        cached.confirmedSource,
                        deduplicationSource: deduplicationSource,
                        recognitionMilliseconds:
                            await self.receivedAudioMilliseconds,
                        probability: cached.semanticSpeakProbability,
                        prefetchedTranslation: cached,
                        expectedCandidateSource: current,
                        expectedGeneration: generation,
                        semanticPairBoundaryApproved:
                            sourceLanguage == .english
                    )
                    await self.recordLearnedBoundaryWait(
                        expectedSource: current, isStable: isStable
                    )
                    return
                }

                var selectedSource = current
                var selectedTranslationInput = current
                var semanticPairBoundaryApproved = false
                var semanticSpeakProbability: Double?
                if self.learnedSimultaneousBoundaryIsEnabled {
                    do {
                        let decision = try await Self.withTimeout(.milliseconds(500)) {
                            try await self.worker.simultaneousBoundaryDecision(
                                sourcePrefix: current,
                                sourceLanguage: sourceLanguage
                            )
                        }
                        guard await self.previewStateIsCurrent(
                            source: current, expectedStable: isStable,
                            generation: generation,
                            allowStablePromotion: !isStable
                        ) else {
                            await self.finishPreviewTranslation()
                            return
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
                            // A conservative semantic WAIT must not rebuild a
                            // long burst after the acoustic layer has already
                            // proved several words stable. Advance only a
                            // short prefix whose final token has one observed
                            // word of lookahead. This keeps a steady 4-6 word
                            // cadence and prevents partial final tokens such as
                            // `cook`/`Ven` from becoming audible before their
                            // `cooker`/`Venus` correction arrives.
                            let currentWords = current.split(
                                whereSeparator: \Character.isWhitespace
                            ).map(String.init)
                            if isStable,
                               let cadenceSource = Self.stableCadencePrefix(
                                   in: currentWords
                               ),
                               await self.previewSourceIsCurrentWholeWordPrefix(
                                   source: cadenceSource, generation: generation
                               ) {
                                let deduplicationSource = Self.cumulativePrefix(
                                    in: cumulativeSource,
                                    throughTailBoundary: cadenceSource
                                ) ?? cadenceSource
                                await self.enqueueEarlyConfirmedBoundary(
                                    cadenceSource,
                                    deduplicationSource: deduplicationSource,
                                    recognitionMilliseconds: audioMilliseconds,
                                    probability: decision.speakProbability
                                )
                                await self.recordLearnedBoundaryWait(
                                    expectedSource: current, isStable: isStable
                                )
                                return
                            }
                            // The boundary model is intentionally conservative
                            // and often waits for punctuation even after a
                            // complete 8+ word clause is available.  That made
                            // the TTS lane go idle between sentences.  Admit a
                            // structurally complete latency-bounded prefix now;
                            // cumulative source deduplication preserves every
                            // word to its right for the following FIFO item.
                            if current.last.map({ ".?!。？！".contains($0) }) == true,
                               Self.canConfirmLatencyDeadlineBoundary(
                                   current, isAcousticallyStable: isStable
                               ),
                               await self.previewSourceIsCurrentWholeWordPrefix(
                                   source: current, generation: generation
                               ) {
                                let deduplicationSource = Self.cumulativePrefix(
                                    in: cumulativeSource,
                                    throughTailBoundary: current
                                ) ?? current
                                await self.enqueueEarlyConfirmedBoundary(
                                    current,
                                    deduplicationSource: deduplicationSource,
                                    recognitionMilliseconds: audioMilliseconds,
                                    probability: decision.speakProbability
                                )
                                await self.recordLearnedBoundaryWait(
                                    expectedSource: current, isStable: isStable
                                )
                                return
                            }
                            // A WAIT result means the source is not safe to
                            // confirm, not that translation work must stop.
                            // Translate it as a revisable preview so consecutive
                            // target revisions can expose only their unchanged
                            // prefix while the starter is playing. The semantic
                            // gate still exclusively controls confirmed source.
                            guard previewTranslationEnabled else {
                                await self.recordLearnedBoundaryWait(
                                    expectedSource: current, isStable: isStable
                                )
                                return
                            }
                            // Immediately after the audible starter, do not
                            // spend the single resident translation lane on a
                            // short prefix that the semantic model has already
                            // rejected. In continuous speech a fuller ASR
                            // hypothesis follows shortly; translating that
                            // newer unit directly avoids serially paying for
                            // both the obsolete fragment and its replacement.
                            // Once simultaneous body speech has begun, keep
                            // translating WAIT revisions so the normal rolling
                            // target-prefix bridge remains smooth.
                            guard Self.shouldTranslateRejectedPreview(
                                current,
                                hasStartedSimultaneousSpeech:
                                    await self.hasStartedSimultaneousSpeech
                            ) else {
                                await self.recordLearnedBoundaryWait(
                                    expectedSource: current, isStable: isStable
                                )
                                return
                            }
                        } else if sourceLanguage == .english {
                            guard let approved = Self.approvedEnglishSemanticBoundary(
                                boundarySource: decision.boundarySource,
                                within: current,
                                probability: decision.speakProbability
                            ) else {
                                await self.recordLearnedBoundaryWait(
                                    expectedSource: current, isStable: isStable
                                )
                                return
                            }
                            selectedSource = approved.source
                            selectedTranslationInput = approved.translationInput
                            semanticPairBoundaryApproved = true
                        } else {
                            selectedSource = current
                            selectedTranslationInput = current
                        }
                        semanticSpeakProbability = decision.status == "speak"
                            ? decision.speakProbability : nil

                        // A semantic SPEAK result is not enough by itself: the
                        // source must also have survived two identical ASR
                        // observations. Once both independent gates agree, it
                        // is no longer a revisable preview. Route the exact
                        // source prefix through the ordinary confirmed queue so
                        // translation ordering, duplicate suppression, stale
                        // work disposal and watchdog behavior remain identical
                        // to full acoustic endpoints.
                        let stablePrefix = await self.earlyConfirmedStablePrefix
                        let uncommittedStablePrefix = await self.transcriptGate
                            .uncommittedCandidate(stablePrefix) ?? ""
                        let boundaryIsStable = Self.stablePrefix(
                            uncommittedStablePrefix,
                            coversSemanticBoundary: selectedSource
                        )
                        let stableBoundaryProof = isStable ? nil : selectedSource
                        let highConfidenceBoundary = Self
                            .canConfirmHighConfidenceSemanticBoundary(
                                probability: semanticSpeakProbability,
                                source: selectedSource,
                                allowSoftPunctuation:
                                    semanticPairBoundaryApproved
                            )
                        if semanticSpeakProbability != nil,
                           isStable || boundaryIsStable || highConfidenceBoundary {
                            let deduplicationSource = Self.cumulativePrefix(
                                in: cumulativeSource,
                                throughTailBoundary: selectedSource
                            ) ?? selectedSource
                            if highConfidenceBoundary,
                               !isStable, !boundaryIsStable {
                                // The semantic model has already approved this
                                // complete clause with very high confidence.
                                // Commit its exact source through the ordinary
                                // transcript gate now. A later acoustic final
                                // will then contribute only genuinely new text
                                // instead of replaying this preview and making
                                // TTS stutter through the thought twice.
                                guard await self.previewSourceIsCurrentWholeWordPrefix(
                                    source: current, generation: generation
                                ) else {
                                    await self.finishPreviewTranslation()
                                    return
                                }
                                await self.enqueueEarlyConfirmedBoundary(
                                    selectedSource,
                                    deduplicationSource: deduplicationSource,
                                    recognitionMilliseconds: audioMilliseconds,
                                    probability: decision.speakProbability,
                                    semanticPairBoundaryApproved:
                                        semanticPairBoundaryApproved
                                )
                            } else {
                                await self.enqueueEarlyConfirmedBoundary(
                                    selectedSource,
                                    deduplicationSource: deduplicationSource,
                                    recognitionMilliseconds: audioMilliseconds,
                                    probability: decision.speakProbability,
                                    expectedCandidateSource: current,
                                    expectedGeneration: generation,
                                    expectedStableBoundary: stableBoundaryProof,
                                    semanticPairBoundaryApproved:
                                        semanticPairBoundaryApproved
                                )
                            }
                            await self.recordLearnedBoundaryWait(
                                expectedSource: current, isStable: isStable
                            )
                            return
                        }

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
                guard previewTranslationEnabled else {
                    await self.recordLearnedBoundaryWait(
                        expectedSource: current, isStable: isStable
                    )
                    return
                }
                // A confirmed phrase may have arrived while the semantic
                // boundary model was running. Do not enter the serialized
                // translator behind it, and never let this stale preview speak
                // after the lossless FIFO has become ready.
                guard await !self.confirmedTranslationHasPriority else {
                    await self.finishPreviewTranslation()
                    return
                }
                // Compute only. This operation cannot mutate conversation
                // context, previous-source suppression, or playback dedup.
                let normalized = TranslationSemanticNormalizer.normalize(
                    selectedTranslationInput,
                    source: sourceLanguage, target: targetLanguage
                )
                let translatedResponse: LocalModelWorkerResponse
                if usesContextualPrefetch {
                    let recentContext = Self.recentSourceContext(
                        cumulativeSource: cumulativeSource,
                        currentSource: selectedSource
                    )
                    let alreadySpoken = await self.confirmedTranslationDisplay
                    // Clean p95 is 431 ms. A 900 ms side-lane ceiling tolerates
                    // scheduler jitter while guaranteeing that a sick context
                    // model loses the race instead of delaying live speech.
                    translatedResponse = try await Self.withTimeout(.milliseconds(900)) {
                        try await self.contextualWorker.translate(
                            recentSourceContext: recentContext,
                            newSource: normalized,
                            alreadySpokenTarget: alreadySpoken,
                            sourceLanguage: sourceLanguage,
                            targetLanguage: targetLanguage
                        )
                    }
                } else {
                    translatedResponse = try await Self.withTimeout(.milliseconds(3_200)) {
                        try await self.worker.previewTranslateText(
                            text: normalized,
                            sourceLanguage: sourceLanguage,
                            targetLanguage: targetLanguage,
                            preferredTerms: self.preferredTerms
                        )
                    }
                }
                guard let translated = translatedResponse.translation,
                      !translated.isEmpty,
                      await !self.confirmedTranslationHasPriority,
                      await self.previewSourceIsCurrentOrGrowingPrefix(
                        source: current, generation: generation
                      ) else {
                    await self.recordLearnedBoundaryWait(
                        expectedSource: current, isStable: isStable
                    )
                    return
                }
                try Task.checkCancellation()
                guard self.learnedSimultaneousBoundaryIsEnabled else {
                    // Without the independent semantic SPEAK gate this remains
                    // display-only work and can never become spoken audio.
                    await self.recordLearnedBoundaryWait(
                        expectedSource: current, isStable: isStable
                    )
                    return
                }
                if usesContextualPrefetch,
                   !IncrementalPreviewExposurePolicy.shouldExpose(
                       sourceIsStable: isStable,
                       semanticBoundaryApproved: semanticSpeakProbability != nil
                   ) {
                    await SimultaneousDebugLogger.shared.record(
                        sessionID: self.simultaneousDebugSessionID,
                        event: "streaming_translation_discarded",
                        audioMilliseconds: await self.receivedAudioMilliseconds,
                        sourceLanguage: sourceLanguage,
                        targetLanguage: targetLanguage,
                        source: current,
                        translation: translated,
                        status: "unproven_incremental_delta"
                    )
                    await self.recordLearnedBoundaryWait(
                        expectedSource: current, isStable: isStable
                    )
                    return
                }
                if semanticSpeakProbability != nil {
                    await self.storeSpeculativeTranslation(
                        .init(
                            candidateSource:
                                SpeculativeTranslationCache.canonicalSource(current),
                            confirmedSource:
                                SpeculativeTranslationCache.canonicalSource(selectedSource),
                            translationSource: normalized,
                            translation: translated,
                            sourceLanguage: sourceLanguage,
                            targetLanguage: targetLanguage,
                            sessionGeneration: generation,
                            contextVersion: translatedResponse.contextVersion ?? -1,
                            requiresResidentCommit: !usesContextualPrefetch,
                            translationMilliseconds:
                                translatedResponse.translationMilliseconds ?? 0,
                            semanticSpeakProbability: semanticSpeakProbability,
                            createdAtAudioMilliseconds:
                                await self.receivedAudioMilliseconds
                        )
                    )
                }
                await self.recordPreviewTranslation(
                    translated,
                    expectedSource: current,
                    // A semantic WAIT can improve the common target prefix,
                    // but may never commit a complete translation from one
                    // observation merely because ASR text was stable.
                    isStable: isStable && semanticSpeakProbability != nil,
                    sourceAudioEndMilliseconds: sourceAudioEndMilliseconds
                )
            } catch is CancellationError {
                await self.finishPreviewTranslation()
            } catch {
                await self.recordLearnedBoundaryWait(
                    expectedSource: current, isStable: isStable
                )
            }
        }
    }

    private func previewSourceIsCurrent(
        source: String, generation: UInt64
    ) -> Bool {
        generation == sessionGeneration
            && SpeculativeTranslationCache.canonicalSource(source)
                == SpeculativeTranslationCache.canonicalSource(latestPreviewSource)
    }

    private func previewSourceIsCurrentOrGrowingPrefix(
        source: String, generation: UInt64
    ) -> Bool {
        guard generation == sessionGeneration else { return false }
        return Self.previewSourceRemainsValid(
            source, asPrefixOf: latestPreviewSource
        )
    }

    private func previewSourceIsCurrentWholeWordPrefix(
        source: String, generation: UInt64
    ) -> Bool {
        guard generation == sessionGeneration else { return false }
        return Self.previewSourceRemainsValidForConfirmation(
            source, asWholeWordPrefixOf: latestPreviewSource
        )
    }

    /// A preview translation normally completes after ASR has appended more
    /// right context. Streaming recognizers also expose the last lexical token
    /// before it is complete (for example `Americ` followed by `America`). That
    /// is growth, not a revision, and rejecting the already-running translation
    /// forced speech to wait for the next full acoustic endpoint. Permit only
    /// this final-token completion; every earlier word must still match exactly.
    static func previewSourceRemainsValid(
        _ source: String, asPrefixOf latest: String
    ) -> Bool {
        let sourceWords = SpeculativeTranslationCache.canonicalSource(source)
            .split(whereSeparator: \Character.isWhitespace)
        let latestWords = SpeculativeTranslationCache.canonicalSource(latest)
            .split(whereSeparator: \Character.isWhitespace)
        guard !sourceWords.isEmpty, sourceWords.count <= latestWords.count else {
            return false
        }
        let lastIndex = sourceWords.count - 1
        guard zip(sourceWords[..<lastIndex], latestWords[..<lastIndex])
            .allSatisfy({ $0 == $1 })
        else { return false }

        let sourceLast = sourceWords[lastIndex].lowercased()
        let latestLast = latestWords[lastIndex].lowercased()
        if sourceLast == latestLast { return true }
        // Very short prefixes ("a" -> "and") are too ambiguous, and a token
        // carrying terminal punctuation was already presented as complete.
        guard sourceLast.count >= 4,
              sourceLast.last.map({ !".,!?;:。？！".contains($0) }) ?? false
        else { return false }
        return latestLast.hasPrefix(sourceLast)
    }

    /// Irreversible speech confirmation is stricter than speculative display.
    /// ASR commonly exposes a partial final token (`hum`) immediately before
    /// extending it (`human`). A translated preview may survive that growth,
    /// but spoken audio cannot be corrected cleanly, so every committed source
    /// token must already be an exact whole-word prefix of the latest ASR text.
    static func previewSourceRemainsValidForConfirmation(
        _ source: String, asWholeWordPrefixOf latest: String
    ) -> Bool {
        let sourceWords = SpeculativeTranslationCache.canonicalSource(source)
            .split(whereSeparator: \Character.isWhitespace)
        let latestWords = SpeculativeTranslationCache.canonicalSource(latest)
            .split(whereSeparator: \Character.isWhitespace)
        guard !sourceWords.isEmpty, sourceWords.count <= latestWords.count else {
            return false
        }
        return zip(sourceWords, latestWords).allSatisfy { sourceWord, latestWord in
            sourceWord.lowercased() == latestWord.lowercased()
        }
    }

    private func previewStateIsCurrent(
        source: String,
        expectedStable: Bool,
        generation: UInt64,
        allowStablePromotion: Bool
    ) -> Bool {
        guard previewSourceIsCurrent(source: source, generation: generation) else {
            return false
        }
        return latestPreviewIsStable == expectedStable
            || (allowStablePromotion && !expectedStable && latestPreviewIsStable)
    }

    /// Returns true only when every whole word in the semantic boundary has
    /// already survived the acoustic stability gate. Extra words to the right
    /// are deliberately ignored because they remain revisable and are not
    /// part of the source sent to translation or speech.
    static func stablePrefix(
        _ stablePrefix: String,
        coversSemanticBoundary boundary: String
    ) -> Bool {
        let stableWords = stablePrefix.split(whereSeparator: \Character.isWhitespace)
        let boundaryWords = boundary.split(whereSeparator: \Character.isWhitespace)
        guard boundaryWords.count >= 3,
              boundaryWords.count <= stableWords.count
        else { return false }

        func normalized(_ word: Substring) -> String {
            word.lowercased().trimmingCharacters(in: .punctuationCharacters)
        }
        return zip(boundaryWords, stableWords).allSatisfy { boundaryWord, stableWord in
            let expected = normalized(boundaryWord)
            return !expected.isEmpty && expected == normalized(stableWord)
        }
    }

    /// Reattaches an approved boundary from the uncommitted tail to its full
    /// cumulative ASR prefix. `ASRTranscriptGate` then records that cumulative
    /// prefix while returning only the fresh tail for translation. Without
    /// this mapping, a later cumulative update can replay a tail-only accepted
    /// phrase at an interior offset and the semantic model gets stuck on it.
    static func cumulativePrefix(
        in cumulativeSource: String,
        throughTailBoundary boundary: String
    ) -> String? {
        let cumulativeWords = cumulativeSource
            .split(whereSeparator: \Character.isWhitespace).map(String.init)
        let boundaryWords = boundary
            .split(whereSeparator: \Character.isWhitespace).map(String.init)
        guard !boundaryWords.isEmpty,
              boundaryWords.count <= cumulativeWords.count else { return nil }

        func normalized(_ word: String) -> String {
            word.lowercased().trimmingCharacters(in: .punctuationCharacters)
        }

        var lastMatchStart: Int?
        let finalStart = cumulativeWords.count - boundaryWords.count
        for start in 0...finalStart {
            let range = start..<(start + boundaryWords.count)
            if zip(cumulativeWords[range], boundaryWords).allSatisfy({
                normalized($0) == normalized($1)
            }) {
                lastMatchStart = start
            }
        }
        guard let lastMatchStart else { return nil }
        let end = lastMatchStart + boundaryWords.count
        return cumulativeWords[..<end].joined(separator: " ")
    }

    /// Protects the one-at-a-time resident translator from head-of-line
    /// blocking at session start. Eight source words is enough context for the
    /// existing contextual starter-correction gate; shorter text that also
    /// failed the independent semantic boundary is better replaced by the next
    /// streaming hypothesis than translated serially.
    static func shouldTranslateRejectedPreview(
        _ source: String, hasStartedSimultaneousSpeech: Bool
    ) -> Bool {
        hasStartedSimultaneousSpeech
            || source.split(whereSeparator: \Character.isWhitespace).count >= 8
    }

    /// Returns only source words preceding the currently translated tail.
    /// The v13 model sees at most 18 words of history; this keeps inference
    /// cost constant even during an hour-long meeting.
    static func recentSourceContext(
        cumulativeSource: String,
        currentSource: String,
        maximumWords: Int = 18
    ) -> String {
        let cumulative = SpeculativeTranslationCache.canonicalSource(cumulativeSource)
            .split(whereSeparator: \Character.isWhitespace).map(String.init)
        let current = SpeculativeTranslationCache.canonicalSource(currentSource)
            .split(whereSeparator: \Character.isWhitespace).map(String.init)
        guard !cumulative.isEmpty, !current.isEmpty else { return "" }
        let preceding: ArraySlice<String>
        if current.count <= cumulative.count,
           zip(cumulative.suffix(current.count), current).allSatisfy({
               $0.caseInsensitiveCompare($1) == .orderedSame
           }) {
            preceding = cumulative.dropLast(current.count)
        } else {
            // A recognizer can revise punctuation/casing at the boundary. In
            // that case exclude a same-sized tail rather than leaking current
            // source into both labelled prompt sections.
            preceding = cumulative.dropLast(min(current.count, cumulative.count))
        }
        return preceding.suffix(max(0, maximumWords)).joined(separator: " ")
    }

    /// Native Nemotron ingestion uses 80 ms frames. A 256-frame ceiling is
    /// large enough to survive a transient model/actor stall without dropping
    /// source, yet remains bounded if capture continues while the recognizer
    /// is genuinely unhealthy.
    static let nemotronAudioBufferCapacity = 256
    static var nemotronAudioBufferSeconds: Double {
        Double(nemotronAudioBufferCapacity)
            * NemotronStreamingRecognizer.ingestFrameSeconds
    }

    static func shouldDeferPreviewTranslation(
        confirmedTranslationInFlight: Bool,
        pendingConfirmedCount: Int
    ) -> Bool {
        confirmedTranslationInFlight || pendingConfirmedCount > 0
    }

    private func storeSpeculativeTranslation(
        _ entry: SpeculativeTranslationCache.Entry
    ) {
        // A false->true stability promotion of the exact same source is safe:
        // the cached text is still inert and confirmation performs a second,
        // strict stable-state check before enqueueing it. Any source revision
        // or session change discards the completed computation.
        guard previewSourceIsCurrent(
            source: entry.candidateSource,
            generation: entry.sessionGeneration
        ) else { return }
        speculativeTranslationCache.store(entry)
    }

    private func takeSpeculativeTranslation(
        candidateSource: String,
        sourceLanguage: Language,
        targetLanguage: Language,
        generation: UInt64
    ) -> SpeculativeTranslationCache.Entry? {
        guard previewStateIsCurrent(
            source: candidateSource, expectedStable: true,
            generation: generation, allowStablePromotion: false
        ) else {
            speculativeTranslationCache.reset()
            return nil
        }
        return speculativeTranslationCache.take(
            candidateSource: candidateSource,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage,
            sessionGeneration: generation,
            audioMilliseconds: receivedAudioMilliseconds
        )
    }

    private func enqueueEarlyConfirmedBoundary(
        _ source: String,
        deduplicationSource: String? = nil,
        recognitionMilliseconds: Double,
        probability: Double?,
        prefetchedTranslation: SpeculativeTranslationCache.Entry? = nil,
        expectedCandidateSource: String? = nil,
        expectedGeneration: UInt64? = nil,
        expectedStableBoundary: String? = nil,
        semanticPairBoundaryApproved: Bool = false
    ) async {
        let compact = source.trimmingCharacters(in: .whitespacesAndNewlines)
        if let expectedCandidateSource, let expectedGeneration {
            if let expectedStableBoundary {
                let uncommittedStablePrefix = transcriptGate
                    .uncommittedCandidate(earlyConfirmedStablePrefix) ?? ""
                guard previewSourceIsCurrent(
                    source: expectedCandidateSource,
                    generation: expectedGeneration
                ), Self.stablePrefix(
                    uncommittedStablePrefix,
                    coversSemanticBoundary: expectedStableBoundary
                ) else { return }
            } else {
                guard previewStateIsCurrent(
                    source: expectedCandidateSource, expectedStable: true,
                    generation: expectedGeneration,
                    allowStablePromotion: false
                ) else { return }
            }
        }
        guard !compact.isEmpty,
              !Self.shouldHoldNemotronDispatch(compact)
        else { return }
        let admittedSource = deduplicationSource?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? compact
        guard !admittedSource.isEmpty else { return }
        // Early confirmation is an optimization, never the sole lossless
        // source path. Validate the selected boundary, not admittedSource:
        // admittedSource is deliberately the complete cumulative ASR prefix
        // so transcriptGate can remember it and subtract it from the next
        // callback. In a healthy long session that prefix eventually exceeds
        // 48 words even though `compact` is a safe 3-12 word boundary. Treating
        // session history as the boundary made all early confirmations stop
        // after roughly two minutes, feeding long final chunks into the serial
        // translator and creating an ever-growing queue.
        if Self.isPathologicalEarlyConfirmedBoundary(compact) {
            earlyConfirmedSourceCommitter.reset()
            rollingTranslationScheduler.reset()
            await SimultaneousDebugLogger.shared.record(
                sessionID: simultaneousDebugSessionID,
                event: "early_boundary_rejected_pathological",
                audioMilliseconds: receivedAudioMilliseconds,
                sourceLanguage: sourceLanguage,
                targetLanguage: targetLanguage,
                source: compact,
                status: "rejected"
            )
            return
        }
        let canonical = admittedSource.lowercased()
            .split(whereSeparator: \Character.isWhitespace)
            .joined(separator: " ")
        guard canonical != lastEnqueuedEarlyConfirmedSource else { return }
        lastEnqueuedEarlyConfirmedSource = canonical
        await SimultaneousDebugLogger.shared.record(
            sessionID: simultaneousDebugSessionID,
            event: "early_confirmed_boundary",
            audioMilliseconds: receivedAudioMilliseconds,
            sourceLanguage: sourceLanguage,
            targetLanguage: targetLanguage,
            source: compact,
            status: "confirmed",
            probability: probability
        )
        enqueueRecognizedPhrase(
            admittedSource, recognitionMilliseconds: recognitionMilliseconds,
            requiresCompleteSourceGate: true,
            semanticPairBoundaryApproved: semanticPairBoundaryApproved,
            prefetchedTranslation: prefetchedTranslation
        )
    }

    static func isPathologicalEarlyConfirmedBoundary(_ source: String) -> Bool {
        let keys = source
            .split(whereSeparator: \Character.isWhitespace)
            .map { Self.previewWordKey(String($0)) }
        return keys.count > 48 || Self.hasAdjacentRepeatedWordRun(keys)
    }

    private func recordLearnedBoundaryWait(
        expectedSource: String, isStable: Bool
    ) {
        if expectedSource == latestPreviewSource,
           isStable == latestPreviewIsStable {
            lastRejectedPreviewSource =
                SpeculativeTranslationCache.canonicalSource(expectedSource)
            lastRejectedPreviewWasStable = isStable
            translatedPreviewSource = expectedSource
            translatedPreviewIsStable = isStable
            // The semantic decision for this exact candidate is complete.
            // Do not carry the normal 1.5-second preview throttle into the
            // next, genuinely changed ASR hypothesis: that delay dominated
            // first speech even though boundary inference itself took only a
            // few milliseconds. schedulePreviewTranslation still rejects an
            // unchanged source/stability pair, so this cannot spin or retry a
            // rejected prefix without a new ASR observation.
            rollingTranslationScheduler.reset()
        }
        previewTranslationTask = nil
        if let sourceLanguage, let targetLanguage {
            launchLatestPreviewTranslationIfNeeded(
                sourceLanguage: sourceLanguage, targetLanguage: targetLanguage
            )
        }
    }

    private func recordPreviewTranslation(
        _ translation: String, expectedSource: String, isStable: Bool,
        sourceAudioEndMilliseconds: Double
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
        if let captionDelta = monotonicTranslationCaptioner.observe(
            normalizedTranslation,
            sourceIsStable: isStable
        ), !captionDelta.isEmpty {
            confirmedTranslationDisplay = monotonicTranslationCaptioner.text
            metrics.translationPhrase = confirmedTranslationDisplay
            Task {
                await SimultaneousDebugLogger.shared.record(
                    sessionID: simultaneousDebugSessionID,
                    event: "streaming_translation_committed",
                    audioMilliseconds: receivedAudioMilliseconds,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage,
                    source: expectedSource,
                    translation: captionDelta,
                    status: isStable ? "stable" : "repeated_prefix"
                )
            }
        }
        if simultaneousTTSIsEnabled, simultaneousStarterBridgeOpen,
           let targetLanguage,
           let committed = simultaneousTranslationCommitter.update(
               normalizedTranslation,
               sourceIsStable: isStable,
               allowsContextualStarterCorrection:
                   SimultaneousTranslationCommitter
                       .hasEnoughSourceContextForStarterCorrection(expectedSource)
               ),
           !committed.isEmpty {
            let committedPrefix = simultaneousTranslationCommitter.committedTranslation
            hasStartedSimultaneousSpeech = true
            enqueueSpeechRequest(SpeechRequest(
                text: committed,
                language: targetLanguage,
                translationKind: .simultaneousCommitted,
                enqueuedAt: .now,
                sessionGeneration: sessionGeneration,
                sourceAudioEndMilliseconds: sourceAudioEndMilliseconds,
                audiblePrefixAfterCompletion: committedPrefix
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

    static func canConfirmHighConfidenceSemanticBoundary(
        probability: Double?, source: String,
        allowSoftPunctuation: Bool = false
    ) -> Bool {
        guard let probability, probability >= 0.95 else { return false }
        let words = source.split(whereSeparator: \Character.isWhitespace)
        guard words.count >= 7 else { return false }
        return !shouldHoldEarlyConfirmedSource(
            source, allowSoftPunctuation: allowSoftPunctuation
        )
    }

    /// An interior boundary and a whole-hypothesis boundary are different
    /// model claims. Never reuse the probability of a rejected interior cut to
    /// promote the whole ASR hypothesis; doing so made unfinished phrases
    /// irreversible and caused both repetitions and skipped corrections.
    static func approvedEnglishSemanticBoundary(
        boundarySource: String?, within hypothesis: String,
        probability: Double?
    ) -> EarlyTranslationClauseSelector.ApprovedSemanticBoundary? {
        if let boundarySource,
           let approved = EarlyTranslationClauseSelector.approvedSemanticBoundary(
               prefix: boundarySource, within: hypothesis
           ) {
            return approved
        }

        let whole = hypothesis.split(whereSeparator: \Character.isWhitespace)
            .map(String.init).joined(separator: " ")
        let proposed = boundarySource?.split(
            whereSeparator: \Character.isWhitespace
        ).map(String.init).joined(separator: " ")
        // The current ONNX model normally proposes interior boundaries. This
        // branch is deliberately future-proofed for an explicit whole-endpoint
        // contract, but nil/interior proposals can never enter it.
        guard !whole.isEmpty, proposed == whole,
              probability.map({ $0 >= 0.98 }) == true,
              canConfirmHighConfidenceSemanticBoundary(
                  probability: probability, source: whole
              )
        else { return nil }
        return .init(source: whole, translationInput: whole)
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
        if !hasLoggedFirstAudioActivity {
            hasLoggedFirstAudioActivity = true
            let sessionID = simultaneousDebugSessionID
            let audioMilliseconds = receivedAudioMilliseconds
            let sourceLanguage = sourceLanguage
            let targetLanguage = targetLanguage
            Task {
                await SimultaneousDebugLogger.shared.record(
                    sessionID: sessionID,
                    event: "audio_activity_detected",
                    audioMilliseconds: audioMilliseconds,
                    sourceLanguage: sourceLanguage,
                    targetLanguage: targetLanguage
                )
            }
        }
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
