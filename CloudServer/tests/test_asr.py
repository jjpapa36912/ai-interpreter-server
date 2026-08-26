import unittest

from app.asr import (
    ConfirmedPhraseAccumulator, StableASRCommitter, StreamCommitLedger,
    StreamingASRSession,
)


class StableASRCommitterTests(unittest.TestCase):
    def test_confirmed_phrase_accumulator_keeps_later_speech_connected(self):
        phrases = ConfirmedPhraseAccumulator()
        self.assertEqual(phrases.append("Before Friday's interview,"), "")
        self.assertEqual(phrases.append("please ask Sarah"), "")
        self.assertEqual(phrases.append("Chen to confirm"), "")
        self.assertEqual(phrases.append("whether the revised launch budget"), "")
        self.assertEqual(phrases.append("is final."),
                         "Before Friday's interview, please ask Sarah Chen to confirm "
                         "whether the revised launch budget is final.")

    def test_confirmed_phrase_accumulator_bounds_unpunctuated_first_phrase(self):
        phrases = ConfirmedPhraseAccumulator()
        self.assertEqual(phrases.append("하나 둘 셋"), "")
        self.assertEqual(phrases.append("넷 다섯 여섯 일곱"), "")
        self.assertEqual(phrases.append("여덟"),
                         "하나 둘 셋 넷 다섯 여섯 일곱 여덟")

    def test_comma_does_not_split_idiom_from_right_context(self):
        phrases = ConfirmedPhraseAccumulator()
        self.assertEqual(phrases.append("The rollout is back on track,"), "")
        self.assertEqual(
            phrases.append("but we are not out of the woods yet."),
            "The rollout is back on track, but we are not out of the woods yet.",
        )

    def test_confirmed_phrase_accumulator_flushes_final_remainder(self):
        phrases = ConfirmedPhraseAccumulator()
        self.assertEqual(phrases.append("short final phrase", final=True),
                         "short final phrase")
    def test_committer_emits_only_new_stable_words(self):
        committer = StableASRCommitter(
            lookahead_words=2, agreement_decodes=2, minimum_commit_words=1
        )
        self.assertEqual(committer.update("we need a", final=False)["committed_delta"], "")
        first = committer.update("we need a better plan today", final=False)
        self.assertEqual(first["committed_delta"], "we")
        second = committer.update("we need a better plan tomorrow", final=False)
        self.assertEqual(second["committed_delta"], "need a")
        final = committer.update("we need a better plan tomorrow", final=True)
        self.assertEqual(final["committed_delta"], "better plan tomorrow")


    def test_committer_never_repeats_committed_prefix(self):
        committer = StableASRCommitter(
            lookahead_words=1, agreement_decodes=2, minimum_commit_words=1
        )
        committer.update("hello from the meeting")
        self.assertEqual(
            committer.update("hello from the meeting today")["committed_delta"],
            "hello from the",
        )
        self.assertEqual(
            committer.update("hello from the meeting today again")["committed_delta"],
            "meeting",
        )

    def test_revision_aligns_after_committed_suffix(self):
        committer = StableASRCommitter(
            lookahead_words=2, agreement_decodes=2, minimum_commit_words=1
        )
        committer.update("yeah")
        self.assertEqual(committer.update("yeah")["committed_delta"], "yeah")
        committer.update("yeah yeah yeah to just have sort of")
        self.assertEqual(
            committer.update("yeah yeah yeah to just have sort of")["committed_delta"],
            "yeah yeah to just have",
        )
        committer.update("yeah to just have the same sort of idiom")
        final = committer.update("yeah to just have the same sort of idiom", final=True)
        self.assertEqual(final["committed_delta"], "the same sort of idiom")

    def test_default_requires_two_decodes_before_commit(self):
        committer = StableASRCommitter()
        self.assertEqual(committer.update("다음 주 화요일에 만나요")["committed_delta"], "")
        self.assertEqual(
            committer.update("다음 주 화요일에 만나요 여러분")["committed_delta"],
            "다음 주 화요일에",
        )


    def test_streaming_session_decodes_on_bounded_cadence(self):
        session = StreamingASRSession(
            sample_rate=10, minimum_decode_seconds=0.8, decode_interval_seconds=0.6
        )
        self.assertFalse(session.append(b"\0\0" * 7))
        self.assertTrue(session.append(b"\0\0"))
        session.mark_decoded()
        self.assertFalse(session.append(b"\0\0" * 5))
        self.assertTrue(session.append(b"\0\0"))


    def test_streaming_session_rejects_partial_pcm_sample(self):
        session = StreamingASRSession()
        with self.assertRaisesRegex(ValueError, "complete samples"):
            session.append(b"\0")

    def test_stream_ledger_deduplicates_buffer_seam_and_keeps_total(self):
        ledger = StreamCommitLedger()
        first = ledger.apply({"committed_delta": "talking about", "committed_words": 2})
        self.assertEqual(first["committed_words"], 2)
        ledger.begin_new_buffer()
        second = ledger.apply({"committed_delta": "about the stuff", "committed_words": 3})
        self.assertEqual(second["committed_delta"], "the stuff")
        self.assertEqual(second["committed_words"], 4)


if __name__ == "__main__":
    unittest.main()
