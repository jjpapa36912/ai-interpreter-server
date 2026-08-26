import unittest

from app.asr import StableASRCommitter, StreamCommitLedger, StreamingASRSession


class StableASRCommitterTests(unittest.TestCase):
    def test_committer_emits_only_new_stable_words(self):
        committer = StableASRCommitter(lookahead_words=2, agreement_decodes=2)
        self.assertEqual(committer.update("we need a", final=False)["committed_delta"], "")
        first = committer.update("we need a better plan today", final=False)
        self.assertEqual(first["committed_delta"], "we")
        second = committer.update("we need a better plan tomorrow", final=False)
        self.assertEqual(second["committed_delta"], "need a")
        final = committer.update("we need a better plan tomorrow", final=True)
        self.assertEqual(final["committed_delta"], "better plan tomorrow")


    def test_committer_never_repeats_committed_prefix(self):
        committer = StableASRCommitter(lookahead_words=1, agreement_decodes=2)
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
        committer = StableASRCommitter(lookahead_words=2, agreement_decodes=2)
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
        self.assertEqual(committer.update("다음 주")["committed_delta"], "")
        self.assertEqual(committer.update("다음 좋아요")["committed_delta"], "다음")
        self.assertEqual(committer.update("다음 주 화요일")["committed_delta"], "")


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
