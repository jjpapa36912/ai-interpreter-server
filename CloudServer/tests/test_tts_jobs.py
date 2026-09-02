import unittest

from app.tts_jobs import TTSJobStore


class TTSJobStoreTests(unittest.TestCase):
    def test_splits_pcm_into_retry_safe_numbered_chunks(self):
        store = TTSJobStore(sample_rate=10, chunk_milliseconds=200)
        job = store.create()
        self.assertIsNotNone(job)

        job.append(b"abcdefghij", store.maximum_chunk_bytes)
        job.finish()

        self.assertEqual(store.maximum_chunk_bytes, 4)
        self.assertEqual(job.wait_for(0, 0).chunk, b"abcd")
        self.assertEqual(job.wait_for(1, 0).chunk, b"efgh")
        last = job.wait_for(2, 0)
        self.assertEqual(last.chunk, b"ij")
        self.assertTrue(last.done)

    def test_waiting_reader_observes_completion_without_audio(self):
        store = TTSJobStore()
        job = store.create()
        self.assertIsNotNone(job)
        job.finish()

        snapshot = job.wait_for(0, 0)
        self.assertIsNone(snapshot.chunk)
        self.assertTrue(snapshot.done)
        self.assertIsNone(snapshot.error)

    def test_capacity_counts_only_active_jobs(self):
        store = TTSJobStore(maximum_jobs=1)
        first = store.create()
        self.assertIsNotNone(first)
        self.assertIsNone(store.create())
        first.finish()
        self.assertIsNotNone(store.create())

    def test_request_id_is_idempotent_and_generation_is_claimed_once(self):
        store = TTSJobStore()
        first = store.create(request_id="speech-1")
        second = store.create(request_id="speech-1")

        self.assertIsNotNone(first)
        self.assertIs(first, second)
        self.assertTrue(first.claim_generation())
        self.assertFalse(second.claim_generation())

    def test_failure_is_visible_to_polling_client(self):
        store = TTSJobStore()
        job = store.create()
        self.assertIsNotNone(job)
        job.fail("generation failed")

        snapshot = job.wait_for(0, 0)
        self.assertTrue(snapshot.done)
        self.assertEqual(snapshot.error, "generation failed")


if __name__ == "__main__":
    unittest.main()
