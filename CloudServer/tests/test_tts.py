import unittest
from unittest.mock import Mock

import numpy as np

from app.settings import Settings
from app.tts import CUDANeuralTTS


class CUDANeuralTTSTests(unittest.TestCase):
    def test_synthesize_returns_clamped_little_endian_pcm(self):
        engine = CUDANeuralTTS(Settings())
        engine.model = Mock()
        engine.model.generate_custom_voice.return_value = (
            [np.asarray([-2.0, 0.0, 2.0], dtype=np.float32)],
            24_000,
        )
        engine.loaded = True

        pcm, sample_rate = engine.synthesize("안녕하세요", "ko")

        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(np.frombuffer(pcm, dtype="<i2").tolist(), [-32767, 0, 32767])
        kwargs = engine.model.generate_custom_voice.call_args.kwargs
        self.assertEqual(kwargs["language"], "Korean")
        self.assertEqual(kwargs["speaker"], "Sohee")

    def test_rejects_unsupported_language_before_loading(self):
        engine = CUDANeuralTTS(Settings())
        with self.assertRaisesRegex(ValueError, "en or ko"):
            engine.synthesize("bonjour", "fr")

    def test_stream_silence_cleanup_spans_chunks(self):
        state = {"started": False, "pending_silence": 0}
        first = CUDANeuralTTS._remove_excess_silence(
            np.r_[np.zeros(1_000), np.ones(100), np.zeros(3_000)], 1_000, state
        )
        second = CUDANeuralTTS._remove_excess_silence(
            np.ones(100), 1_000, state
        )

        # Leading padding is gone and the 3 s cross-chunk pause is 160 ms.
        self.assertEqual(len(first), 100)
        self.assertEqual(len(second), 260)
        self.assertTrue(np.all(second[:160] == 0))


if __name__ == "__main__":
    unittest.main()
