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
        self.assertIn("군더더기 소리", kwargs["instruct"])
        self.assertIn("입력된 문장만 정확히", kwargs["instruct"])

    def test_rejects_unsupported_language_before_loading(self):
        engine = CUDANeuralTTS(Settings())
        with self.assertRaisesRegex(ValueError, "en or ko"):
            engine.synthesize("bonjour", "fr")

    def test_requested_fast_delivery_is_generated_by_the_voice_model(self):
        engine = CUDANeuralTTS(Settings())
        engine.model = Mock()
        engine.model.generate_custom_voice.return_value = (
            [np.asarray([0.0], dtype=np.float32)], 24_000,
        )
        engine.loaded = True

        engine.synthesize("자연스럽게 따라갑니다.", "ko", speed=1.22)

        instruct = engine.model.generate_custom_voice.call_args.kwargs["instruct"]
        self.assertIn("20퍼센트 빠르게", instruct)
        self.assertIn("음절을 늘이지 마세요", instruct)

    def test_accelerated_backend_yields_true_streaming_chunks(self):
        engine = CUDANeuralTTS(Settings())
        engine.model = Mock()
        engine.model.generate_custom_voice_streaming.return_value = iter([
            (np.asarray([0.1], dtype=np.float32), 24_000, None),
            (np.asarray([0.2], dtype=np.float32), 24_000, None),
        ])
        engine.loaded = True
        engine.streaming_available = True

        chunks = list(engine.synthesize_stream("확정된 문장입니다.", "ko"))

        self.assertEqual(len(chunks), 2)
        kwargs = engine.model.generate_custom_voice_streaming.call_args.kwargs
        self.assertFalse(kwargs["non_streaming_mode"])
        self.assertEqual(kwargs["chunk_size"], 4)

    def test_accelerated_streaming_is_disabled_by_default(self):
        settings = Settings()

        self.assertFalse(settings.tts_experimental_streaming)
        self.assertTrue(CUDANeuralTTS(settings)._use_accelerated_backend(True))

if __name__ == "__main__":
    unittest.main()
