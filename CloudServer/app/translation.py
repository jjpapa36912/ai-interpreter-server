from __future__ import annotations

import re
import threading
from collections import deque

from .prompts import translation_messages
from .settings import Settings


class CUDATranslator:
    """Resident EN/KO translator with bounded conversational context."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._context: dict[tuple[str, str, str], deque[tuple[str, str]]] = {}

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is not available")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.settings.translation_model, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.settings.translation_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: self.settings.model_max_memory, "cpu": "48GiB"},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).eval()

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        session_id: str,
    ) -> str:
        text = " ".join(text.split()).strip()
        if not text:
            raise ValueError("text must not be empty")
        self.load()
        import torch

        key = (session_id, source_language, target_language)
        history = self._context.setdefault(key, deque(maxlen=3))
        messages = translation_messages(
            text, source_language, target_language, tuple(history)
        )
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._generation_lock, torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                use_cache=True,
            )
        generated = output[0, inputs.input_ids.shape[-1]:]
        translation = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        translation = re.sub(r"(?s)^.*?</think>", "", translation).strip()
        translation = translation.splitlines()[0].strip().strip('"')
        if not translation:
            raise RuntimeError("translation model returned empty output")
        history.append((text, translation))
        return translation

    def reset(self, session_id: str) -> None:
        for key in [key for key in self._context if key[0] == session_id]:
            del self._context[key]

