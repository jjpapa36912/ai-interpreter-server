#!/usr/bin/env python3
"""Limit qwen-tts initialization to the 12 Hz model used by this service.

qwen-tts 0.1.1 eagerly imports its legacy 25 Hz tokenizer, which requires a
separate SoX/torchaudio stack even when serving only a 12 Hz checkpoint.  That
unrelated import is both unnecessary and ABI-fragile in CUDA containers.
"""

from pathlib import Path
import os
import site


def locate_package() -> Path:
    configured = os.getenv("AI_INTERPRETER_QWEN_TTS_PACKAGE")
    if configured:
        package = Path(configured)
        if package.is_dir():
            return package
        raise RuntimeError(f"configured qwen_tts package does not exist: {package}")
    for root in map(Path, site.getsitepackages()):
        package = root / "qwen_tts"
        if package.is_dir():
            return package
    raise RuntimeError("qwen_tts package directory was not found")


def main() -> None:
    package = locate_package()
    (package / "__init__.py").write_text(
        "from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem\n"
        "__all__ = ['Qwen3TTSModel', 'VoiceClonePromptItem']\n",
        encoding="utf-8",
    )
    (package / "core" / "__init__.py").write_text(
        "from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import "
        "Qwen3TTSTokenizerV2Config\n"
        "from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import "
        "Qwen3TTSTokenizerV2Model\n",
        encoding="utf-8",
    )
    tokenizer = package / "inference" / "qwen3_tts_tokenizer.py"
    source = tokenizer.read_text(encoding="utf-8")
    source = source.replace("    Qwen3TTSTokenizerV1Config,\n", "")
    source = source.replace("    Qwen3TTSTokenizerV1Model,\n", "")
    source = source.replace(
        '        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)\n'
        "        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)\n\n",
        "",
    )
    tokenizer.write_text(source, encoding="utf-8")
    print(f"patched 12 Hz-only qwen_tts initialization at {package}")


if __name__ == "__main__":
    main()
