# Reuse the already-published model cache. Updating application code or the TTS
# runtime must not download the multi-GB model snapshots again on every build.
FROM ghcr.io/jjpapa36912/ai-interpreter-server:prototype AS cached-models

# NVIDIA NGC avoids the Docker Hub pull path that failed on gcube and includes
# a Blackwell-compatible CUDA/PyTorch runtime for the RTX 5090.
FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    AI_INTERPRETER_TRANSLATION_MODEL=google/madlad400-3b-mt \
    AI_INTERPRETER_TRANSLATION_ADAPTER=/models/madlad400-3b-en-ko-v3-adapter \
    AI_INTERPRETER_MODEL_MAX_MEMORY=14GiB \
    AI_INTERPRETER_ASR_MODEL=turbo \
    AI_INTERPRETER_ASR_BEAM_SIZE=1 \
    AI_INTERPRETER_ASR_COMPUTE_TYPE=float16 \
    AI_INTERPRETER_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

WORKDIR /srv/ai-interpreter
COPY --from=cached-models /models/huggingface /models/huggingface
COPY CloudServer/requirements.txt ./requirements.txt
COPY CloudServer/scripts/patch_qwen_tts_12hz.py ./scripts/patch_qwen_tts_12hz.py
RUN python -m pip install --no-cache-dir -r requirements.txt
# NGC 25.08 ships a PyTorch 2.8 development snapshot. Pairing that snapshot
# with a released torchaudio wheel causes an undefined-symbol ABI failure.
# Install the official CUDA 12.8 PyTorch family together; the
# RTX 5090 driver is backward-compatible with this CUDA runtime.
RUN PIP_CONSTRAINT= python -m pip install --no-cache-dir --force-reinstall \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
# Avoid qwen-tts pulling another generic torch build over the verified pair.
# Keep imports in separate layers so CI annotations identify the exact binary
# or package that fails instead of collapsing three checks into one message.
RUN python -c "import torch; print(torch.__version__)"
RUN python -c "import torchaudio; print(torchaudio.__version__)"
RUN python -c "import torchvision; print(torchvision.__version__)"
RUN python -m pip install --no-cache-dir --no-deps qwen-tts==0.1.1
RUN python -m pip install --no-cache-dir --no-deps faster-qwen3-tts==0.3.2
RUN python ./scripts/patch_qwen_tts_12hz.py

# Keep model layers independent of frequently changed application code. Future
# server-only fixes then reuse the multi-GB weights instead of downloading them.
COPY CloudServer/scripts/preload_models.py ./scripts/preload_models.py
RUN python ./scripts/preload_models.py

COPY CloudServer/models/madlad400-3b-en-ko-v3-adapter \
    /tmp/madlad400-3b-en-ko-v3-adapter
RUN mkdir -p /models/madlad400-3b-en-ko-v3-adapter && \
    cp /tmp/madlad400-3b-en-ko-v3-adapter/*.json \
       /tmp/madlad400-3b-en-ko-v3-adapter/spiece.model \
       /models/madlad400-3b-en-ko-v3-adapter/ && \
    cat /tmp/madlad400-3b-en-ko-v3-adapter/adapter_model.safetensors.part-* \
       > /models/madlad400-3b-en-ko-v3-adapter/adapter_model.safetensors && \
    echo "78c04013a677e63ed4ecd6c3563148f337f820235820a37bb444f3752b00b20f  /models/madlad400-3b-en-ko-v3-adapter/adapter_model.safetensors" \
       | sha256sum -c - && \
    rm -rf /tmp/madlad400-3b-en-ko-v3-adapter

COPY CloudServer/app ./app
COPY CloudServer/tools ./tools
COPY CloudServer/scripts ./scripts

RUN chmod +x ./scripts/start-server.sh

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["./scripts/start-server.sh"]
