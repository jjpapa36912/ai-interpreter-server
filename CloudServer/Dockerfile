# Build incrementally from the last known-good production image. The base
# already contains the CUDA/PyTorch stack, runtime dependencies, model cache,
# and trained translation adapter.
FROM ghcr.io/jjpapa36912/ai-interpreter-server:prototype

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

# The streaming wrapper is pure Python and the required runtime packages are
# already present. Do not re-resolve or replace the working CUDA environment.
RUN python -m pip install --no-cache-dir --no-deps faster-qwen3-tts==0.3.2

COPY CloudServer/app ./app
COPY CloudServer/tools ./tools
COPY CloudServer/scripts ./scripts
RUN chmod +x ./scripts/start-server.sh

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["./scripts/start-server.sh"]
