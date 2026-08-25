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
    AI_INTERPRETER_ASR_COMPUTE_TYPE=float16

WORKDIR /srv/ai-interpreter
COPY CloudServer/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY CloudServer/app ./app
COPY CloudServer/tools ./tools
COPY CloudServer/scripts ./scripts

# Download public base weights before copying the frequently updated adapter,
# allowing GHCR's previous image to reuse this large immutable layer.
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

RUN chmod +x ./scripts/start-server.sh

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["./scripts/start-server.sh"]
