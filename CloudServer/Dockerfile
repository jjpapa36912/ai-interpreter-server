# NVIDIA NGC avoids the Docker Hub pull path that failed on gcube and includes
# a Blackwell-compatible CUDA/PyTorch runtime for the RTX 5090.
FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    AI_INTERPRETER_TRANSLATION_MODEL=Qwen/Qwen3-8B \
    AI_INTERPRETER_MODEL_MAX_MEMORY=18GiB

WORKDIR /srv/ai-interpreter
COPY CloudServer/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY CloudServer/app ./app

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

