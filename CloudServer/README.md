# AI Interpreter CUDA Server

This is the NVIDIA/CUDA boundary for the macOS interpreter. The first deployable
slice keeps a resident Qwen3 translator on the GPU and exposes authenticated
health, readiness, translation, and context-reset endpoints. Streaming ASR and
TTS are added behind the same API in the next slices.

## gcube settings

- Registry: GitHub Container Registry (`ghcr.io`)
- Container port: `8000`
- GPU: RTX 5090 32 GB
- Minimum CUDA: `12.8`
- Shared memory: `8 GB`
- Istio proxy: enabled
- Replica count: `1`

Set `AI_INTERPRETER_SERVER_TOKEN` to a long random value in the workload
environment. Do not put it in the image or source tree.

## API smoke test

```bash
curl "$SERVICE_URL/health"
curl -H "Authorization: Bearer $AI_INTERPRETER_SERVER_TOKEN" "$SERVICE_URL/ready"
curl -X POST "$SERVICE_URL/v1/translate" \
  -H "Authorization: Bearer $AI_INTERPRETER_SERVER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The launch is not out of the woods yet.","source_language":"en","target_language":"ko","session_id":"smoke"}'
```

The first translation downloads and loads the model, so it is intentionally a
cold-start request. Keep one replica resident during a test session and stop the
workload immediately afterward.

