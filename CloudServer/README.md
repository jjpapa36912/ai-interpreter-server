# AI Interpreter CUDA Server

This is the NVIDIA/CUDA boundary for interpreter clients. The production
prototype keeps resident faster-whisper Turbo and MADLAD-400 3B + EN/KO v3
adapter models on the GPU and exposes authenticated
health, readiness, streaming ASR, translation, and context-reset endpoints.

## gcube settings

- Registry: GitHub Container Registry (`ghcr.io`)
- Container port: `8000`
- GPU: RTX 5090 32 GB
- Minimum CUDA: `12.8`
- Shared memory: `8 GB`
- Istio proxy: enabled
- Replica count: `1`

Create two HTTP services on the same workload, both targeting container port
`8000`: one public URL for ASR/translation and a second public URL for TTS.
Set the client variables independently:

```bash
AI_INTERPRETER_STREAMING_SERVER_URL=https://ASR-SERVICE
AI_INTERPRETER_TTS_SERVER_URL=https://TTS-SERVICE
```

The separate ingress routes keep the long-lived ASR WebSocket from blocking
the streaming TTS response. They do not require a second GPU or replica.

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

`/health` becomes available while models are loading. Do not start a live session
until `/ready` returns `{"status":"ready"}`. This guarantees that the first utterance
does not pay model-download or model-load latency. Keep one replica resident during
a test session and stop the workload immediately afterward.

The image contains the public base-model snapshots and reconstructs the trained
adapter from repository-safe chunks during the GitHub Actions build. A G-Cube
deployment must not require `ssh-copy-id`, SCP, manual Dropbear restarts, or
runtime model downloads. If `/ready` reports any other model names, the old
image is running and the app test must not begin.

Validate a new deployment before opening the app:

```bash
AI_INTERPRETER_SERVER_TOKEN="$TOKEN" python CloudServer/tools/validate_deployment.py \
  --url "$SERVICE_URL"
```

## Streaming ASR protocol

Connect to `wss://SERVICE/v1/asr/stream?language=en&sample_rate=16000` with the
same bearer token, then send mono 16 kHz signed little-endian PCM16 as binary
frames. Server messages contain a provisional `hypothesis` and a monotonic
`committed_delta`; consumers must translate/speak only `committed_delta`. Send
the text message `finish` at end of stream. A close code of `1013` means the
resident ASR model is not ready yet.
