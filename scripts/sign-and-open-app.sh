#!/bin/zsh

set -euo pipefail

project_root="${0:A:h:h}"
nemotron_model="${project_root}/ModelLab/models/trained/nemotron-asr-acoustic-v2.q8_0.gguf"
requested_identity="${APP_SIGNING_IDENTITY:-}"
signing_identity="${requested_identity}"

# codesign rejects a duplicated certificate name as ambiguous. Resolve the
# requested display name to one certificate's unique SHA-1 identity first.
if [[ -z "${signing_identity}" ]]; then
    # New Macs can generate a certificate whose display name differs from the
    # previous machine. Select any valid Apple Development identity instead of
    # pinning the old certificate name and team suffix.
    signing_identity="$(security find-identity -v -p codesigning \
        "${HOME}/Library/Keychains/login.keychain-db" 2>/dev/null | \
        awk '/"Apple Development:/ { print $2; exit }')"
elif [[ ! "${signing_identity}" =~ '^[[:xdigit:]]{40}$' ]]; then
    identity_hash="$(security find-identity -v -p codesigning \
        "${HOME}/Library/Keychains/login.keychain-db" 2>/dev/null | \
        awk -v identity="${signing_identity}" 'index($0, "\"" identity "\"") { print $2; exit }')"
    if [[ -z "${identity_hash}" ]]; then
        print -u2 "사용 가능한 개발자 서명 인증서를 찾지 못했습니다: ${signing_identity}"
        exit 1
    fi
    signing_identity="${identity_hash}"
fi

if [[ -z "${signing_identity}" ]]; then
    print -u2 "사용 가능한 Apple Development 인증서와 개인 키가 없습니다."
    print -u2 "Xcode > Settings > Accounts > 계정 선택 > Manage Certificates… > + > Apple Development를 생성하세요."
    exit 1
fi

cd "${project_root}"
# `open` reuses an already-running process even after its bundle was rebuilt.
# Stop only this development app so the new binary is guaranteed to launch.
pkill -TERM -x UniversalInterpreter 2>/dev/null || true
APP_SIGNING_IDENTITY="${signing_identity}" ./scripts/build-app.sh
if [[ "${OPEN_APP:-1}" == "1" ]]; then
    if [[ "${HYBRID_GPU_TTS:-0}" == "1" ]]; then
        if [[ -z "${AI_INTERPRETER_STREAMING_SERVER_URL:-}" ]]; then
            print -u2 "AI_INTERPRETER_STREAMING_SERVER_URL이 설정되지 않았습니다."
            exit 1
        fi
        # GPU streaming ASR removes Nemotron's long first-hypothesis lookahead.
        # The validated local translator keeps translation quality, and its
        # final text returns over the same WebSocket for GPU neural speech.
        open --env AI_INTERPRETER_DISABLE_REMOTE_TRANSLATION=1 \
            --env AI_INTERPRETER_DISABLE_GPU_STREAMING=0 \
            --env AI_INTERPRETER_GPU_LOCAL_TRANSLATION=1 \
            --env AI_INTERPRETER_SIMULTANEOUS_TTS=0 \
            --env AI_INTERPRETER_TTS_SERVER_URL="${AI_INTERPRETER_TTS_SERVER_URL:-${AI_INTERPRETER_STREAMING_SERVER_URL}}" \
            --env AI_INTERPRETER_TTS_SERVER_TOKEN="${AI_INTERPRETER_TTS_SERVER_TOKEN:-${AI_INTERPRETER_STREAMING_SERVER_TOKEN:-}}" \
            --env AI_INTERPRETER_NEMOTRON_MODEL="${nemotron_model}" \
            "${project_root}/outputs/AI Interpreter.app"
        print "AI Interpreter를 로컬 번역 + GPU WebSocket 음성으로 실행했습니다."
    elif [[ "${DISABLE_GPU_TRANSLATION:-1}" == "1" ]]; then
        # Force a genuinely local process even when launchctl still contains a
        # streaming-server URL from an earlier GPU test.  Previously only the
        # legacy HTTP translator was disabled, so the app silently selected
        # GPU Streaming and made a "local" validation hit G-Cube.
        open --env AI_INTERPRETER_DISABLE_REMOTE_TRANSLATION=1 \
            --env AI_INTERPRETER_DISABLE_GPU_STREAMING=1 \
            --env AI_INTERPRETER_SIMULTANEOUS_TTS=0 \
            --env AI_INTERPRETER_NEMOTRON_MODEL="${nemotron_model}" \
            "${project_root}/outputs/AI Interpreter.app"
        print "AI Interpreter를 GPU 번역 서버 없이 빌드하고 실행했습니다."
    else
        if [[ -z "${AI_INTERPRETER_STREAMING_SERVER_URL:-}" ]]; then
            print -u2 "AI_INTERPRETER_STREAMING_SERVER_URL이 설정되지 않았습니다."
            exit 1
        fi
        open \
            --env AI_INTERPRETER_DISABLE_REMOTE_TRANSLATION=0 \
            --env AI_INTERPRETER_DISABLE_GPU_STREAMING=0 \
            --env AI_INTERPRETER_STREAMING_SERVER_URL="${AI_INTERPRETER_STREAMING_SERVER_URL}" \
            --env AI_INTERPRETER_STREAMING_SERVER_TOKEN="${AI_INTERPRETER_STREAMING_SERVER_TOKEN:-}" \
            --env AI_INTERPRETER_TTS_SERVER_URL="${AI_INTERPRETER_TTS_SERVER_URL:-${AI_INTERPRETER_STREAMING_SERVER_URL}}" \
            --env AI_INTERPRETER_TTS_SERVER_TOKEN="${AI_INTERPRETER_TTS_SERVER_TOKEN:-${AI_INTERPRETER_STREAMING_SERVER_TOKEN:-}}" \
            --env AI_INTERPRETER_NEMOTRON_MODEL="${nemotron_model}" \
            "${project_root}/outputs/AI Interpreter.app"
        print "AI Interpreter를 GPU 번역 서버와 함께 빌드하고 실행했습니다."
    fi
else
    print "AI Interpreter를 개발자 서명으로 빌드했습니다."
fi
