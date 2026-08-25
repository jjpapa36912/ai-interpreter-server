# UniversalInterpreter

macOS 양방향 실시간 통역 기술 검증 프로젝트입니다. 듣기와 말하기를 동시에 제어하되 각 세션의 오류와 리소스 정리를 독립적으로 처리합니다.

PHASE 9에서 번역된 영어 PCM을 48 kHz mono 공유 메모리 ring buffer를 통해 Audio Server Plug-in에 연결했습니다. PHASE 10은 선택한 단일 앱만 캡처하고 `excludesCurrentProcessAudio`를 사용해 번역 재생음이 다시 입력되는 루프를 방지합니다. 드라이버의 실제 HAL 로딩은 유효한 Apple Development 서명이 필요합니다.

PHASE 12 호환성 진단은 Zoom, Chrome/Google Meet, Microsoft Teams, Safari, Discord를 식별합니다. Google Meet는 Chrome 프로세스 단위로 캡처되므로 같은 Chrome 프로세스에서 재생되는 다른 탭 오디오가 함께 포함될 수 있습니다.

PHASE 13 Debug Metrics는 양방향 세션별 실행시간, 입력·출력 오디오 시간, 첫 번역 오디오 응답시간, 변환 큐 깊이와 드롭 비율을 실제 스트림에서 측정합니다.

PHASE 14는 듣기·말하기 모드 선택, 실행 앱 선택, 통합 시작/중지, 컴포넌트 상태와 현재 지연을 한 화면에 배치합니다. 테스트 톤과 세부 성능 수치는 접을 수 있는 개발 진단 영역에 분리했습니다.

권한 패널에서 마이크와 화면·시스템 오디오 녹음 권한을 확인하고, 권한 요청 또는 해당 macOS 개인정보 보호 설정으로 바로 이동할 수 있습니다.

## 개발 실행

전체 Xcode 설치 후 `Package.swift`를 열거나 터미널에서 다음을 실행합니다.

```sh
swift run UniversalInterpreter
```

권한 설명을 포함한 macOS 앱 번들은 다음 명령으로 `outputs/AI Interpreter.app`에 생성합니다.

```sh
./scripts/build-app.sh
```

기본값은 로컬 테스트용 ad-hoc 서명입니다. 개인정보 보호 권한을 안정적으로 유지하려면 Keychain에 등록된 인증서 이름을 지정합니다.

```sh
APP_SIGNING_IDENTITY="Apple Development: Dong Jun Kim (T2Z2N3K335)" ./scripts/build-app.sh
```

현재 개발 인증서로 서명하고 앱을 바로 실행하려면 다음 명령을 사용합니다.

```sh
./scripts/sign-and-open-app.sh
```

시스템 설정에서 토글이 켜져 있는데도 TCC가 접근을 거부하면 현재 앱의 화면 녹화 권한 기록만 초기화합니다.

```sh
./scripts/reset-screen-capture-permission.sh
```

통역 실행 경로는 외부 Live API를 사용하지 않습니다. `말투 보존` 모드는 ModelLab의 로컬 모델과 고정 화자 합성을 사용하고, `절약 모드`는 macOS 로컬 프레임워크를 사용합니다.

`GPU 동시통역`은 자체 배포한 CUDA 서버의 WebSocket ASR와 번역 모델을 사용합니다. 서버의 HTTP 서비스 주소를 지정한 뒤 앱 안에서 `GPU 동시통역`을 선택합니다.

```sh
export AI_INTERPRETER_STREAMING_SERVER_URL="https://your-service.example"
export AI_INTERPRETER_STREAMING_SERVER_TOKEN="replace_if_configured"
./scripts/sign-and-open-app.sh
```

서버 장애 시에는 앱에서 기존 모드로 즉시 전환하거나 다음 환경변수로 GPU 경로를 강제 비활성화할 수 있습니다.

```sh
export AI_INTERPRETER_DISABLE_GPU_STREAMING=1
./scripts/sign-and-open-app.sh
```

배포용 로컬 모델 상태를 확인하거나 표준 Application Support 경로에 연결하려면 다음을 사용합니다. `--copy`는 독립 실행용으로 전체 런타임을 복사하므로 약 11GB 이상의 추가 공간이 필요합니다.

```sh
./scripts/install-model-runtime.sh --check
./scripts/install-model-runtime.sh --link
```

최종 실제 테스트 전에는 정적 품질 게이트를 실행합니다. 이 검사는 모델·의존성·번역 기준선·말투 기준선·앱 연결 및 배포 구성을 확인하며 Metal 실시간 측정은 마지막 통합 테스트까지 보류합니다.

```sh
PYTHONPATH=ModelLab ModelLab/.venv/bin/python \
  ModelLab/scripts/final_quality_gate.py \
  --output ModelLab/results/final-static-gate.json
```

가상 마이크가 이전 문장을 반복하는 경우 최신 ring-buffer 보호 코드로 드라이버를 다시 설치합니다.

```sh
./scripts/rebuild-and-install-virtual-mic.sh
```
