# sherpa_stt — 한국어 스트리밍 STT (초기 구현)

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)의 스트리밍 transducer 모델로 수업 음성을 실시간 전사하고, 팀 데이터 계약 v0.1.0의 `transcript_partial`/`transcript_final` 이벤트를 [통합 서비스](../integration/service.py)에 발행하는 모듈입니다. 청크 시각은 영상 분석과 같은 세션 공통 시계(`service.elapsed_ms(session_id)`)를 사용합니다.

> 상태: 프런트에서 선택적으로 연결할 수 있습니다. 라이브 세션은 출력 루프백 또는 마이크 중 하나를 선택해 같은 `IntegrationService`로 전사 이벤트를 전달합니다. 아래 "확인 결과"는 한 대의 Windows 11 PC, Python 3.13 임시 환경에서 확인한 것이며 저장소 기준 Python 3.14 환경과 실제 Zoom 수업에서의 검증은 별도입니다.

## 처리 흐름

```text
LoopbackCapture (SoundCard WASAPI 루프백: 스피커로 재생되는 소리)   ┐ 100ms 청크
MicrophoneCapture (sounddevice: 마이크)                            ┘ 둘 중 하나 선택
  ↓ AudioChunk (메모리 전용) + ChunkClock: 세션 공통 시계 기준 start_ms/end_ms
SherpaTranscriber
  ├─ to_mono_float32 → SherpaEngine (sherpa-onnx OnlineRecognizer)
  ├─ 텍스트 변경 시 transcript_partial (revision 증가)
  ├─ 엔드포인트 감지 시 transcript_final (is_edited: false)
  │    구간 = 토큰 타임스탬프를 세션 시각으로 변환한 추정값
  └─ 청크 공백 > max_gap_ms 이면 현재 발화 확정 + analysis_status(audio_unavailable)
  ↓ service.make_event() → service.receive_event()
IntegrationService (부분 전사는 메모리만, 확정 전사는 세션 JSON에 저장)
```

## 파일

| 파일 | 내용 |
|---|---|
| [config.py](config.py) | `SttConfig`: 모델 경로·ID·버전, 샘플레이트, 엔드포인트 규칙, 공백 허용치. `public_info()`는 로컬 경로를 뺀 공개 설정 |
| [audio.py](audio.py) | 계약 5절 `AudioChunk`, mono float32 변환, 세션 시계 기준 청크 시각 `ChunkClock` |
| [engine.py](engine.py) | `SherpaEngine`: 모델 파일 탐색, 인식기 생성, 토큰 기반 한국어 띄어쓰기 복원과 토큰 구간 |
| [download_model.py](download_model.py) | 기본 한국어 모델을 `sherpa_stt/models/`에 내려받기 |
| [transcriber.py](transcriber.py) | `SherpaTranscriber`: 청크 → 전사 이벤트, 상태 이벤트 |
| [loopback.py](loopback.py) | `LoopbackCapture`: 출력 장치 루프백 → `AudioChunk` 큐 |
| [microphone.py](microphone.py) | `MicrophoneCapture`: 마이크 → `AudioChunk` 큐 |
| [runner.py](runner.py) | `LiveTranscription`: 입력·전사 스레드 실행, 시작/종료 상태 이벤트, 실시간 계측 `stats()` |
| [demo.py](demo.py) | 콘솔 데모(루프백·마이크·WAV) |
| [tests/](tests/) | 가짜 엔진 기반 단위 테스트 |

## 시간축 맞춤

- 영상 분석([integrated_analysis.py](../video_recognition/integrated_analysis.py))은 프레임을 받을 때 `service.elapsed_ms(session_id)`를 `capture_ms`로 씁니다. STT 청크도 같은 서비스·같은 세션의 같은 함수를 기준으로 하므로, **같은 `IntegrationService`와 `session_id`를 넘기면** 두 결과가 한 시간축에 놓입니다.
- `ChunkClock`은 도착 지터(약 ±40ms)를 없애려고 샘플 수로 시각을 이어 붙입니다. 세션 시계보다 200ms 넘게 뒤처지면 다시 맞추고, 앞서면 끌어내립니다. 끝 시각은 현재 세션 시각을 넘지 않습니다.
- 발화 구간: 첫 토큰 - 200ms ~ 마지막 토큰 + 400ms를 세션 시각으로 바꾼 뒤 현재 세그먼트 안으로 제한합니다. 단어 정렬이 아닌 추정값이며 여유값은 실험값입니다. 토큰 타임스탬프가 없으면 세그먼트 전체(직전 확정 이후 ~ 엔드포인트)를 씁니다.
- 출력 장치 지연(재생 → 루프백 수신)은 보정하지 않습니다. 화면 캡처 지연도 영상 쪽에서 보정하지 않으므로 둘 다 수십~수백 ms 오차가 있을 수 있습니다.

## 계약 대응

- `provenance.module`은 `speech`, `model_id`/`model_version`은 `SttConfig` 값입니다.
- `analysis_status.payload.component`는 `stt`입니다. 모델 로드 실패는 `unavailable` + `model_not_configured`, 입력 장치 실패·중단이나 청크 공백은 `audio_unavailable`로 알립니다.
- 루프백은 재생되는 소리가 없을 때도 0으로 채운 블록이 연속으로 들어옵니다(실측). 이것은 실제 출력이 무음인 상태입니다. 장치 누락을 무음으로 위장하는 것이 아닙니다.
- 전사 텍스트는 sherpa-onnx 결과의 `text`가 아니라 토큰을 이어 붙여 만듭니다. `text`는 한글 사이 공백을 지우기 때문입니다(`주민등록증을보여주시겠어요?` → `주민등록증을 보여 주시겠어요?`). 직전 발화에서 넘어온 앞쪽 구두점 토큰은 버립니다.
- 확정 전사의 `revision`은 같은 발화의 마지막 부분 전사 `revision`과 같습니다.
- 원본 오디오는 파일로 쓰지 않습니다. 세션 JSON에는 확정 전사만 들어갑니다. 전사 텍스트도 개인 식별 가능 정보이므로 실제 수업 결과를 커밋하지 않습니다.
- 세션 종료 전에 `LiveTranscription.stop()` 또는 `SherpaTranscriber.finish()`를 호출해야 마지막 발화가 확정됩니다. 종료된 세션은 `transcript_final`을 받지 않습니다.

## 설치

```powershell
video_recognition/.venv/Scripts/python.exe -m pip install -r sherpa_stt/requirements.txt
```

Python 3.13 임시 환경에서 `sherpa-onnx 2.0.1`, `SoundCard 0.4.6`, `numpy 2.5.4`의 설치·실행을 확인했습니다. 저장소 기준 환경(Python 3.14)에서는 확인하지 않았습니다.

### 모델

기본값은 `sherpa-onnx-streaming-zipformer-korean-2024-06-16`(압축 약 420MB)입니다. 아래 명령으로 기본 위치에 받으면 `--model-dir`를 생략할 수 있습니다. `sherpa_stt/models/`는 `.gitignore`로 커밋에서 제외됩니다.

```powershell
video_recognition/.venv/Scripts/python.exe -m sherpa_stt.download_model
```

네트워크가 중간에 끊기면 `curl.exe -L -C - -o ko.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2`로 이어받은 뒤 `sherpa_stt/models/`에 압축을 풉니다.

```text
sherpa_stt/models/sherpa-onnx-streaming-zipformer-korean-2024-06-16/
  tokens.txt
  encoder-epoch-99-avg-1.int8.onnx   ← 사용 (int8 우선)
  decoder-epoch-99-avg-1.onnx        ← 사용 (원본 우선)
  joiner-epoch-99-avg-1.int8.onnx    ← 사용 (int8 우선)
  test_wavs/                         ← 동작 확인용 샘플과 trans.txt
```

## 실행

저장소 루트에서 실행합니다.

```powershell
# 출력 장치(루프백) 목록 / 마이크 목록
video_recognition/.venv/Scripts/python.exe -m sherpa_stt.demo --list-devices
video_recognition/.venv/Scripts/python.exe -m sherpa_stt.demo --source microphone --list-devices
# 스피커로 재생되는 수업 프로그램·영상 음성 전사 (기본, Ctrl+C로 종료)
video_recognition/.venv/Scripts/python.exe -m sherpa_stt.demo
# 마이크 전사
video_recognition/.venv/Scripts/python.exe -m sherpa_stt.demo --source microphone --device 1
# 16-bit PCM WAV를 실시간 속도로 전사하고 세션 JSON 저장
video_recognition/.venv/Scripts/python.exe -m sherpa_stt.demo --wav sherpa_stt/models/sherpa-onnx-streaming-zipformer-korean-2024-06-16/test_wavs/0.wav --save out/session.json
```

종료하면 실시간 계측(`rtf`, 청크 처리 지연 `lag_ms_*`)을 출력합니다. 이 값은 세션 JSON에 저장하지 않습니다.

영상 분석과 같은 세션에 붙이는 코드:

```python
from sherpa_stt.config import SttConfig
from sherpa_stt.engine import SherpaEngine
from sherpa_stt.runner import LiveTranscription

config = SttConfig()                 # 기본: sherpa_stt/models/ 아래 한국어 모델
engine = SherpaEngine(config)        # 약 5초. 세션 시작 전에 불러 둔다
# service, sid = 영상 분석이 쓰는 IntegrationService와 세션 ID
live = LiveTranscription(service, sid, config, source="loopback", engine=engine)
live.start()
...
live.stop()                          # 반드시 service.stop_session(sid) 보다 먼저
service.stop_session(sid)
```

## 확인 결과

2026-11-05, Windows 11 PC 한 대(Realtek 스피커), Python 3.13 임시 환경에서 확인했습니다. CER 평가는 하지 않았습니다.

**1) 파일 직접 입력**: 모델 동봉 `test_wavs`를 100ms 청크로 넣었습니다.

| 파일 | 정답(trans.txt) | 인식 |
|---|---|---|
| 0.wav | 그는 괜찮은 척하려고 애쓰는 것 같았다. | 걔는 괜찮은 척하려구 애 쓰는 거 같았다 |
| 1.wav | 지하철에서 다리를 벌리고 앉지 마라. | 지하철에서 다리를 벌리고 하진 마라. |
| 2.wav | 부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집 아이와 비교하는 것이다. | 부모가 저질른 큰 실수 중 하나는 자기 아이를 다른 집안과 비교하는 것이다. |
| 3.wav | 주민등록증을 보여 주시겠어요? | 주민등록증을 보여 주시겠어요? |

**2) 루프백 실시간**: 세션 시작 후 같은 WAV 4개를 스피커로 재생하고, 루프백으로 받아 전사했습니다. 재생 구간은 재생 시작·종료 시점의 세션 시각이고, WAV 자체의 앞뒤 무음이 포함됩니다.

| 파일 | 재생 구간(ms) | 전사 구간(ms) | 인식 |
|---|---|---|---|
| 0.wav | 1675~5201 | 2150~6150 | 걔는 괜찮은 척하려고 에쓰는 것 같았다. |
| 1.wav | 7467~10871 | 7910~11270 | 지하철에서 다리를 벌리고 앉지 마라. |
| 2.wav | 13002~19630 | 13269~20150 | 부모가 저지르는 큰 실수 중 하나는 자기 아이를 다른 집안과 비교하는 것이다. |
| 3.wav | 21732~24412 | 22070~24710 | 주민등록증을 보여 주시겠어요? |

- 실시간 계측: RTF 0.16~0.18(실시간보다 약 5~6배 빠름), 청크 처리 지연 중앙값 46~59ms, 최대 163ms, 버린 청크 0개
- 확정 이벤트 생성 시각(`created_ms`)은 재생 종료 후 약 1.2~2.0초입니다. 대부분 엔드포인트 무음 대기(1.2초)입니다.
- 루프백 시험 4회 중 1회는 시작 직후 STT가 `unavailable`로 멈췄습니다. 이후 재현되지 않았고 원인은 확인하지 못했습니다.

## 한계

- **선택한 창의 소리만 분리하지 않습니다.** 루프백은 선택한 출력 장치로 나가는 모든 소리(알림음, 다른 앱 소리 포함)를 받습니다. 창 단위 분리에는 Windows 프로세스 루프백 API가 필요합니다.
- 루프백은 스피커로 재생되는 소리만 받습니다. 교수자 본인의 목소리는 수업 프로그램이 되돌려 재생하지 않으므로 대개 루프백에 들어가지 않습니다. 교수자 발화를 전사하려면 `source="microphone"`을 쓰거나 두 입력을 함께 쓰는 방식을 정해야 합니다.
- 실제 수업 프로그램(Zoom 등) 음성, 장시간(10분 이상) 동작, 출력 장치 변경·블루투스 전환은 확인하지 않았습니다.

## 테스트

```powershell
video_recognition/.venv/Scripts/python.exe -m unittest discover -s sherpa_stt/tests -t .
```

numpy만 있으면 실행됩니다. sherpa-onnx·오디오 장치는 필요하지 않습니다.

## 남은 일

- 마이크와 출력 루프백 동시 수집
- 선택한 창의 오디오만 분리하는 Windows 프로세스 루프백
- 실제 수업 음성으로 CER(목표 20% 미만)과 표시 지연(목표 5초 이내) 측정
- 청크 형식·길이·부분/확정 경계는 계약 10절 "STT" 항목으로 팀 합의 필요
