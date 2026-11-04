# VS Code + Windows 한국어 sherpa-onnx STT (팀 공통 데이터 계약 0.1.0)

모드: MP4/MKV/MP3/WAV 파일 전사 (`file`), 마이크 실시간 (`mic`), Windows 스피커 출력 캡처 (`loopback`). 팀 계약에 맞는 `transcript_partial` / `transcript_final` 이벤트를 생성하며 확정 전사만 `output/session.json`에 기록합니다. 전사 구간은 추정치이며 단어 수준 강제 정렬이 아닙니다.

## 준비물

- Windows 10/11, VS Code + Python (Microsoft) extension
- Python 3.11 (권장). `py -3.11 --version`
- FFmpeg: `winget install --id Gyan.FFmpeg -e`; 새 터미널에서 `ffmpeg -version`

## 1. VS Code 터미널(PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PowerShell 실행 정책 오류 시 **그 터미널에서만** `Set-ExecutionPolicy -Scope Process Bypass` 실행 후 활성화하세요. VS Code의 `Python: Select Interpreter`에서 `.venv` 선택.

## 2. 한국어 모델 자동 다운로드

```powershell
.\download_model.ps1
```

다운로드 모델: `sherpa-onnx-streaming-zipformer-korean-2024-06-16` (한국어 전용). `models/` 안에 압축이 해제됩니다. **모델 수백 MB는 ZIP에 포함되지 않습니다.** 다운로드 실패 시 공식 모델 링크를 참조하세요.

## 3. 실행 순서

### (A) 모델 샘플 WAV로 1차 성공 확인

```powershell
python run_stt.py --mode file --input "models/sherpa-onnx-streaming-zipformer-korean-2024-06-16/test_wavs/0.wav"
```

### (B) 동영상에서 음성 추출 + 전사

```powershell
python run_stt.py --mode file --input "C:\Users\USER\Videos\lecture.mp4" --title "DB 수업"
```

영상 속 음성만 FFmpeg로 16kHz/mono/float32 PCM으로 변환하여 스트리밍 인식기에 공급합니다. `--timeline-offset-ms 1500`을 쓰면 음성 이벤트 타임라인 시작을 1.5초 뒤로 옮깁니다. **파일 자체의 비디오/오디오 PTS는 자동으로 보정하지 않으므로** 필요 시 영상/음성 시작 오프셋을 측정해야 합니다.

### (C) 실시간 마이크

```powershell
python run_stt.py --list-devices
python run_stt.py --mode mic
python run_stt.py --mode mic --device 1
```

마이크 인덱스는 각 PC마다 다릅니다. 중지 `Ctrl+C`.

### (D) Windows 시스템 소리 (Zoom/Teams/브라우저)

```powershell
python run_stt.py --mode loopback
```

Windows 기본 스피커에서 재생되는 모든 소리가 입력됩니다. Zoom 소리가 다른 출력 장치로 나가면 캡처되지 않을 수 있습니다. 이 방법은 앱별 분리 캡처가 아닙니다. 헤드셋이 기본 출력 장치인지 확인하세요.

### (E) 전사 이벤트 디버깅

```powershell
python run_stt.py --mode file --input "lecture.mp4" --events-jsonl output/debug.jsonl
```

`events-jsonl`에는 임시 전사도 기록되므로 **실제 수업 자료에서는 사용하지 않는 것이 좋습니다.**

## 4. 결과 확인

`output/session.json` (팀 문서 형식):

```json
{
  "schema_version": "0.1.0",
  "session": {"session_id": "UUID", "title": "수업", "started_at_utc": "UTC", "duration_ms": 10000},
  "config": {"config_version": "0.1.0", "speech_model_id": "sherpa-onnx-streaming-zipformer-korean-2024-06-16", "speech_model_version": "2024-06-16"},
  "events": [
    {
      "schema_version": "0.1.0", "session_id": "UUID", "event_id": "UUID",
      "event_type": "transcript_final", "start_ms": 0, "end_ms": 3000,
      "created_ms": 3500, "created_at_utc": null,
      "payload": {"utterance_id": "UUID", "revision": 2, "text": "안녕하세요", "is_edited": false},
      "quality": {"status": "valid", "reason_codes": []},
      "provenance": {"module": "speech", "config_version": "0.1.0", "model_id": "...", "model_version": "2024-06-16"}
    }
  ]
}
```

이것은 **형식 예시**이며 실제 출력은 모델 결과, UUID, 시각에 따라 달라집니다.

## 5. 정확도 평가

강의 음성을 사람이 직접 받아 적어 `reference.txt`(UTF-8)에 저장한 뒤:

```powershell
python evaluate_cer.py --reference reference.txt --session output/session.json
```

이 CER은 공백 및 기호를 제거하는 문자 정규화 기준입니다. 보고서에는 이 기준을 함께 명시하세요.

## 6. 단위 테스트

```powershell
python -m unittest discover -s tests -v
```

단위 테스트는 가짜 인식기를 사용하여 이벤트/저장을 점검합니다. 모델 인식 성능·Windows 캡처 동작은 **별도 실기기 테스트 필요**.

## 7. 제한 사항

- 파일 전사는 FFmpeg를 통한 음성 트랙 추출이며 영상 프레임을 분석하지 않습니다.
- `start_ms/end_ms`: 오디오 스트림의 대략적인 발화 경계, 무음이 포함될 수 있음. 정확한 단어/영상 프레임 정렬을 보장하지 않음.
- 실시간 입력의 시작 시점은 첫 청크 수신 시각을 통해 추정합니다. **팀의 영상·오디오 캡처가 공통 세션 시계를 사용하도록 통합 시 조정**해야 합니다.
- `created_ms`: CLI 오프라인 파일 모드에서는 분석 실행 시각 기준이며 원본 영상 시각과 다른 개념입니다.
- `session.json`에는 전사 내용이 저장되므로 실제 수업 파일을 GitHub에 올리지 마세요.
- CER 20% 이하 및 평균 지연 5초 이하는 팀 제안서의 목표이지 구현만으로 보장되지 않습니다.
- 입력 장치 중단/끊김 자동 복구, 상태 이벤트 발행, 입력 장치 변경 등의 견고화는 통합 과정에서 필요합니다.

## 8. 코드 구조

```text
VSCode_Sherpa_STT/
├── run_stt.py            # file/mic/loopback CLI
├── stt_core.py           # 팀 스키마 AudioChunk / Session / STTProcessor
├── evaluate_cer.py       # CER 계산
├── download_model.ps1    # 한국어 모델 다운로드
├── requirements.txt      # 의존성
├── INTEGRATION.md        # 팀원 연동 안내
├── tests/test_contract.py
└── .gitignore
```

공식 한국어 모델 문서: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html
