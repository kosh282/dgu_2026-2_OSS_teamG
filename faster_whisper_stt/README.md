# faster-whisper 실시간 비교

이 폴더는 sherpa-onnx 전사와 비교하기 위한 독립 실험입니다. 프로젝트의 기존 STT 경로를 바꾸지 않습니다. faster-whisper는 스트리밍 인식기가 아니므로 최근 오디오 창(기본 8초)을 1.5초마다 다시 인식해 콘솔의 부분 결과를 갱신합니다. 같은 결과가 연속 두 번 나오면 콘솔에 안정화 표시를 합니다. 겹치는 창은 독립 발화가 아니므로 이 비교 데모는 전사 이벤트나 세션 JSON을 만들지 않습니다. 오디오 창은 메모리에서만 유지합니다.

## 별도 설치

저장소 루트에서 실행합니다. 별도 Python 3.13 설치가 있으면 `py -3.13 -m venv faster_whisper_stt/.venv`로 만들고, 아니면 `py -m venv faster_whisper_stt/.venv`를 사용합니다. Python 3.14의 `ctranslate2` 휠 제공 여부는 아직 확인되지 않았으므로 3.13을 권장합니다.

```powershell
py -3.13 -m venv faster_whisper_stt/.venv
faster_whisper_stt/.venv/Scripts/python.exe -m pip install --upgrade pip
faster_whisper_stt/.venv/Scripts/python.exe -m pip install -r faster_whisper_stt/requirements.txt
```

## 실행

```powershell
# 스피커 출력 루프백
faster_whisper_stt/.venv/Scripts/python.exe -m faster_whisper_stt.live_demo --model small

# 마이크 입력 (장치 번호는 기존 sherpa 데모에서 확인)
faster_whisper_stt/.venv/Scripts/python.exe -m faster_whisper_stt.live_demo --source microphone

# 더 빠른 갱신과 짧은 문맥, 선택 프로그램 오디오
faster_whisper_stt/.venv/Scripts/python.exe -m faster_whisper_stt.live_demo --window-seconds 6 --update-seconds 1 --target-process-id 1234
```

처음 실행하면 모델이 `faster_whisper_stt/models/`에 자동으로 내려받아집니다. 기본 `small`/`int8`은 CPU 우선 설정이며, 지원되는 NVIDIA CUDA 환경에서는 `--compute-type float16`을 선택할 수 있습니다. 큰 모델은 지연이 커질 수 있습니다. `--stable-passes 2`는 결과 안정화 표시 기준입니다. 실행 환경에 따라 오디오가 쌓여 지연이 증가할 수 있으므로 콘솔의 마지막 추정 지연과 결과 안정성을 sherpa의 개발자 계측값과 함께 비교하세요.

이 스크립트는 별도의 비교 데모이며 현재 프런트의 전사 선택 메뉴나 데이터 계약 이벤트에 연결하지 않습니다. 롤링 창에서 수정되는 문장을 기존 이벤트 계약으로 안전하게 확정하는 정책을 평가한 뒤 앱 연동을 진행해야 합니다.
