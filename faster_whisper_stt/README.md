# faster-whisper 실시간 비교

이 폴더는 sherpa-onnx 전사와 비교하기 위한 독립 실험입니다. 프로젝트의 기존 STT 경로를 바꾸지 않습니다. faster-whisper는 스트리밍 인식기가 아니므로 최근 오디오 창(기본 8초)을 반복 인식합니다. `live_demo`는 한 번의 추론을 마친 뒤 기본 2초를 기다리므로 실제 갱신 간격은 2초에 추론 시간을 더한 값입니다. 새 입력이 없으면 동일 창을 재분석하지 않습니다. `인식 중`은 겹치는 창의 원본 가설이라 중복될 수 있습니다. 단어와 시각이 연속해서 일치하면 중복을 줄여 `정리된 전사 추가`로 표시하고 종료할 때 `정리된 전체 전사`를 보여 줍니다. 마지막에 아직 안정화되지 않은 내용은 `미확정 끝부분`으로 별도 표시합니다.

이 후처리는 실험용 휴리스틱입니다. 단어 경계나 인식 내용이 바뀌면 누락·중복·잘못된 확정이 가능하고, 한번 정리된 내용은 다시 수정하지 않습니다. 단어 타임스탬프 산출로 부하도 증가할 수 있습니다. 출력은 콘솔에만 있으며 전사 이벤트나 세션 JSON을 만들지 않습니다. 오디오 창은 메모리에서만 유지합니다.

## 별도 설치

프런트의 선택형 전사는 `runner.py`의 `WhisperLiveTranscription`을 사용합니다. 콘솔 비교 데모와 달리 겹치지 않는 기본 10초 구간을 공통 계약 이벤트로 전달합니다. `WhisperConfig`는 16kHz 입력과 양의 정수 구간 길이·청크 길이·스레드 수를 초기화 때 검사합니다. 0초 구간의 반복 처리나 잘못된 표본률을 모델 실행 전에 거부합니다. [프런트 안내](../frontend/README.md)를 참고하세요.

저장소 루트에서 실행합니다. 별도 Python 3.13 설치가 있으면 `py -3.13 -m venv faster_whisper_stt/.venv`로 만들고, 아니면 `py -m venv faster_whisper_stt/.venv`를 사용합니다. Python 3.14의 `ctranslate2` 휠 제공 여부는 아직 확인되지 않았으므로 3.13을 권장합니다.

```powershell
py -3.13 -m venv faster_whisper_stt/.venv
faster_whisper_stt/.venv/Scripts/python.exe -m pip install --upgrade pip
faster_whisper_stt/.venv/Scripts/python.exe -m pip install -r faster_whisper_stt/requirements.txt
```

## 실행

### 중첩 없는 전사 테스트

`faster_whisper_stt/.venv/Scripts/python.exe -m faster_whisper_stt.direct_demo --source loopback`은 기본 2초 음성 구간을 중첩 없이 한 번씩 인식하여 바로 콘솔에 출력합니다. `--source microphone`은 마이크, `--chunk-seconds 3`은 구간 길이를 변경합니다. VAD, 단어 정렬, 중첩 병합을 사용하지 않으므로 무음 오인식이나 구간 경계에서 단어가 잘릴 수 있고, 모델 자체 처리와 최소 모노 변환은 남습니다. 추론이 밀리면 대기열 길이를 표시하며 최대치에 도달하면 중단합니다. 종료 시 입력 큐에 남은 오디오는 저장하지 않고, 전사 대기열과 마지막으로 받은 짧은 구간은 처리합니다. 프런트·세션 기록에는 연결되지 않습니다.

### sherpa와 동일 입력 비교

두 모델을 한 프로세스에서 실행하므로 비교 환경에 sherpa를 추가로 설치합니다.

```powershell
faster_whisper_stt/.venv/Scripts/python.exe -m pip install sherpa-onnx==2.0.1
faster_whisper_stt/.venv/Scripts/python.exe -m faster_whisper_stt.compare_demo --source loopback --model base
```

기본 출력 장치에서 한 번 캡처하고 모노로 변환한 동일한 16kHz float32 블록을 두 모델에 전달합니다. 마이크는 `--source microphone`, 비교 시간 제한은 예를 들어 `--duration 60`을 사용합니다. 설치된 sherpa 모델 샘플로 장치 없이 확인하려면 `--wav sherpa_stt/models/sherpa-onnx-streaming-zipformer-korean-2024-06-16/test_wavs/0.wav`를 지정합니다. WAV는 16kHz 모노 16비트 PCM이어야 합니다. 종료는 Ctrl+C입니다.

`[sherpa 인식 중]`과 `[sherpa 확정]`, `[whisper]` 원본 가설·정리 전사를 구분해 비교합니다. Whisper는 8초 롤링 창을 재분석하므로 결과가 겹치고 원본 가설은 바뀔 수 있습니다. 종료 시 `[whisper 마지막 창]`은 남은 마지막 창의 별도 추론이며 이전 결과와 겹칩니다. `[공통 입력]`은 입력 레벨과 큐·누락·장치 불연속 및 sherpa 처리 RTF를 표시하고, `[whisper] [처리]`는 창별 추론 시간을 보여 줍니다. 큐 블록 누락이 확인되면 비교를 중단합니다. 장치 불연속이 있다면 두 모델에 전달되기 전부터 오디오가 손실됐을 수 있습니다.

Whisper는 모델 내부 VAD를 사용하므로 모델에 공급한 샘플은 같아도 인식기 내부 처리는 다릅니다. Whisper의 정리 전사는 실험용이고, 두 모델의 최종 발화 확정 방식과 시각 표시도 다릅니다. 시스템 부하 역시 두 모델 동시 실행 조건이며 단독 실행과 다릅니다. 오디오와 전사 텍스트는 파일로 저장하지 않습니다.

```powershell
# 스피커 출력 루프백
faster_whisper_stt/.venv/Scripts/python.exe -m faster_whisper_stt.live_demo --model small

# 마이크 입력 (장치 번호는 기존 sherpa 데모에서 확인)
faster_whisper_stt/.venv/Scripts/python.exe -m faster_whisper_stt.live_demo --source microphone

# 더 빠른 갱신과 짧은 문맥, 선택 프로그램 오디오
faster_whisper_stt/.venv/Scripts/python.exe -m faster_whisper_stt.live_demo --window-seconds 6 --update-seconds 1 --target-process-id 1234
```

처음 실행하면 모델이 `faster_whisper_stt/models/`에 자동으로 내려받아집니다. 기본 `small`/`int8`은 CPU 우선 설정이며, 지원되는 NVIDIA CUDA 환경에서는 `--compute-type float16`을 선택할 수 있습니다. 큰 모델은 지연이 커질 수 있습니다. `--stable-passes 2`는 단어를 정리된 전사에 추가하기 위한 연속 일치 횟수입니다. 실행 환경에 따라 오디오가 쌓여 지연이 증가할 수 있으므로 콘솔의 마지막 추정 지연과 결과 안정성을 sherpa의 개발자 계측값과 함께 비교하세요.

이 스크립트는 별도의 비교 데모이며 현재 프런트의 전사 선택 메뉴나 데이터 계약 이벤트에 연결하지 않습니다. 롤링 창에서 수정되는 문장을 기존 이벤트 계약으로 안전하게 확정하는 정책을 평가한 뒤 앱 연동을 진행해야 합니다.
