# faster_whisper_stt — 한국어 음성 입력과 전사

수업 음성(시스템 출력 루프백 또는 마이크)을 메모리 큐로 받아 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)로 한국어 전사하고, 확정 전사를 `transcript_final` 이벤트로 [통합 서비스](../integration/service.py)에 전달합니다. 프런트의 유일한 전사 경로입니다. 청크 시각은 영상 분석과 같은 세션 공통 시계(`service.elapsed_ms(session_id)`)를 씁니다. 원본 음성은 파일로 저장하지 않습니다.

이전에는 sherpa-onnx 스트리밍 모델(`sherpa_stt/`)도 선택할 수 있었지만, 강의 녹화 음성(데모 영상)에서 1분당 0~9자만 출력해 사실상 인식하지 못해 제거했습니다(아래 측정 기록). 그 결과 부분 전사(`transcript_partial`, '인식 중' 표시)는 만들지 않고, 문장은 말 멈춤 단위로 늦게 확정됩니다.

## 구성

| 파일 | 역할 |
|---|---|
| `runner.py` | `WhisperConfig`, `WhisperLiveTranscription`(프런트 실행기), `segment_text`(지어낸 문장 거르기) |
| `segmenter.py` | `PauseSegmenter`: 말 멈춤 지점에서 구간을 자름 |
| `audio.py` | `AudioChunk`(계약 5절, 메모리 전용), `ChunkClock`(세션 시계 시각), `to_mono_float32` |
| `loopback.py` | SoundCard 기본 출력 장치 루프백, 선택 프로세스 루프백(`ProcessLoopbackRecorder`) |
| `microphone.py` | sounddevice 마이크 입력, `list_input_devices()` |
| `native/` | 프로세스 트리 WASAPI 루프백 도우미 소스와 빌드 스크립트(실험 경로) |
| `direct_demo.py`, `live_demo.py` | 콘솔 확인용 데모. 세션 이벤트를 만들지 않음 |

## 설치

프런트와 같은 환경에 설치합니다. 저장소 루트에서 실행합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -m pip install -r faster_whisper_stt/requirements.txt
```

선택한 모델(`small` 기본, `base`)은 첫 사용 시 `faster_whisper_stt/models/`에 자동으로 내려받습니다(small 약 480MB). 모델 폴더는 Git에 올리지 않습니다. Python 3.13 가상환경(Windows 11)에서 faster-whisper 1.2.1 설치·실행을 확인했습니다. Python 3.14의 `ctranslate2` 휠 제공 여부는 확인하지 않았습니다.

### 이전 sherpa 정리

sherpa를 쓰던 PC에는 pull 후에도 Git이 관리하지 않던 모델 폴더(`sherpa_stt/models`, 약 430MB)와 패키지가 남습니다. 실행에는 영향이 없고 `sherpa_stt/`는 루트 `.gitignore`로 커밋에서 제외되지만, 정리하려면 저장소 루트에서 다음을 실행합니다.

```powershell
if (Test-Path sherpa_stt) { Remove-Item -Recurse -Force sherpa_stt }
video_recognition/.venv/Scripts/python.exe -m pip uninstall -y sherpa-onnx sherpa-onnx-core
```

## 프런트 실행기 동작

`WhisperLiveTranscription`은 겹치지 않는 구간을 한 번씩 인식해 확정 전사로 전달합니다(값은 실험값).

- **말 멈춤 분할**(`PauseSegmenter`): 8초(`min_segment_seconds`) 이상 모인 뒤 말이 0.5초 멈추면 자르고, 계속 말하면 최대 20초(`window_seconds`) 직전 3초 중 가장 조용한 지점에서 자릅니다. Whisper는 구간 길이와 관계없이 30초 입력으로 계산하므로 짧게 자를수록 계산량이 늘고 문맥이 줄어듭니다. 대신 문장이 화면에 나타나기까지 구간 길이만큼(평균 약 11초) 더 걸립니다. 20ms 에너지를 잡음 바닥과 비교한 추정이라 큰 배경음악은 말소리로 볼 수 있습니다. 말소리 프레임이 없는 구간은 모델에 보내지 않습니다.
- **음성 구간 감지**: faster-whisper 내장 Silero VAD(`vad_filter=True`)를 켭니다.
- **불안정한 결과 거르기**(`segment_text`): 구간 전체가 반복으로 채워진 결과(압축률 2.4 초과), 무음 확률 0.6 초과이면서 평균 로그확률 -1 미만인 결과, 용어 힌트 목록을 그대로 옮겨 적은 결과를 버립니다. "감사합니다"처럼 실제로 말할 수 있는 문장을 내용만으로 삭제하지 않습니다. 실제로 같은 말을 반복했을 수 있으므로 남긴 결과의 텍스트는 고치지 않고 인식한 그대로 씁니다("OnClick, OnClick, OnClick" 같은 반복 출력도 그대로 남을 수 있습니다).
- **디코딩 제한**: 온도 0만 쓰고(재시도 없음) 출력은 구간 1초당 6토큰+10으로 제한합니다. 아래 측정 기록의 47초 구간 문제를 막기 위한 값입니다.
- **앞 문장 힌트**: 직전 확정 문장의 끝 120자를 `initial_prompt`로 넣어 문맥과 용어를 잇습니다. 실행기가 말 멈춤 구간마다 `transcribe()`를 새로 호출하므로 한 호출 내부의 세그먼트만 잇는 `condition_on_previous_text`는 끕니다. 잘못 인식한 내용이 다음 구간에 영향을 줄 수 있습니다.
- **강의 자료 용어 힌트**: 프런트는 준비 화면에 올린 PDF/PPTX에서 여러 쪽에 반복되는 용어(`ai_agent.lecture_terms`, 최대 20개·150자)를 골라 `WhisperLiveTranscription(hint_terms=...)`로 넘기고, 실행기는 이를 faster-whisper `hotwords`로 넣습니다. 자료가 없으면 힌트 없이 동작합니다. 자료에 없는 말을 비슷한 용어로 바꿔 적을 수 있고 처리 시간이 늘어납니다.
- 기본 `beam_size=1`, CPU 스레드 2개, CPU `int8` 추론입니다.
- 청크 번호가 빠지면(캡처 큐가 가득 차 버림) 쌓인 구간을 먼저 인식하고 `analysis_status`(사유 `audio_unavailable`)로 알립니다. 시작 실패는 `unavailable` 상태로 알리고 `False`를 반환합니다.

`WhisperConfig`는 16kHz 입력, 양의 정수 구간 길이·최소 구간·멈춤 길이·빔 크기·청크 길이·스레드 수, 최소 구간 ≤ 최대 구간, 온도·반복 금지 값을 초기화 때 검사합니다. `stats()`는 처리 RTF, 구간 끝→인식 완료 지연, 최근 입력 레벨, 처리·빈 구간 수, 버린 청크·장치 불연속을 메모리에서만 제공합니다. [프런트 안내](../frontend/README.md)를 참고하세요.

### 설정 비교 기록 (2026-11-15, 참고용)

데모 영상(강의 녹화, 고음이 약한 음성) 30~270초를 0.1초 청크로 흘려 넣어 비교했습니다. 정답 자막이 없어 `large-v3-turbo`(빔 5, 전체 문맥, VAD) 전사를 **임시 정답**으로 썼으므로 아래 CER은 실제 오류율이 아닌 설정 간 상대 비교입니다. 영어 용어의 한글 표기 차이("onclick"/"온 클릭")도 오류로 셉니다. 단일 4분 구간·단일 장치(4코어 노트북) 결과이며 다른 프로그램 부하에 따라 처리 시간(RTF)이 0.5~1.3까지 달라졌습니다.

같은 구간에서 제거 전 sherpa-onnx 스트리밍 모델(`sherpa-onnx-streaming-zipformer-korean-2024-06-16`)은 볼륨 정규화·고정밀 모델·고음 보강을 해도 1분당 0~9자만 출력했습니다. 모델 동봉 정답 문장(깨끗한 낭독)은 오류율 1% 수준으로 인식했으므로 강의 녹화 음성(압축·고음 감쇠, 빠른 말, 전문용어)에서의 한계로 보았습니다.

| 설정 (small, int8) | 상대 CER | RTF |
|---|---:|---:|
| 기존: 10초 고정, VAD 없음 | 31.5% | 0.37 |
| 10초 고정 + VAD + 거르기 | 34.0% | 0.48 |
| 말 멈춤 분할 + VAD + 거르기 | 30.2% | 0.53 |
| 위 + 빔 5 | 28.3% | 0.67 |
| 위(빔 1) + 앞 문장 힌트 | 27.7~28.0% | 0.52 |
| 위 + 빔 5 + 앞 문장 힌트 | 31.9% | (부하 중 측정) |
| 말 멈춤 분할 + VAD + 거르기, 스레드 2 | 30.2% | 0.57 |
| base, 말 멈춤 분할 + VAD + 거르기 | 42.3% | 0.27 |

구간 길이 비교(모두 VAD + 거르기 + 앞 문장 힌트, 스레드 2, 구간 길이 외 같은 조건):

| 구간 길이 | 상대 CER | RTF | 평균 구간 |
|---|---:|---:|---:|
| 3~10초 | 28.8% | 0.65 | 6.6초 |
| **8~20초 (기본값)** | **22.6%** | **0.34** | 11.4초 |
| 12~28초 | 22.9% | 0.30 | 15.4초 |
| base, 8~20초 | 42.5% | 0.18 | 11.4초 |

3~10초 설정으로 대시보드 세션(영상 분석 켬, 재식별 끔)에서 데모 영상을 3분 재생했을 때 영상 분석은 약 2.4Hz로 4명 점수를 유지했지만 Whisper RTF가 1.0~1.09로 올라 전사 지연이 37초까지 늘었습니다. 이 결과로 기본 구간을 8~20초로 바꿨습니다. 같은 조건으로 다시 3분 재생했을 때 Whisper RTF는 0.2~0.57, 구간 끝부터 확정 전사까지의 지연은 약 7~8초로 늘지 않았고 대기열 누락은 없었습니다. 영상 분석은 2.2~4.3Hz로 4명 점수를 유지했고 세션 종료는 6.1초 걸렸습니다. 다른 장치, 재식별을 켠 경우, 수업 전체 길이(10분 이상)에서는 확인하지 않았습니다.

용어 힌트 비교(8~20초 기본 설정). 데모 강의의 실제 자료가 없어 슬라이드에 있을 법한 용어 20개(안드로이드, 액티비티, 추상 클래스, 인터페이스, onClick, OnClickListener 등)를 직접 골라 넣었으므로 실제 자료보다 유리할 수 있습니다. 핵심 용어 수는 임시 정답의 등장 횟수와 비교했고 영문/한글 표기를 같은 용어로 셌습니다. 임시 정답 자체도 "추상"을 "초상"으로 적는 등 틀린 곳이 있어 상대 CER은 힌트 쪽에 불리합니다.

| 설정 | 상대 CER | RTF | 핵심 용어 맞힘(55회) | 용어 오인식("인터뷰"·"주연" 등) |
|---|---:|---:|---:|---:|
| 힌트 없음 | 22.6% | 0.35 | 39 | 10 |
| hotwords 힌트 | 25.8% | 0.55 | 52 | 1 |
| 프롬프트 앞에 용어 | 26.8% | 0.54 | 53 | 1 |

hotwords를 넣은 대시보드 실행에서 Whisper가 한 구간을 약 60초 동안 처리하지 못해 대기열이 차고 0.9초 분량을 버렸습니다. 같은 음성으로 재현하니 "OnClick, OnClick, OnClick" 같은 반복 출력 때문에 faster-whisper 기본 온도 재시도(최대 6회)가 매번 최대 길이까지 다시 디코딩해 8초 구간에 42~47초가 걸렸습니다. 그래서 기본값을 온도 0만 사용(재시도 없음), 출력 길이를 구간 1초당 6토큰+10으로 제한, 힌트 목록을 그대로 옮겨 적은 출력은 버리기로 바꿨습니다. 아래 표는 반복된 단어 묶음을 한 번으로 줄이는 후처리를 함께 쓴 측정입니다. 이후 실제로 반복한 말도 줄어들 수 있어 이 후처리는 뺐습니다(텍스트는 인식한 그대로 사용). 반복 n-gram 금지(`no_repeat_ngram_size=3`)도 시험했지만 정상적으로 다시 나온 말을 바꿔 정확도가 떨어져 기본은 끕니다. 바꾼 뒤 같은 4분 구간 결과:

| 설정 | 상대 CER | RTF | 핵심 용어 맞힘(55회) | 용어 오인식 | 가장 오래 걸린 구간 |
|---|---:|---:|---:|---:|---:|
| 힌트 없음 | 23.5% | 0.56 | 38 | 10 | 13.3초 |
| **용어 힌트 (기본 경로)** | 26.3% | 0.50 | **50** | **1** | **8.1초** |

이 측정은 다른 프로그램 부하가 있는 상태라 RTF가 앞선 표보다 높습니다. 영상 분석과 함께 실행할 때 RTF가 1을 넘으면 전사가 점점 늦어지므로, 데모 중에는 다른 무거운 프로그램을 닫거나 base를 선택하세요.

## 콘솔 데모

### 중첩 없는 전사 테스트

`video_recognition/.venv/Scripts/python.exe -m faster_whisper_stt.direct_demo --source loopback`은 기본 2초 음성 구간을 중첩 없이 한 번씩 인식하여 바로 콘솔에 출력합니다. `--source microphone`은 마이크, `--chunk-seconds 3`은 구간 길이를 변경합니다. VAD, 단어 정렬, 중첩 병합을 사용하지 않으므로 무음 오인식이나 구간 경계에서 단어가 잘릴 수 있습니다. 추론이 밀리면 대기열 길이를 표시하며 최대치에 도달하면 중단합니다. 종료 시 입력 큐에 남은 오디오는 저장하지 않고, 전사 대기열과 마지막으로 받은 짧은 구간은 처리합니다. 프런트·세션 기록에는 연결되지 않습니다.

### 롤링 창 데모

`live_demo`는 최근 오디오 창(기본 8초)을 반복 인식하는 콘솔 확인용입니다. 한 번의 추론을 마친 뒤 기본 2초를 기다리므로 실제 갱신 간격은 2초에 추론 시간을 더한 값입니다. `인식 중`은 겹치는 창의 원본 가설이라 중복될 수 있습니다. 단어와 시각이 연속해서 일치하면 중복을 줄여 `정리된 전사 추가`로 표시하고 종료할 때 `정리된 전체 전사`를 보여 줍니다. 아직 안정화되지 않은 끝부분은 `미확정 끝부분`으로 표시합니다. 이 후처리는 실험용 휴리스틱이라 누락·중복·잘못된 확정이 가능합니다. 출력은 콘솔에만 있으며 전사 이벤트나 세션 JSON을 만들지 않습니다.

```powershell
# 스피커 출력 루프백
video_recognition/.venv/Scripts/python.exe -m faster_whisper_stt.live_demo --model small

# 마이크 입력. 장치 번호 확인:
video_recognition/.venv/Scripts/python.exe -c "from faster_whisper_stt.microphone import list_input_devices; print(list_input_devices())"
video_recognition/.venv/Scripts/python.exe -m faster_whisper_stt.live_demo --source microphone --device 1

# 더 빠른 갱신과 짧은 문맥, 선택 프로그램 오디오
video_recognition/.venv/Scripts/python.exe -m faster_whisper_stt.live_demo --window-seconds 6 --update-seconds 1 --target-process-id 1234
```

기본 `small`/`int8`은 CPU 우선 설정이며, 지원되는 NVIDIA CUDA 환경에서는 `--compute-type float16`을 선택할 수 있습니다. `--stable-passes 2`는 단어를 정리된 전사에 추가하기 위한 연속 일치 횟수입니다.

### 선택 프로그램 오디오(실험)

`--target-process-id`는 선택한 프로세스 트리의 소리만 받는 `native/process_loopback.exe`가 필요합니다. Visual Studio 2022 C++ 빌드 도구가 있는 Windows에서 `faster_whisper_stt/native/build_process_loopback.ps1`로 빌드합니다. 프런트는 이 경로를 쓰지 않고 기본 출력 장치 전체 소리를 받습니다.

## 테스트

```powershell
video_recognition/.venv/Scripts/python.exe -m unittest discover -s faster_whisper_stt -p "test_*.py" -v
```

모델·오디오 장치 없이 가짜 모델과 가짜 캡처로 분할, 거르기, 디코딩 옵션, 용어 힌트, 시각 계산, 캡처 큐 누락, 시작·처리 실패를 확인합니다.
