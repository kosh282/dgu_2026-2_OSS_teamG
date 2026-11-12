# 수업 참여도 분석 서비스

온라인 수업 화면을 선택해 영상 특징을 분석하고, 수업 타이머·메모·세션 기록을 함께 관리하는 팀 프로젝트입니다. 영상은 선택한 창에서 로컬로 캡처합니다.

## 현재 구현 상태

- 프런트는 Streamlit 기반이며 수업 제목 입력, 창 연결, 수업 시작·종료, 메모, 세션 JSON 내보내기 기능을 제공합니다.
- 영상 분석은 MediaPipe로 얼굴·상체 특징을 추출합니다. 개발자 모드에서는 캡처 화면, 랜드마크 오버레이, 후처리 특징과 실험 점수를 확인할 수 있습니다.
- 머리 방향 80%와 눈 관측 20%를 결합한 규칙 기반 실험 점수는 통합부의 구간 이벤트로 전달됩니다. 재식별을 켜면 ID별, 끄면 단기 위치 트랙별 점수를 익명으로 집계합니다. 실제 다인 화면의 정확도는 검증 전이며 실험 점수는 최종 참여도 판단이 아닙니다.
- sherpa-onnx 한국어 스트리밍 전사는 프런트에서 선택적으로 사용할 수 있습니다. 강의 조교(`ai_agent/`)는 낮은 실험 점수와 확정 전사를 바탕으로 실시간 제안을 만들며 PDF·PPTX 자료는 선택 사항입니다. 개인 API 키로 OpenAI·Gemini·Claude를 선택하면 자료 유무와 무관하게 AI를 활용하고, AI 검색어가 있으면 한국어 위키백과에서 제목과 링크를 검색합니다. 검색 결과의 내용과 정확성은 사용자가 확인해야 합니다. 실제 API 호출과 수업 자료에서의 정확도는 아직 검증하지 않았습니다. 자세한 내용은 [강의 조교 안내](ai_agent/README.md)를 참고하세요.
- 창 단위 캡처는 현재 Windows용입니다. macOS에서는 창 선택·캡처 기능을 사용할 수 없습니다.

## 설치

Windows에서 저장소 루트 폴더를 열고 PowerShell에서 실행합니다. Python 3.14 환경을 기준으로 의존성을 고정했습니다.

```powershell
git clone https://github.com/kosh282/dgu_2026-2_OSS_teamG.git
cd dgu_2026-2_OSS_teamG
python -m venv video_recognition/.venv
video_recognition/.venv/Scripts/python.exe -m pip install --upgrade pip
video_recognition/.venv/Scripts/python.exe -m pip install -r video_recognition/requirements-integrated.txt
video_recognition/.venv/Scripts/python.exe -m pip install -r frontend/requirements.txt
video_recognition/.venv/Scripts/python.exe -m pip install -r sherpa_stt/requirements.txt
video_recognition/.venv/Scripts/python.exe -m sherpa_stt.download_model
# 강의 조교 자료 읽기
video_recognition/.venv/Scripts/python.exe -m pip install -r ai_agent/requirements.txt
```

## 실행 및 사용

```powershell
video_recognition/.venv/Scripts/python.exe -m streamlit run frontend/app.py
```

브라우저에서 터미널에 표시된 주소(보통 `http://localhost:8501`)를 엽니다. 수업 제목과 동의를 입력하고 영상 분석, 실험 점수, 음성 전사 사용 여부를 선택한 뒤 세션을 시작합니다. 영상 분석을 사용하면 수업 창도 선택합니다. 개발자 모드에서 영상 오버레이와 분석값을 확인할 수 있고, 수업을 마치면 세션 JSON을 내려받을 수 있습니다.

## 데모

Windows에서 웹캠 영상 창을 캡처하는 분석 데모:

```powershell
video_recognition/.venv/Scripts/python.exe -m video_recognition.webcam_test
```

macOS에서는 웹캠 프레임을 직접 분석하는 별도 데모를 사용합니다. Mac용 가상환경과 영상 분석 의존성 설치 방법은 [영상 분석 안내](video_recognition/README.md#macos-웹캠-직접-입력)를 참고하세요.

```bash
video_recognition/.venv/bin/python -m video_recognition.webcam_test_mac
```

두 데모는 브라우저 대신 OpenCV 창에 영상·각도·실험 점수를 표시합니다. 창을 선택하고 `C`로 정면 보정을 시작하고 `Q`로 종료합니다. 원본 영상은 저장하지 않으며, 진단 로그는 `video_recognition/demo_logs/`에 생성됩니다. Mac 데모는 Windows 수업 창 캡처를 대신하지 않습니다.

자세한 사용법은 [프런트 안내](frontend/README.md), [음성 전사 안내](sherpa_stt/README.md), [영상 분석 안내](video_recognition/README.md), [점수 모델 안내](engagement_model/README.md), [강의 조교 안내](ai_agent/README.md)를 참고하세요.

## 개발·진단 테스트 명령어

아래 명령은 Windows PowerShell에서 **저장소 루트**를 기준으로 실행합니다. 위 설치 단계가 먼저 필요합니다. 옵션의 `--`는 긴 대시가 아닌 일반 하이픈 두 개입니다.

### 프런트 실시간 개발자 화면

```powershell
video_recognition/.venv/Scripts/python.exe -m streamlit run frontend/app.py
```

수업 창을 연결하고 영상 분석을 선택해 수업을 시작한 다음, 하단 **개발자 도구 → 개발자 모드 켜기**를 누릅니다. 웹캠 모델 데모와 같은 별도 창에서 실제 분석 화면, 실험 점수와 자세 입력 품질을 확인합니다. 창에 초점을 두고 `V`로 오버레이, `[`·`]`로 상세 페이지, `C`로 재보정, `Q`로 창 숨기기를 할 수 있습니다. 모델이 꺼졌거나 관측이 불가능하면 점수는 표시되지 않습니다. 다인 모드 점수는 전체 집계이며 개별 각도와 구분해야 합니다. 개발자 화면은 분석기와 같은 프레임을 사용하며 별도 카메라는 열지 않습니다.

### 웹캠 모델·각도 진단

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --single-model --debug
```

카메라를 정면으로 보고 `C` 또는 화면의 보정 버튼을 누릅니다. `V`는 오버레이, `D`는 상세 진단, `[`·`]`는 페이지 전환, `Q`는 종료입니다. 현재 방향 점수와 10초 집계는 실험값입니다.

좌우 비대칭을 조사하려면 아래 명령을 사용합니다. 정면을 보고 `C`를 누른 뒤 정면·왼쪽·정면·오른쪽·정면 순서로 화면의 이동 및 유지 안내를 따릅니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --yaw-diagnostic --yaw-log --single-model --diagnostic-move-seconds 3 --diagnostic-hold-seconds 5
```

초별 각도·기준점 변화 통계는 `engagement_model/diagnostic_logs/`에 저장됩니다. 영상과 기준점 원본은 저장하지 않지만 진단 수치는 민감할 수 있으므로 사용 후 직접 삭제하세요. 자세한 해석과 한계는 [웹캠 진단 안내](engagement_model/docs/webcam-diagnostics.md)를 참고하세요.

### 수업 창 캡처 확인

```powershell
video_recognition/.venv/Scripts/python.exe -m video_recognition.integrated_analysis --list
video_recognition/.venv/Scripts/python.exe -m video_recognition.integrated_analysis --hwnd 12345 --session class-test.json
```

먼저 수업 프로그램 창을 열고 첫 명령으로 표시된 번호를 확인한 뒤, `12345`를 실제 번호로 바꿉니다. 두 번째 명령은 별도의 분석을 실행하고 결과 JSON을 저장합니다. 자세한 안내는 [영상 분석 안내](video_recognition/README.md)를 참고하세요.

### 자동 테스트

```powershell
video_recognition/.venv/Scripts/python.exe -B -m unittest discover -s engagement_model/tests -v
video_recognition/.venv/Scripts/python.exe -B -m unittest discover -s frontend/tests -v
```

자동 테스트는 실제 카메라·수업 창·음성 장치에서의 동작 검증을 대신하지 않습니다.
