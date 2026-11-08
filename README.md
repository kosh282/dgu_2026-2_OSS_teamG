# 수업 참여도 분석 서비스

온라인 수업 화면을 선택해 영상 특징을 분석하고, 수업 타이머·메모·세션 기록을 함께 관리하는 팀 프로젝트입니다. 영상은 선택한 창에서 로컬로 캡처합니다.

## 현재 구현 상태

- 프런트는 Streamlit 기반이며 수업 제목 입력, 창 연결, 수업 시작·종료, 메모, 세션 JSON 내보내기 기능을 제공합니다.
- 영상 분석은 MediaPipe로 얼굴·상체 특징을 추출합니다. 개발자 모드에서는 캡처 화면, 랜드마크 오버레이, 후처리 특징과 실험 점수를 확인할 수 있습니다.
- 단일 얼굴용 규칙 기반 실험 점수는 통합부의 구간 이벤트로 전달되어 라이브 타임라인과 세션 JSON에 표시됩니다. 복수 얼굴의 점수 집계는 아직 구현되지 않았으며, 실험 점수는 최종 참여도 판단이 아닙니다.
- sherpa-onnx 한국어 스트리밍 전사는 프런트에서 선택적으로 사용할 수 있습니다. 출력 루프백 또는 마이크 하나를 연결하며, 실제 환경에서의 음질·지연 검증은 별도로 필요합니다. 강의 조교(`ai_agent/`)는 교수자가 올린 PDF·PPTX 자료와 확정 전사를 로컬에서 비교해 수업 중 자료를 제안하고, 사후 검토에서 구간별 피드백과 수업 시각에 맞춘 자료 제안을 만듭니다. 문장 작성에 Claude를 선택적으로 쓸 수 있으며 기본값은 꺼짐입니다. 실제 Claude 호출과 실제 수업 자료에서의 제안 정확도는 아직 검증하지 않았습니다. 자세한 내용은 [강의 조교 안내](ai_agent/README.md)를 참고하세요.
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
# 강의 조교(자료 읽기, 선택적 Claude)
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
