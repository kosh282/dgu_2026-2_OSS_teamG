# 수업 참여도 분석 서비스

온라인 수업 화면을 선택해 영상 특징을 분석하고, 수업 타이머·메모·세션 기록을 함께 관리하는 팀 프로젝트입니다. 영상은 선택한 창에서 로컬로 캡처합니다.

## 현재 구현 상태

- 프런트는 Streamlit 기반이며 수업 제목 입력, 창 연결, 수업 시작·종료, 메모, 세션 JSON 내보내기 기능을 제공합니다.
- 영상 분석은 MediaPipe로 얼굴·상체 특징을 추출합니다. 개발자 모드에서는 캡처 화면, 랜드마크 오버레이, 후처리 특징과 실험 점수를 확인할 수 있습니다.
- 규칙 기반 실험 점수기는 영상 분석 파이프라인의 메모리 콜백에 연결되어 있습니다. 현재 한 얼굴만 점수화하며, 복수 얼굴 점수 집계와 공통 세션의 점수 구간 이벤트 및 프런트 점수 타임라인 연동은 아직 구현되지 않았습니다. 따라서 현재 점수는 최종 참여도 판단으로 해석하면 안 됩니다.
- sherpa-onnx 한국어 스트리밍 전사는 프런트에서 선택적으로 사용할 수 있습니다. 출력 루프백 또는 마이크 하나를 연결하며, 실제 환경에서의 음질·지연 검증은 별도로 필요합니다. 강의 자료 검색·추천과 수업 후 피드백 자동 생성은 아직 연결되지 않았습니다.
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
```

## 실행 및 사용

```powershell
video_recognition/.venv/Scripts/python.exe -m streamlit run frontend/app.py
```

브라우저에서 터미널에 표시된 주소(보통 `http://localhost:8501`)를 엽니다. 수업 제목을 입력하고 수업 창을 연결한 뒤 **수업 시작**을 누르면 분석과 타이머가 시작됩니다. 개발자 모드에서 영상 오버레이와 분석값을 확인할 수 있고, 수업을 마치면 세션 JSON을 내려받을 수 있습니다.

## 데모

웹캠 또는 캡처 창 분석 데모:

```powershell
video_recognition/.venv/Scripts/python.exe -m video_recognition.webcam_test
```

자세한 사용법은 [프런트 안내](frontend/README.md), [음성 전사 안내](sherpa_stt/README.md), [영상 분석 안내](video_recognition/README.md), [점수 모델 안내](engagement_model/README.md)를 참고하세요.
