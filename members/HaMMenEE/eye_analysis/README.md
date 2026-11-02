# 수업 창 영상 분석

`integrated_analysis.py`는 수업 창을 캡처해 `face_analysis.py`의 공통 MediaPipe 분석기를 호출하고, 특징을 모델 입력 콜백과 통합부 상태 이벤트로 연결합니다. 눈 개방 비율과 머리 각도는 실험적 특징이며 참여도 점수가 아닙니다. 얼굴 순서는 프레임 간 개인 식별자가 아닙니다.

## 설치 및 모델

현재 컴퓨터에서는 Python 3.14 전용 `.venv`에 `requirements-integrated.txt`의 버전을 설치했습니다. 다른 환경에서는 해당 파일을 사용해 별도 가상환경에 설치하세요. 기존 `requirements.txt`는 Python 3.10 기반 팀원 EyeGestures 실험용으로 유지합니다.

```powershell
python -m venv members/HaMMenEE/eye_analysis/.venv
members/HaMMenEE/eye_analysis/.venv/Scripts/python.exe -m pip install -r members/HaMMenEE/eye_analysis/requirements-integrated.txt
```

MediaPipe 공식 Face Landmarker 모델은 `models/face_landmarker.task`에 준비했습니다. 모델 파일을 커밋하고 원격에 올리면 저장소를 내려받을 때 함께 제공됩니다. 파일 출처는 [MediaPipe 공식 모델](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)입니다. 다른 `.task` 모델도 Git에 추가할 수 있으며, `.venv`는 계속 제외됩니다.

## 수업 창 실행

저장소 루트에서 실행합니다. 첫 명령으로 핸들을 확인하고 두 번째 명령에 선택한 숫자를 넣습니다.

```powershell
members/HaMMenEE/eye_analysis/.venv/Scripts/python.exe -m members.HaMMenEE.eye_analysis.integrated_analysis --list
members/HaMMenEE/eye_analysis/.venv/Scripts/python.exe -m members.HaMMenEE.eye_analysis.integrated_analysis --hwnd 12345 --intermediate vision-intermediate.jsonl --session session.json
```

`Ctrl+C`로 종료합니다. 중간 결과 경로는 선택 사항이며 지정하면 JSONL로 저장됩니다. 최종 세션 JSON에는 이 특징이 포함되지 않지만 중간 파일은 자동 삭제되지 않으므로 보관과 삭제를 직접 관리하세요. 현행 공유 계약은 개인별 중간 특징을 로그에 저장하지 않는다고 명시하므로 이 선택적 실험 저장에 대해서는 팀 합의가 필요합니다.

창의 보이는 화면 영역을 캡처하므로 가려진 내용도 함께 캡처되며 최소화된 창은 처리할 수 없습니다. 배율·다중 모니터·작은 참가자 타일에서의 동작과 다중 얼굴 정확도는 실제 수업 환경에서 검증해야 합니다. 모델 입력은 콜백 자리만 있으며 참여 모델은 연결되지 않았습니다.

## 기존 EyeGestures 실험

팀원 원본 `integrated_analysis.py`는 `eye_gestures_demo.py`로 보존했습니다. 기존 실험은 단일 웹캠과 시선 보정을 사용하므로 여러 참가자에 대한 시선 추정으로 간주하지 않습니다. 공통 경로는 원본의 MediaPipe 얼굴 검출을 사용하고, 눈 비율과 머리 각도는 팀원의 기존 `face_analysis.py`에서 공통 분석기로 옮겼습니다. EyeGestures의 시선·고정·도약 결과는 공통 경로에 연결하지 않았습니다.
