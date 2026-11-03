# 교수자 라이브 세션 도우미

온라인 수업 중 교수자가 수업 창을 연결해 얼굴 방향과 상체 움직임 같은 영상 특징을 확인하고, 수업 메모와 세션 기록을 남길 수 있는 프로토타입입니다. 영상은 선택한 창에서 로컬로 처리합니다.

현재는 창 캡처와 MediaPipe 특징 추출, 수업 타이머, 메모, 세션 JSON 저장 및 개발자용 프리뷰를 사용할 수 있습니다. 참여도 점수 모델, 음성 전사(STT), 강의 자료 검색·자동 제안은 아직 연결되지 않았으므로 화면의 검출 후보 수를 실제 참가자 수나 참여도 점수로 해석하면 안 됩니다.

## 다운로드 및 설치

현재 실시간 창 캡처 기능은 Windows에서 동작합니다. Python 3.14 환경에서 확인했습니다.

1. GitHub 저장소를 내려받고 프로젝트 폴더로 이동합니다.

```powershell
git clone https://github.com/kosh282/dgu_2026-2_OSS_teamG.git
cd dgu_2026-2_OSS_teamG
```

2. 가상환경을 만들고 영상 분석 및 프런트엔드 패키지를 설치합니다.

```powershell
python -m venv video_recognition/.venv
video_recognition/.venv/Scripts/python.exe -m pip install --upgrade pip
video_recognition/.venv/Scripts/python.exe -m pip install -r video_recognition/requirements-integrated.txt
video_recognition/.venv/Scripts/python.exe -m pip install -r frontend/requirements.txt
```

## 실행

저장소 루트 폴더에서 아래 명령을 실행합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -m streamlit run frontend/app.py
```

브라우저에서 터미널에 표시된 로컬 주소(보통 `http://localhost:8501`)를 엽니다.

## 수업 사용 방법

1. 수업 브라우저나 프로그램을 열어 둡니다.
2. 대시보드의 **수업 제목**을 입력하고 분석할 창을 선택한 뒤 **창 연결**을 누릅니다. 이 단계에서는 분석과 타이머가 아직 시작되지 않습니다.
3. 라이브 세션의 **수업 시작**을 누릅니다. 선택한 창의 영상 분석과 수업 타이머가 시작됩니다.
4. 필요한 경우 **개발자 도구**에서 개발자 모드를 켜 캡처 프레임과 랜드마크를 별도 창으로 확인합니다.
5. 수업이 끝나면 **수업 종료**를 누르고 **공통 계약 세션 JSON 저장**을 눌러 기록을 다운로드합니다.

수업 제목은 저장된 JSON의 `session.title`에 기록됩니다. 수업 중 작성한 메모는 같은 세션의 이벤트로 저장됩니다.

## 개발자 프리뷰 키

- `V`: 얼굴 윤곽·홍채 메시와 상체 오버레이 켜기/끄기
- `M`: 촘촘한 얼굴 삼각 메시 켜기/끄기
- `[` / `]`: 개발자 텍스트 창에서 얼굴 데이터 페이지 이동
- `C`: 정면 응시 기준을 약 3초 동안 임시 수집
- `Q`: 개발자 프리뷰 창 닫기. 수업 분석은 계속됩니다.

추가 정보는 [프런트엔드 안내](frontend/README.md)와 [영상 분석 안내](video_recognition/README.md)를 참고하세요.
