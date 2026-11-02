# 수업 창 영상 분석

`integrated_analysis.py`는 수업 창을 캡처해 `face_analysis.py`의 공통 MediaPipe 분석기를 호출하고, 특징을 모델 입력 콜백과 통합부 상태 이벤트로 연결합니다. 눈 개방 비율과 머리 각도는 실험적 특징이며 참여도 점수가 아닙니다. 얼굴 순서는 프레임 간 개인 식별자가 아닙니다.

## 코드 구조

```text
선택한 창 → window_capture.py → integrated_analysis.py
                                  ├─ face_analysis.py: 얼굴 특징·품질
                                  ├─ pose_analysis.py: 어깨 위치·품질
                                  ├─ head_calibration.py: 선택적 정면 보정
                                  ├─ model_consumer: 메모리 내 특징 전달
                                  └─ IntegrationService: 상태·세션 저장
preview.py: 분석 입력과 기준점 표시 / webcam_test.py: 웹캠 시험
```

창 캡처는 Windows Graphics Capture를 사용합니다. 얼굴과 상체 결과는 독립적으로 생성하므로 두 배열의 같은 순서가 같은 사람이라는 뜻은 아닙니다. 분석 상태 이벤트와 선택적 중간 파일 외에 프레임별 개인 특징은 최종 세션 JSON에 넣지 않습니다.

## 설치 및 모델

현재 컴퓨터에서는 Python 3.14 전용 `.venv`에 `requirements-integrated.txt`의 버전을 설치했습니다. 다른 환경에서는 해당 파일을 사용해 별도 가상환경에 설치하세요. 기존 `requirements.txt`는 Python 3.10 기반 팀원 EyeGestures 실험용으로 유지합니다.

```powershell
python -m venv video_recognition/.venv
video_recognition/.venv/Scripts/python.exe -m pip install -r video_recognition/requirements-integrated.txt
```

MediaPipe 공식 Face Landmarker 모델은 `models/face_landmarker.task`에 준비했습니다. 모델 파일을 커밋하고 원격에 올리면 저장소를 내려받을 때 함께 제공됩니다. 파일 출처는 [MediaPipe 공식 모델](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)입니다. 다른 `.task` 모델도 Git에 추가할 수 있으며, `.venv`는 계속 제외됩니다.

## 수업 창 실행

Windows에서 수업 브라우저 또는 프로그램 창을 먼저 열고, **저장소 루트**에서 실행합니다. Windows Graphics Capture를 통해 선택한 창을 대상으로 하며 창이 최소화되거나 닫히면 분석을 종료하고 상태를 알립니다. 이 컴퓨터에는 필요한 가상환경과 모델이 준비되어 있습니다.

1. PowerShell의 작업 폴더를 저장소 루트로 옮긴 뒤 창 목록을 확인합니다.

```powershell
cd 'C:\Users\user\Documents\공개SW구현\dgu_2026-2_OSS_teamG'
video_recognition/.venv/Scripts/python.exe -m video_recognition.integrated_analysis --list
```

2. 출력된 목록에서 수업 창 제목 앞의 숫자를 확인합니다. 아래 `12345`는 **예시이므로 실제 창 번호로 반드시 바꿔야 합니다.** 창을 닫고 다시 열면 번호도 바뀔 수 있습니다.

```powershell
video_recognition/.venv/Scripts/python.exe -m video_recognition.integrated_analysis --hwnd 12345 --session class-test.json
```

분석을 시작하면 캡처된 화면과 MediaPipe 기준점을 보여주는 별도 미리보기 창이 열립니다. 초록 점은 얼굴, 주황 점과 선은 상체 기준점입니다. 창 이동이나 크기 변경 후에도 선택한 창을 계속 대상으로 합니다. 미리보기 창에 초점을 두고 `V`를 누르면 원본과 기준점 표시를 전환하며, `Q` 또는 창 닫기로 분석을 종료합니다. 터미널에서 `Ctrl+C`를 눌러도 종료됩니다.

터미널에는 약 2초마다 특징값이 표시됩니다. 이를 끄려면 `--quiet`를, 미리보기 창을 열지 않으려면 `--no-preview`를 추가하세요. 종료 시 세션이 저장되며, 저장 경로에 같은 이름의 파일이 있으면 **덮어쓰므로** 사용하지 않은 파일명을 지정하세요.

`--intermediate vision-intermediate.jsonl`은 선택 사항이며, 지정하면 프레임별 특징이 JSONL로 저장됩니다. 이 중간 파일은 최종 세션 JSON에 포함되지 않지만 자동 삭제되지 않으므로 보관과 삭제를 직접 관리하세요. 현행 공유 계약은 개인별 중간 특징을 로그에 저장하지 않는다고 명시하므로 이 선택적 실험 저장에 대해서는 팀 합의가 필요합니다.

선택한 창 핸들로 Windows Graphics Capture 프레임을 받으므로 화면 좌표를 잘라내지 않습니다. 정지 화면에서 새 프레임이 없으면 이전 프레임을 새 분석 결과로 반복하지 않고 기다립니다. 앱의 보호 영상·최소화·여러 모니터에서의 동작과 작은 참가자 타일의 인식 정확도는 실제 환경에서 검증해야 합니다. 모델 입력은 콜백 자리만 있으며 참여 모델은 연결되지 않았습니다. Mac은 현재 창 선택과 캡처를 지원하지 않습니다.

## 웹캠으로 직접 시험하기

저장소 루트에서 실행하면 웹캠 영상 창, 실제 캡처 화면 창, 분석 결과 창이 열립니다. 실제 캡처 화면은 분석기에 전달된 바로 그 프레임이므로 잘림이나 다른 창에 가려진 부분을 확인할 수 있습니다.

```powershell
video_recognition/.venv/Scripts/python.exe -m video_recognition.webcam_test
```

웹캠 영상 창을 열고 카메라를 바라보세요. 종료하려면 창을 선택한 뒤 `Q`를 누르거나 터미널에서 `Ctrl+C`를 누릅니다. 다른 카메라를 쓰려면 `--camera 1`, 10초 뒤 자동 종료하려면 `--duration 10`을 추가합니다.

캡처 미리보기에서 초록 점은 얼굴 기준점, 주황 점과 선은 가시성이 확보된 상체 기준점입니다. `V` 키로 기준점 표시를 끄거나 켜 원본 캡처와 비교할 수 있습니다. 표시된 점은 모델의 추정 결과이며 정확한 인식을 보장하지 않습니다.

결과 창에는 얼굴 수와 MediaPipe가 반환한 얼굴별 기준점 개수, 코와 턱의 x·y·z 좌표가 표시됩니다. x·y는 영상 크기에 대한 정규화 좌표이고 z는 MediaPipe의 상대 깊이 값입니다. 눈 개방 비율과 머리 각도는 이 기준점에서 별도로 계산한 실험값입니다. 얼굴 순서는 프레임마다 바뀔 수 있습니다. 이 시험은 웹캠 영상·기준점·분석 결과를 파일로 저장하지 않습니다.

## 모델 입력 특징과 제한

`FeatureWindow`의 `features`에는 얼굴 방향(머리 pitch/yaw/roll 각도), 눈 개방 비율, 코의 정규화 위치, 표정 형태를 나타내는 일부 MediaPipe blendshape 점수와 상체 어깨 중심 위치가 담깁니다. blendshape는 감정이나 참여도를 나타내는 점수가 아닙니다. 상체 분석에는 `models/pose_landmarker_lite.task`가 필요하며 [MediaPipe 공식 모델](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task)을 사용합니다.

머리와 상체 이동 후보는 각각 해당 프레임에서 검출이 한 개이고 직전 프레임과 1초 이내일 때, 화면에서의 정규화 좌표 거리와 시간 간격으로 표시합니다. 이 방식은 동일인 추적이 아니며 카메라 이동·화면 배치 변경·검출 대상 교체에도 값이 변할 수 있습니다. 첫 프레임, 미검출, 다중 검출 또는 시간 간격이 너무 큰 경우는 `null`입니다. 얼굴 목록과 상체 목록은 서로 같은 사람 순서가 아닙니다. 다중 참가자의 개인별 이동에는 별도의 검출·연결 방식과 검증이 필요합니다.

특징은 `VisionPipeline`의 `model_consumer` 콜백이 지정되면 메모리로 전달됩니다. 실행 예제에는 실제 모델을 연결하지 않았으며, 세션 JSON에도 개인별 특징은 저장하지 않습니다. 선택적 중간 파일을 요청한 경우에만 별도로 저장합니다.

팀원이 같은 프로세스에서 연결할 때는 기존 `IntegrationService`와 세션 ID를 공유하고, 모델이 준비된 뒤 콜백을 전달합니다.

```python
from video_recognition.integrated_analysis import VisionPipeline, WindowCapture
from video_recognition.face_analysis import FaceAnalyzer
from video_recognition.pose_analysis import PoseAnalyzer

def model_consumer(feature_window):
    # 모델 담당자가 품질 확인·시간 구간 집계·추론을 구현할 자리입니다.
    pass

# service, session_id, hwnd는 실행 중인 앱에서 전달합니다.
pipeline = VisionPipeline(service, session_id, WindowCapture(hwnd), FaceAnalyzer(),
                          model_consumer=model_consumer, pose_analyzer=PoseAnalyzer())
try:
    # 실제 앱에서는 새 프레임이 없으면 FrameNotAvailable을 처리하며 반복 호출합니다.
    pipeline.process_one()
finally:
    pipeline.close()  # 세션 종료 전에 호출합니다.
```

이 예제는 연결 위치를 보여주는 코드입니다. 화면이 정지해 아직 새 프레임이 없을 때는 `video_recognition.window_capture.FrameNotAvailable`이 발생하며, 이를 분석 실패가 아닌 대기로 처리해야 합니다. 실제 앱에서는 `stop_session()`을 호출하기 전에 캡처·분석을 종료합니다.

### 보정 없이 사용하는 기본 모델 입력

기본 `FeatureWindow`는 개인 정면 보정을 요구하지 않습니다. `features.faces`에는 `head_angles_degrees` 원본 각도, `head_orientation_periodic` 각 축의 `sin`·`cos`, 눈 개방 비율, 표정 형태 점수, 얼굴별 `quality`가 포함됩니다. sin·cos는 ±180도 숫자 경계를 연속적으로 표현하기 위한 값이며, 개인별 정면 보정이나 각도 추정 정확성 보정이 아닙니다. `upper_bodies`에는 어깨 중심과 상체별 `quality`가 포함됩니다. 얼굴과 상체의 배열 순서가 같은 사람을 뜻하지 않습니다.

필요한 얼굴 기준점이나 양쪽 어깨 기준점이 화면 밖이면 관련 측정은 `null`과 품질 사유로 전달합니다. 어깨 가시성이 0.5 미만인 경우도 위치를 만들지 않습니다. 화면 안에 있고 `quality`가 `valid`여도 실제 인식이 정확하거나 가림이 없다는 보장은 없습니다. 최상위 `quality`는 얼굴이나 상체가 일부 부족하면 `partial`, 둘 다 없으면 `not_observable`입니다. 모델이 이 품질 상태를 반영해야 하며, 관측 불가를 0으로 바꾸면 안 됩니다.

`model_consumer`는 모든 프레임의 특징·품질을 받는 연결 지점일 뿐 실제 참여 모델은 아직 연결되지 않았습니다. 프레임 내 개인별 특징은 메모리에서 전달되며 `save_session()` 결과에 포함되지 않습니다. 모델 담당자와 특징 이름·단위·집계 방식에 합의하기 전의 실험 형식(`experimental-0.5`)입니다. 개인별 정면 보정은 아래 선택 기능으로 유지하며, 보정하지 않아도 원본 특징과 품질이 콜백으로 전달됩니다.

## 단일 얼굴 정면 기준 보정 실험

웹캠 테스트 또는 수업 창 미리보기를 실행하고 **얼굴이 한 명만 보이는 상태에서 정면을 바라본 뒤 `C` 키**를 누릅니다. 기준점 화면에 `collecting`이 표시되면 약 3초간 머리를 고정합니다. 10개 이상의 안정적인 프레임이 모이면 `ready`가 되고 `head_angles_relative_degrees`가 정면 기준 상대 각도로 출력됩니다. 웹캠 결과 창의 `raw pitch/yaw/roll`은 원래 각도, `calibrated pitch/yaw/roll`은 정면 기준으로 보정한 세 축의 각도입니다. 수업 창 터미널 출력에서도 확인할 수 있습니다.

수집 중 얼굴이 가려지거나 여러 명이 보이면 수집을 다시 시작합니다. 기준을 잡은 뒤 얼굴이 사라지거나 여러 얼굴이 감지되면 기준이 무효화되고 `waiting`으로 돌아갑니다. 다시 보정하려면 한 명이 보이는 상태에서 `C`를 누르세요. 얼굴이 바뀌었는지 확인하는 추적 기능은 없으므로, 검출이 끊기지 않은 채 다른 사람으로 바뀌면 잘못된 기준이 적용될 수 있습니다. 이 기능은 단일 인물 실험용이며 다중 참가자별 보정이 아닙니다.

상대 `pitch`는 확인한 환경에서 위쪽이 음수, 아래쪽이 양수입니다. `yaw`의 좌우 부호는 영상의 좌우 반전과 관찰 기준에 따라 달라질 수 있어 최종 규칙을 확정하지 않았습니다. 각도는 실제 측정의 정확성을 보장하지 않으며, 정면 응시나 참여도를 판정하지 않습니다. 보정 전·실패 시 상대 각도는 `null`입니다. 화면 없이 단일 인물을 시험할 때만 `--calibrate`를 지정해 시작 즉시 보정할 수 있습니다.

## 기존 EyeGestures 실험

팀원 원본 `integrated_analysis.py`는 `eye_gestures_demo.py`로 보존했습니다. 기존 실험은 단일 웹캠과 시선 보정을 사용하므로 여러 참가자에 대한 시선 추정으로 간주하지 않습니다. 공통 경로는 원본의 MediaPipe 얼굴 검출을 사용하고, 눈 비율과 머리 각도는 팀원의 기존 `face_analysis.py`에서 공통 분석기로 옮겼습니다. EyeGestures의 시선·고정·도약 결과는 공통 경로에 연결하지 않았습니다.
