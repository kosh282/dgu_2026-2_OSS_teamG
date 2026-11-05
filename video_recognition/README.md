# 수업 창 영상 분석

`integrated_analysis.py`는 수업 창을 캡처해 `face_analysis.py`의 공통 MediaPipe 분석기를 호출하고, 특징을 모델 입력 콜백과 통합부 상태 이벤트로 연결합니다. 웹캠 데모와 개발자 미리보기에는 실험 행동 점수를 표시하고, 라이브 세션은 시간 구간 집계를 이벤트로 저장합니다. 검증된 참여도 점수는 아니며 얼굴 순서는 개인 식별자가 아닙니다.

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

선택한 창 핸들로 Windows Graphics Capture 프레임을 받으므로 화면 좌표를 잘라내지 않습니다. 정지 화면에서 새 프레임이 없으면 이전 프레임을 새 분석 결과로 반복하지 않고 기다립니다. 앱의 보호 영상·최소화·여러 모니터에서의 동작과 작은 참가자 타일의 인식 정확도는 실제 환경에서 검증해야 합니다. 웹캠 데모는 시점별 점수를 표시하고 라이브 세션은 별도 구간 집계를 통합부에 전달합니다. Mac은 현재 창 선택과 캡처를 지원하지 않습니다.

## 웹캠으로 직접 시험하기

저장소 루트에서 실행하면 웹캠 영상 창, 실제 캡처 화면 창, 분석 결과 창이 열립니다. 실제 캡처 화면은 분석기에 전달된 바로 그 프레임이므로 잘림이나 다른 창에 가려진 부분을 확인할 수 있습니다.

```powershell
video_recognition/.venv/Scripts/python.exe -m video_recognition.webcam_test
```

웹캠 영상 창을 열고 카메라를 바라보세요. 종료하려면 창을 선택한 뒤 `Q`를 누르거나 터미널에서 `Ctrl+C`를 누릅니다. 다른 카메라를 쓰려면 `--camera 1`, 10초 뒤 자동 종료하려면 `--duration 10`을 추가합니다. 기본으로 `video_recognition/demo_logs/` 아래에 실행별 JSONL 진단 로그가 생성됩니다. 위치를 바꾸려면 `--log-dir C:\\temp\\vision-logs`처럼 지정합니다. 로그에는 영상 원본을 넣지 않고 캡처 시각, 상대/원본 각도, 모델 판정과 점수 상태를 저장합니다.

캡처 미리보기에는 개발자 모드와 같은 얼굴 윤곽·홍채 메시, 상체 선, 후처리 표시가 겹쳐집니다. `V` 키로 오버레이를 끄거나 켜고, `M` 키로 촘촘한 얼굴 삼각망을 전환합니다. `[`와 `]`로 얼굴 정보 페이지를 이동하고 `C`로 정면 기준 수집을 시작합니다. 표시된 점과 선은 모델의 추정 결과이며 정확한 인식을 보장하지 않습니다.

결과 창은 프런트엔드 개발자 모드와 `video_recognition/developer_view.py`의 같은 렌더러를 사용합니다. 얼굴·상체 검출 수와 품질, 분석 주기/해상도, 각도와 눈·표정·움직임 특징 및 실험 점수를 표시합니다. 정면 보정 전이나 관측이 부족하면 점수는 보류됩니다. 실험 점수는 검증된 참여도가 아닙니다. 진단 JSONL에는 각도와 모델 결과가 기록됩니다.

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

`model_consumer`에는 단일 얼굴용 규칙 기반 실험 모델이 연결되어 웹캠 데모와 개발자 미리보기에서 점수·보류 이유를 보여줍니다. 라이브 세션은 별도 집계기로 구간 이벤트를 발행하며 웹캠 데모는 발행하지 않습니다. 프레임 내 개인별 특징은 세션 JSON에 포함되지 않습니다. 영상 특징 자체는 정면 보정 없이도 전달되지만 현재 실험 점수에는 정면 보정이 필요합니다.

모델 점수가 일시 보류되면 공통 개발자 패널과 웹캠 데모는 마지막 유효 점수를 최대 2초 동안 `HELD` 표시와 함께 보여주고 경과 시간을 표시합니다. 현재 점수 필드 자체는 계속 `null`이며, 2초가 지나면 이전 표시값도 만료됩니다. 이 표시 전용 캐시는 모델 점수 계산·세션 이벤트·타임라인·저장 데이터에 사용하지 않습니다.

## 단일 얼굴 정면 기준 보정 실험

웹캠 테스트 또는 수업 창 미리보기를 실행하고 **얼굴이 한 명만 보이는 상태에서 정면을 바라본 뒤 `C` 키**를 누릅니다. 결과 패널에는 상태, 샘플 수(10개 이상 필요), 경과 시간(3초 필요), 진행률과 마지막 입력 문제 코드가 표시됩니다. `collecting` 중 얼굴 미검출, 복수 얼굴, 품질 불가, 머리 각도 결측, 긴 프레임 공백, 움직임이 감지되면 사유가 나타납니다. 보정이 완료되면 `ready`가 되고 `head_angles_relative_degrees`가 프레임 간 스무딩 후 정면 기준 상대 각도로 출력됩니다. 원본 기반 비교 각도는 `head_angles_relative_raw_degrees`에 별도로 유지합니다. 모델은 스무딩 상대각을 사용합니다. 현재 모델의 시험용 상하 임계값은 정면 pitch ±10도, 아래 보기 +15도(해제 +10도), 위쪽 이탈 −20도입니다. 안정 점수가 있으면 아래 보기를 5초간 유지하고, 이후 10초 동안 최대 20점을 완만하게 감점합니다. 결과 화면과 JSONL 실행 로그에 입력 각도·판정·적용 정책을 함께 기록합니다. 각도 기준과 감쇠 정책은 실제 성능 검증 전 비교 실험값입니다.

모델 입력의 yaw는 영상 특징 원본을 수정하지 않고 점수 어댑터에서만 중립 보정합니다. 기본값은 ±2도 데드밴드, 2~15도 구간 0.75배 완화이며, 15도 이상은 큰 회전을 유지하도록 구간 오프셋만 적용합니다. 개발자 패널과 JSONL 로그는 보정 전 상대 yaw와 모델 입력 yaw를 나란히 보여줍니다. 설정은 `engagement_model/config.py`에서 조정할 수 있습니다.

수집 중 얼굴이 가려지거나 여러 명이 보이면 샘플이 지워지고 마지막 문제 코드가 표시됩니다. 기준을 잡은 뒤 얼굴이 사라지거나 여러 얼굴이 감지되면 기준이 무효화되고 `waiting`으로 돌아갑니다. 다시 보정하려면 한 명이 보이는 상태에서 `C`를 누르세요. 보정은 사용자가 정면을 바라보는지 자동 판별하지 않고, 3초간 자세가 안정적인지만 확인합니다. 얼굴이 바뀌었는지 확인하는 추적 기능은 없으므로 검출이 끊기지 않은 채 다른 사람으로 바뀌면 잘못된 기준이 적용될 수 있습니다. 이 기능은 단일 인물 실험용이며 다중 참가자별 보정이 아닙니다.

상대 `pitch`는 확인한 환경에서 위쪽이 음수, 아래쪽이 양수입니다. `yaw`의 좌우 부호는 영상의 좌우 반전과 관찰 기준에 따라 달라질 수 있어 최종 규칙을 확정하지 않았습니다. 각도는 실제 측정의 정확성을 보장하지 않으며, 정면 응시나 참여도를 판정하지 않습니다. 보정 전·실패 시 상대 각도는 `null`입니다. 화면 없이 단일 인물을 시험할 때만 `--calibrate`를 지정해 시작 즉시 보정할 수 있습니다.

## 기존 EyeGestures 실험

팀원 원본 `integrated_analysis.py`는 `eye_gestures_demo.py`로 보존했습니다. 기존 실험은 단일 웹캠과 시선 보정을 사용하므로 여러 참가자에 대한 시선 추정으로 간주하지 않습니다. 공통 경로는 원본의 MediaPipe 얼굴 검출을 사용하고, 눈 비율과 머리 각도는 팀원의 기존 `face_analysis.py`에서 공통 분석기로 옮겼습니다. EyeGestures의 시선·고정·도약 결과는 공통 경로에 연결하지 않았습니다.
## 얼굴 특징 후처리 확장 지점

현재 기본 인식기는 MediaPipe Face Landmarker 결과에서 얼굴 특징을 만들고, `face_postprocessing.py`가 프레임 간 신호를 정리합니다. 후처리기는 감지 모델과 분리되어 있어 추후 모델이 바뀌어도 같은 얼굴 특징 사전과 중심 좌표를 제공하면 재사용할 수 있습니다. `VisionPipeline` 생성자에 다른 `face_postprocessor`를 주입할 수 있고, 기본 구현의 설정은 `PostprocessingConfig`에서 조정합니다.

현재 적용하는 실험 신호:

- `head_angles_degrees`: 원본 pitch/yaw/roll을 유지합니다.
- `head_angles_smoothed_degrees`: 위치 기반 임시 얼굴 상태별 1-Euro 필터 결과입니다. 각도 경계(-180/180도)를 고려해 필터링합니다.
- `head_angle_change_candidate`: 연속 관측 간 스무딩 각도 변화량과 시간 간격입니다. 수업 태도 판정은 아닙니다.
- `eye_aspect_ratio_left/right`: 양쪽 눈 EAR 원본값입니다.
- `blink_metrics`: 눈 감김 후보, 누적 횟수, 최근 60초 횟수입니다. 기본 EAR 기준 0.21 및 2프레임 연속 감김은 초기 참고값으로, 해상도와 촬영 조건에 맞춰 검증해야 합니다.
- `transient_track_key`: 위치만 이용해 프레임 사이 필터 상태를 이어주는 실행 중 임시 키입니다. 개인 식별 ID가 아니며 최종 세션과 중간 JSONL 저장본에서는 제거합니다.

개발자 모드에서는 캡처 영상 위에 임시 키, 스무딩 각도, 깜빡임 횟수를 겹쳐 표시하고 텍스트 창에는 원본/스무딩 각도와 EAR·변화량을 함께 보여줍니다. 개인 보정 각도와 원본 값도 기존처럼 확인할 수 있습니다.

이 기능은 MediaPipe를 중복 실행하지 않고, [ATemova/face-mesh](https://github.com/ATemova/face-mesh)의 MIT 라이선스 알고리즘 아이디어 중 1-Euro 스무딩, EAR 깜빡임, 중심점 임시 추적만 이 프로젝트 출력 형식에 맞게 적용합니다. 저작권 및 라이선스 표기는 저장소의 `THIRD_PARTY_NOTICES.md`를 참고하세요. PERCLOS, 졸음/하품 판정, 시선 추정 및 참여도 점수는 적용하지 않았습니다. 현재 신호는 촬영 각도·화면 타일 크기·가림·프레임 누락의 영향을 받으므로 참여도나 집중도에 대한 검증된 지표로 해석하면 안 됩니다.

후속 재사용 시에는 (1) 새 감지기가 원본 각도, 좌우 EAR, 정규화 얼굴 중심을 내도록 어댑터를 작성하고, (2) 기존 입력 사전에 맞추거나 별도 후처리기를 `VisionPipeline`에 주입하고, (3) 출력 필드 의미나 단위가 바뀌면 `feature_version`을 올리고, (4) 여러 얼굴이 교차하거나 검출 순서가 바뀌는 영상에서 임시 추적과 신호 분리가 유지되는지 확인하세요. 위치 기반 추적은 얼굴이 교차하면 상태가 바뀔 수 있으며, 장시간 신원 재식별을 제공하지 않습니다.

## 분석 주기와 모델 입력 크기

기본값은 얼굴 분석 초당 10회, 상체 분석 초당 5회, 모델 입력 긴 변 최대 1920픽셀입니다. 회의 화면에서 여러 참가자가 작게 표시되는 상황을 고려해 입력 크기를 높였으며, 처리 부하와 작은 얼굴 검출률을 확인한 뒤 이 값을 조정할 수 있습니다. 창 캡처와 개발자 영상 프리뷰에는 원본 해상도를 유지하고, MediaPipe 두 분석기에 전달하는 사본만 종횡비를 보존해 축소합니다. 프레임을 큐에 쌓지 않고 새로 도착한 화면 중 분석 시점의 최신 프레임을 사용합니다. 실제 측정 주기와 원본/분석 해상도는 `features.sampling` 및 개발자 텍스트 창에서 확인할 수 있습니다. 상체 결과를 다음 측정 시점까지 재사용하며 오래된 값은 품질 사유에 반영합니다.

CLI에서는 `--face-hz`, `--pose-hz`, `--max-analysis-dimension`으로 조정할 수 있습니다. 예를 들어 `--face-hz 10 --pose-hz 5 --max-analysis-dimension 1920`이며, 최대 크기를 `0`으로 주면 축소하지 않습니다. 얼굴 분석을 낮추면 짧은 깜빡임을 놓칠 수 있으므로 현재 깜빡임 값은 보조 실험 신호로 해석해야 합니다.

## FaceMesh 저장소 기능과 기존 파이프라인 중복 정리

참고 저장소 [ATemova/face-mesh](https://github.com/ATemova/face-mesh)는 별도 얼굴 검출기라기보다 MediaPipe Tasks Face Landmarker에 후처리·시각화·세션 도구를 더한 Python 패키지입니다. 이 프로젝트는 이미 MediaPipe로 얼굴을 검출하므로 FaceMesh 패키지의 검출기를 함께 실행하지 않습니다.

| FaceMesh 기능 | 현재 파이프라인의 대응 | 적용 방식 |
| --- | --- | --- |
| 얼굴 랜드마크와 blendshape | `FaceAnalyzer`의 기존 MediaPipe 결과 | 감지·특징 계산은 기존 결과만 사용 |
| 좌우 EAR, 눈 열림, 깜빡임 카운터 | 좌우 EAR와 `BlinkMetrics` | EAR는 한 번 계산하고 눈 열림 평균을 깜빡임 판정에도 재사용 |
| 머리 pitch/yaw/roll | 기존 `solvePnP` 각도와 1-Euro 스무딩 | 기존 축·정면 보정 호환을 위해 각도 추정은 유지하고, 별도 추정기를 병렬 실행하지 않음 |
| 전체 메시 랜드마크 1-Euro 스무딩 | `LandmarkOverlaySmoother` | 개발자 화면 표시 복사본에만 적용. 분석용 EAR·각도·blendshape는 원본 랜드마크 사용 |
| 홍채 시선 및 화면 좌표 보정 | 현재 미적용 | 수업 화면 타일에서 신뢰할 수 있는 시선 기준을 별도로 검증해야 함 |
| PERCLOS·마이크로수면·하품(MAR) | 현재 미적용 | 졸림·하품을 수업 참여도와 동일시하지 않으며, 기준도 검증되지 않음 |
| 다중 얼굴 위치 추적 | 임시 위치 기반 키 | 스무딩/깜빡임 상태 연결용이며 신원 식별이나 재식별 ID가 아님 |
| 윤곽·홍채·삼각망 렌더링 | 개발자 모드의 FaceMesh 오버레이 | 윤곽/눈/눈썹/입술/홍채는 기본 표시, 촘촘한 삼각망은 `M` 키로 선택 |
| 녹화·JSONL/CSV 내보내기·재생·타임라인 | 기존 세션 계약과 선택적 중간 JSONL | 저장 계약을 유지하고 임시 추적 키와 랜드마크는 최종 세션에 넣지 않음 |

개발자 영상 창에서 `V`는 얼굴/상체 선 표시를 전환하고, `M`은 촘촘한 삼각망을 켜고 끕니다. 텍스트 창은 한 페이지에 두 얼굴의 전체 후처리 결과를 표시하며 `[`와 `]`로 검출된 얼굴 페이지를 넘깁니다. 각 얼굴에 임시 키·추적 방식·후처리 버전, 원본/보정/스무딩 각도, 각도 변화량과 간격, 좌우 EAR·눈 열림, 양쪽 눈 감김 상태, 좌우/전체 깜빡임 누계, 최근 창 개수·길이·분당 비율·이벤트 여부를 표시합니다.

연결선 정의는 MediaPipe Tasks의 `FaceLandmarksConnections`에서 읽고, 픽셀 선은 이 프로젝트의 OpenCV 프리뷰에서 그립니다. 478개 정규화 랜드마크를 다시 추론하지 않으며 FaceMesh 패키지도 추가 설치하지 않습니다. 표시용 랜드마크 스무딩은 분석 결과나 저장값을 바꾸지 않습니다. 촘촘한 삼각망은 얼굴이 많이 검출될수록 프리뷰 프레임률을 낮출 수 있습니다. API는 [MediaPipe FaceLandmarksConnections](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarksConnections), 참고 라이브러리는 [ATemova/face-mesh](https://github.com/ATemova/face-mesh#readme)에서 확인할 수 있습니다. 참고 알고리즘 저작권과 MIT 조건은 저장소의 `THIRD_PARTY_NOTICES.md`에 기록했습니다.
