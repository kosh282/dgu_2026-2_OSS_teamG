# 참여도 모델

## 1. 프로젝트 개요 및 목적

온라인 수업 영상에서 얻은 머리 방향과 관측 시간을 이용해 0~100의 실험 점수를 계산합니다.
점수는 교수자의 수업 성찰을 위한 참고 지표이며 실제 집중도·이해도·성적의 측정값이 아닙니다.
관측 불가와 판단 보류는 0점 대신 `null`과 사유로 표현합니다.

- 독립 웹캠과 프런트의 기본 모델은 **Gaussian softened**입니다.
- `EngagementRuntime()` 자체의 기본 모델은 Legacy로 유지합니다. 프런트는 `GaussianRuntime`을 명시적으로 생성합니다.
- 독립 데모에서 모델을 비교할 수 있으며, 프런트는 Gaussian을 사용합니다. 재식별을 함께 켜면 다인 Gaussian 경로로 실행합니다.
- 개별 계산기는 단일 얼굴 입력을 사용하며, 프런트 다인 경로는 확정된 재식별 ID마다 계산기를 따로 둡니다.
- 현재 Gaussian 점수와 최근 10초 집계 점수는 다른 값입니다.
- 원본 영상·얼굴 이미지·영구 식별자를 이 모듈의 저장 항목으로 추가하지 않습니다.

상세 문서:

- [모델별 수식·원식과 변형·버전](docs/model-formulas.md)
- [실험 조건·버전별 결과·검증 범위](docs/experiments.md)
- [웹캠 조작·자동 좌우 진단·결과 해석](docs/webcam-diagnostics.md)
- [MediaPipe 통합 범위·검증·남은 제약](docs/integration-review.md)

### MediaPipe 얼굴 변환행렬 자세 추정

프런트와 독립 웹캠은 MediaPipe 얼굴 변환행렬을 기본으로 사용합니다. 유효하지 않은 행렬은 관측 불가로 처리합니다. Gaussian 점수와 정면 보정은 유지되며, 각도 정확도와 수업 화면 다인 성능은 별도 검증이 필요합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --single-model --debug
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --yaw-diagnostic --diagnostic-down --front-stability --single-model --yaw-log
```

이전 `--pose-source`, `--validated-pose`, `--compare-face-transform` 옵션은 제거했습니다. `--gaze`는 별도 시선 모델 선택 기능입니다. 실험용 회전행렬 보정(`--calibration-method matrix`)은 기본 Euler 보정과 별개입니다.

## 프런트 다인 Gaussian

`MultiFaceGaussianRuntime`은 재식별 임시 트랙→ID 연결을 메모리 인자로 별도 전달받습니다.
원본 FeatureWindow와 단일 얼굴 어댑터는 변경하지 않습니다. ID별 사본에 HeadCalibration과
GaussianRuntime을 적용해 보정 기준·시간 상태·점수를 분리합니다. ID가 확정되기 전에는 점수화하지 않습니다.
각 ID는 처음 안정된 얼굴 각도를 약 3초 수집하며 정면인지 자동 확인하지는 못합니다.
다인 경로의 보정 중에도 다른 사람의 유효한 집계 이력은 유지합니다.

ID 이탈·재식별 불확실 구간은 현재/최근 점수 모두 null입니다. 같은 ID로 재연결되면 완료된
보정 기준을 유지하되 시간 상태는 초기화합니다. 미완료 보정은 연속 관측을 다시 수집합니다.
현재 관측자 재보정, 최대32명/마지막 확인 후120초 수명, 수업 종료 시 삭제를 적용합니다.
이름·ID·임베딩·개인별 점수 이력은 저장하지 않습니다.

그룹 provenance는 `multi-face-gaussian` / `multi-gaussian-0.1`입니다.
`WindowPublisher`는 다인 런타임의 `aggregate_window()`를 사용합니다. 각 유효 관측의
완화된 Pitch/Yaw로 Gaussian을 한 번 계산하고, 최근성 선형 커널의 시간 적분으로 가중합니다.
분자는 모든 유효한 사람·구간의 `시간가중치 × Gaussian 점수` 합,
분모는 같은 사람·구간의 시간가중치 합입니다. 개인의 이동 평균을 다시 평균하지 않습니다.
보류·미검출·보정 중인 사람은 0점을 넣지 않습니다. 이 값은 관측 가능한 사람들에 대한 참고 집계이며
전체 수강생 평균·출석률·집중도는 아닙니다.

구간은 최근10초/출력2초이며 기존 최소 관측·점수화 시간 기준을 적용합니다.
비율은 해당 구간에 적어도 한 사람의 유효 관측/점수가 있는 **벽시계 시간**을 분모로 계산합니다.
`model_extensions.multi-face-gaussian.scoreable_person_duration_ms`는 별도의
점수화 가능한 사람·밀리초 합이므로 구간 길이보다 클 수 있습니다.
예를 들어 두 명을 각각10초 유효 관측하면20000입니다. 이 값은 최근성 가중치 합 자체는 아닙니다.
개별 원시 각도는 집계 메모리에만 있고 저장 이벤트에는 없습니다. 공통 참가자 수 분모는 null입니다.

프런트에서 재식별을 켜야 다인 경로를 사용합니다. 재식별을 끄면 단일 얼굴 Gaussian으로 실행합니다.
MobileGaze 시선 실험은 포함하지 않습니다. [프런트 실행 안내](../frontend/README.md)를 참고하세요.
- [정리 전 README 원문·과거 실험·변경 이력](docs/implementation-history.md)

## 2. 전체 데이터 처리 흐름

```text
video_recognition: 얼굴/상체 특징 → 스무딩 → 정면 보정 → FeatureWindow
  ↓ model_consumer 콜백
input_adapter.py: 형식·단일 얼굴·보정·품질 검사, 상대각/Yaw 완화
  ↓
runtime.py: 상태·관측 시간 관리, Legacy 또는 Huang 계산
  또는 gaussian.py: 같은 상태 관리를 재사용해 Gaussian 계산
  ↓ latest: 실행 메모리에 보관
선택적으로 WindowPublisher 연결 → engagement_window → IntegrationService
```

`latest` 갱신과 JSON 이벤트 발행은 별도 단계입니다.
독립 `webcam_demo.py`는 런타임을 직접 연결하며 `WindowPublisher`를 연결하지 않습니다.
따라서 독립 웹캠 점수가 세션 파일에 저장되거나 프런트로 전달되는 것은 아닙니다.
현재 `frontend/live_session.py`는 단일/다인 Gaussian 런타임과 publisher를 연결합니다.

| 파일 | 책임 |
|---|---|
| [config.py](config.py) | `ModelConfig`: 실험 설정과 값 검증 |
| [input_adapter.py](input_adapter.py) | `HeadAngles`, `Observation`, `parse_feature_window()`, `soften_angle()` |
| [scoring.py](scoring.py) | 공통 상태 분류, Legacy 규칙, Huang 순수 함수·구간 계산 |
| [runtime.py](runtime.py) | `EngagementRuntime`, 세 모델을 같은 입력으로 실행하는 `ComparisonRuntime` |
| [gaussian.py](gaussian.py) | `GaussianRuntime`, 방향 점수·시간 가중 집계 |
| [window_publisher.py](window_publisher.py) | 구간 집계와 공통 계약 이벤트 발행 |
| [webcam_demo.py](webcam_demo.py) | 기존 영상 기능을 재사용하는 독립 웹캠·메모리 진단 |
| [tests/](tests/) | 입력·산정식·시간 상태·복귀·계약·실행기 모의 검사 |

## 3. Legacy / Huang / Gaussian

| 항목 | Legacy | Huang 온라인 | Gaussian 기본 |
|---|---|---|---|
| 생성 | `EngagementRuntime()` | `EngagementRuntime(ModelConfig.huang())` | `GaussianRuntime()` |
| 점수 | 상태·각도 강도·유지시간 규칙 | Pitch/Yaw 지표의 정규화 거리 결합 | 상대 Pitch/Yaw의 Gaussian 각도 공간 변형 |
| 정면 | 90점에서 5초간 100점으로 증가 | 정확한 정지 0도는 온라인 중립 처리로 약 64.6점 | 중립 입력 100점, 방향 편차에 따라 연속 감소 |
| ambiguous | 보류 | 유효 관측이면 계산 | 유효 관측이면 계산 |
| away | 초기 3초는 이전 안정 점수, 이후 각도·시간 감점 | 별도 Legacy 유예 없음 | 현재 방향 점수에 유예 없음 |
| 아래 보기 | 이전 안정 점수로 5초 유예 후 10초간 최대 20점 감점 | 보류 | 하향 전이부터 보류 |
| 시간 구간 | 기본 10초 | 큰 구간 30초·작은 구간 3초 | 최근 10초 시간 가중 |

각 모델이 점수화하는 구간과 점수 의미는 같지 않습니다.
Legacy의 높은 표시값이 현재 관측의 유효한 새 점수인지, 계산 유예나 화면 캐시인지 구분해야 합니다.
실제 얼굴·각도 입력의 좌우 비대칭은 모델의 수학적 대칭성과 별개의 검증 대상입니다.

선택 예시:

```python
from engagement_model.config import ModelConfig
from engagement_model.runtime import EngagementRuntime
from engagement_model.gaussian import GaussianRuntime

legacy = EngagementRuntime()
huang = EngagementRuntime(ModelConfig.huang())
gaussian = GaussianRuntime()  # softened, angle-time-0.3
reference = GaussianRuntime(ModelConfig.gaussian(gaussian_angle_mode='relative'))
```

## 4. 현재 Gaussian 산정식과 설정값

논문의 화면 투영 좌표 대신 프로젝트의 정면 보정 상대각을 사용하는 **변형식**입니다.
P/Y는 `head_angles_relative_degrees`의 Pitch/Yaw(도)입니다.
기본 softened 모드는 상태 판정용 원본 상대각을 보존하고 점수 입력만 완화합니다.

```text
h(a; d,k,b) = 0                                              if |a| <= d
             sign(a)*[k*(min(|a|,b)-d)+max(0,|a|-b)]          otherwise
P_eff = h(P; 2, 0.75, 10)
Y_eff = h(Y; 2, 0.75, 15)
S_current = 100*exp(-0.5*((P_eff/15)^2+(Y_eff/20)^2))
```

Yaw 완화는 Legacy와 같은 `soften_angle()`을 재사용합니다.
Pitch 완화와 sigma 15/20도는 프로젝트 실험 설정이며 논문의 최적값이 아닙니다.
큰 각도의 기울기는 1을 유지합니다. Roll은 점수에 사용하지 않습니다.
relative 비교 모드는 완화 전 P/Y를 그대로 사용합니다.
스무딩·보정은 영상 모듈에서 수행하며 모델이 얼굴 각도를 다시 추정하지 않습니다.

softened 0.3이 현재 기본이며, relative 0.2는 하향 전이 보류를 유지하는 비교 모드입니다.
0.1은 과거 설명으로, 현재 코드에 별도 실행 모드가 없습니다. 버전별 실험 표를 혼용하지 마세요.

최근 10초는 실제 시각에 따른 선형 최근성 커널로 유효 구간만 가중평균합니다.
이미 계산된 런타임 이동평균을 publisher가 다시 평균하지 않습니다.
시간 적분은 불규칙 표본을 위한 프로젝트 변형이며 논문의 표본 순서 가중식과 구분합니다.

### 주요 설정값

`ModelConfig`는 불변 설정입니다. 변경하려면 새 설정 객체를 생성합니다.
아래 값은 검증 전 실험값이며 논문 값과 프로젝트 정책의 구분은 상세 수식 문서를 참고하세요.

| 설정 | 기본값 | 적용 |
|---|---:|---|
| `gaussian_sigma_pitch_degrees` / `gaussian_sigma_yaw_degrees` | 15 / 20° | Gaussian |
| `gaussian_angle_mode` | factory 기본 `softened` | 이전 방식은 `relative` |
| `gaussian_pitch_deadband_degrees` / `gaussian_pitch_small_angle_scale` | 2° / 0.75 | Gaussian Pitch 완화 |
| `yaw_neutral_deadband_degrees` / `yaw_small_angle_scale` | 2° / 0.75 | 공통 Yaw 완화 |
| `yaw_full_scale_start_degrees` / `front_pitch_degrees` | 15° / 10° | Yaw/Pitch 완화 경계 |
| `gaussian_down_transition_pitch_degrees` | 10° | Gaussian 하향 전이 |
| `gaussian_down_confirm_seconds` | 0.3초 | 아래 방향 확인 |
| `window_seconds` | Legacy/Gaussian 10초, Huang 30초 | 집계 구간 |
| `output_interval_seconds` | 2초 | publisher 발행 간격 |
| `max_observation_hold_seconds` | 0.3초 | 관측 유지 상한 |
| `min_observable_ratio` / `min_scoreable_ratio` | 0.7 / 0.7 | 구간 품질 기준 |
| `huang_small_seconds` / `huang_amplitude_degrees` / `huang_max_yaw_degrees` | 3초 / 5° / 90° | Huang 온라인 |

Legacy의 각도·시간·점수 계수는 `config.py`와 [상세 수식](docs/model-formulas.md)을 참고합니다.
`recovery_seconds=2`는 설정에 있으나 별도 2초 복귀 확인 로직으로 연결돼 있지 않습니다.

### 현재 점수와 JSON 이벤트

| 값 | 의미 | 현재 전달 범위 |
|---|---|---|
| `current_direction_score` | 최신 유효 관측의 Gaussian 점수, 집계 70% 기준과 독립 | 런타임·독립 웹캠 전용 |
| `score` | Gaussian 최근 10초 집계 | 런타임·웹캠 |
| `payload.model_output.value` | publisher가 해당 근거 구간으로 계산한 집계 | `engagement_window` 이벤트 |
| `status`, `reason_codes` | 런타임 집계의 계산·보류 상태 | 현재 방향 점수 품질과 구분 |

현재 점수가 유효해도 집계는 보류일 수 있습니다. 장시간 하향 후 복귀 사례가 이에 해당합니다.
기존 10초 이력의 점수화 시간 70%를 다시 채우는 데 합성 사례에서 약 7초가 필요할 수 있습니다.
초기 런타임은 첫 관측 이후 부분 이력으로 계산합니다.
정기 이벤트는 publisher 시작 기준 한 구간이 지난 뒤 처음 발행하고 이후 2초마다 발행합니다.
종료 시 `finish()`로 마지막 부분 구간도 발행할 수 있습니다.

JSON에는 현재 방향 점수·진단 각도·`downward_status`·임시 추적 키를 넣지 않습니다.
모델 전용 확장에는 버전·각도 완화·sigma·시간/품질 설정과 관측/점수화 시간 비율을 기록합니다.
이 비율은 참가자 수의 비율이 아닙니다. 공통 인원 분모는 미정이므로 null이며 `signals`는 비어 있습니다.
계약 기준은 [공유 데이터 계약](../공유용_데이터_계약/README.md)과 `integration/contracts.py`입니다.

## 5. 상태와 보류 정책

공통 `classify_orientation()`은 상대 Pitch와 Yaw 완화 후 각도로 다음 순서로 분류합니다.

| 조건 | `orientation_state` |
|---|---|
| 유효 각도 없음 | `not_observable` |
| Pitch ≥ +15°, 또는 직전 아래 보기이고 Pitch > +10° | `looking_down` |
| Yaw 절댓값 ≤ 15°, Pitch 절댓값 ≤ 10° | `front` |
| Yaw 절댓값 ≥ 30°, 또는 Pitch ≤ −20° | `away` |
| 나머지 | `ambiguous` |

Gaussian은 이 상태 분류를 유지하며 별도 `downward_status`로 하향 불확실성을 표시합니다.

- 상대 Pitch > 10°: `transition_uncertain`, 현재·런타임 집계 점수 보류.
- 기존 아래 보기 상태: `looking_down_uncertain`, 연속 +15° 이상 0.3초면 `looking_down_confirmed`.
- 확인은 머리 방향의 확인이며 필기·참여·미참여의 확인이 아닙니다.
- 전이 관측은 처음부터 점수화 시간에서 제외합니다. 관측 가능 시간에는 포함할 수 있습니다.
- 미검출·보정 실패·시간/추적 연속성 문제는 사유와 함께 보류 또는 시간 상태 초기화합니다.
- 정면 복귀의 첫 유효 관측부터 Gaussian 현재 점수를 재개합니다.

필기·교재·휴대전화·손동작을 구분하는 기능은 없습니다.
EAR·깜빡임·표정·상체 이동을 참여 점수나 필기 근거로 합산하지 않습니다.

## 6. 실행 방법 및 테스트 명령어

### 선택 옵션: 보정 없는 시선 방향 모델

독립 데모의 `--gaze`는 MobileGaze **MobileOne S0 ONNX**로 얼굴 영상에서 시선 방향을
추정합니다. 여러 화면 지점을 바라보는 보정은 없습니다. 카메라 기준 방향 후보이므로
실제 모니터 주시·집중도·필기를 확정하지 않습니다. 단일 얼굴용이며 프런트와
`video_recognition.webcam_test`/`webcam_test_mac`에는 연결하지 않았습니다.
주 모델은 Gaussian 전용입니다. 비교 표시를 켜도 Legacy/Huang 정책은 바뀌지 않습니다.

| 화면 표시 | 의미 |
|---|---|
| `forward_candidate` | 모델 시선의 abs(Yaw)≤20°, abs(Pitch)≤15°인 정면 방향 후보 |
| `away_candidate` | abs(Yaw)≥35° 또는 abs(Pitch)≥30°인 이탈 방향 후보 |
| `unknown` | 중간 각도, 미검출, 다중 얼굴, 작은 얼굴, 눈 관측 불가 등 |
| `Gaze Y/P` | 시선 모델의 각도. 기존 머리 `Relative P/Y`와 다른 값 |
| `inference … ms` | 마지막 시선 추론·전처리 시간. 전체 영상 처리 지연이 아님 |

Gaussian에서 이 옵션을 켜면 머리만 숙인 경우에도 각도 점수를 계속 계산합니다.
상대 머리 Pitch≥15°와 모델 시선 Pitch≤−30°/|Yaw|<35°가 함께 유효 추론으로
0.4초 이상 확인될 때 `gaze_down_confirmed`로 점수를 보류합니다. 0점으로 바꾸지 않습니다.
대기 상태는 `gaze_down_pending`입니다. 시선 불명확은 `gaze_unavailable`로 표시하며
유효한 머리 각도가 있으면 각도 점수만 계산합니다. 얼굴 관측 불가는 기존 null 정책입니다.
확인 전 관측은 계산하고 이미 계산한 구간을 소급해서 제외하지 않습니다.
모든 임계값은 검증 전 실험값입니다. 기본 실행·Legacy·Huang의 보류 정책은 바뀌지 않습니다.

기존 얼굴 경계를 재사용하고 최대 약 3Hz로 시선만 추가 추론합니다.
333ms 미만 동안 현재 얼굴 품질이 유효하면 직전 시선 결과를 사용하고,
작은 얼굴·눈 감김·미검출·임시 추적 변경에서는 캐시를 무효화합니다.
얼굴 이미지·시선 이력은 저장하지 않고 기존 FeatureWindow와 JSON 계약도 변경하지 않습니다.
시선 지점 보정과 별개로 기존 **머리 각도 점수를 위한 C/Space 정면 보정은 여전히 필요**합니다.

```bash
# macOS: 기본 영상 환경을 설치한 뒤 최초 한 번
video_recognition/.venv/bin/python -m pip install -r video_recognition/requirements-gaze.txt
video_recognition/.venv/bin/python -m video_recognition.download_gaze_model
# 실행
video_recognition/.venv/bin/python -m engagement_model.webcam_demo --gaze --single-model --debug
```

```powershell
# Windows: 기본 영상 환경을 설치한 뒤 최초 한 번
video_recognition/.venv/Scripts/python.exe -m pip install -r video_recognition/requirements-gaze.txt
video_recognition/.venv/Scripts/python.exe -m video_recognition.download_gaze_model
# 실행
video_recognition/.venv/Scripts/python.exe -m engagement_model.webcam_demo --gaze --single-model --debug
```

모델은 명시적 다운로드 명령으로만 받고 실행 중 영상은 전송하지 않습니다.
설치 범위·출처·검증 한계는 [시선 모델 문서](../video_recognition/docs/pretrained-gaze.md)를 참고하세요.

저장소 루트에서 실행합니다. 영상 환경은 `video_recognition/requirements-integrated.txt`를 기준으로 합니다.
기존 가상환경이 없다면 해당 requirements로 별도 환경을 준비해야 합니다.

```powershell
# 기본 Gaussian 웹캠: 정면 준비 후 버튼 또는 C/Space로 수동 보정
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo
# 상세 진단 / 선택 모델 단독 / 이전 Gaussian 입력 방식
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --debug
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --model huang --single-model
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --gaussian-angle-mode relative
# 수동 보정 후 정면→왼쪽→정면→오른쪽→정면 자동 진단
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --yaw-diagnostic --single-model
```

창에 포커스를 준 뒤 영문 C/Space 또는 CALIBRATE 버튼으로 보정합니다.
collecting 동안 약 3초 정면 유지, D 상세 진단, V 오버레이, Q/Esc 종료입니다.
일반 데모의 `--calibrate`는 자동 보정이며 수동 좌우 진단과 병용할 수 없습니다.
카메라·안내 시간·sigma 후보 옵션과 진단 해석은 [웹캠 문서](docs/webcam-diagnostics.md)를 참고하세요.

독립 웹캠은 완료 보정을 일시적인 단일 얼굴 미검출·품질 저하 동안 유지합니다.
다중 얼굴·손상된 보정값은 재보정 대상으로 표시합니다. 사용자가 바뀌거나 카메라가 움직이면 재보정하세요.
이 복귀 정책은 독립 실행기에만 적용하며 팀원 웹캠·프런트 보정 정책을 바꾸지 않습니다.

### 테스트 명령어

```powershell
# 카메라 없이 모델·시간 상태·계약·실행기 모의 검사
video_recognition/.venv/Scripts/python.exe -B -m unittest discover -s engagement_model/tests -v
# 동일 합성 입력의 세 모델 비교: 실제 정확도 평가가 아님
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.tests.test_huang --compare-all
```

## 7. 현재 구현 상태와 검증 중인 문제

### 코드로 확인한 구현

구현: 세 모델 선택, 입력 검증, 상태·시간 관리, 하향 보류, Gaussian 현재/집계 분리,
독립 웹캠 자동 복귀·기본 얼굴 시각화, 수동 보정과 자동 구간 안내, publisher의 계약 발행입니다.
시간 역전·관측 공백·트랙 변경·잘못된 입력에 대한 처리와 회귀 테스트가 있습니다.
현재 CLI 기본값은 `webcam_demo.py::main()`, 프런트 기본값은 `LiveSession._start_model()`에서 확인합니다.
이전 README의 «미구현» 목록·테스트 개수는 현재 상태표로 사용하지 않습니다.
테스트 명령과 실제 관찰 결과를 함께 확인하고, 미실행 검증을 통과했다고 간주하지 마세요.
테스트 커버리지와 기존 실험의 해석 범위는 [실험·검증 문서](docs/experiments.md)에 구분했습니다.
### 검증 중인 문제와 미확인 사항

웹캠에서 큰 좌우 상대각 차이와 회전 중 Pitch/Roll 변동이 관찰됐지만 원인은 확정하지 못했습니다.
보정·완화 산술의 일치가 원본 추정이나 보정 자세의 정확성을 증명하지는 않습니다.
실제 좌우 회전량의 정답과 실제 참여도 정확도는 현재 코드만으로 확인할 수 없습니다.
A/B/C ambiguous 보류·이탈 유예 비교는 분석 실험이며 Gaussian 기본 정책으로 구현하지 않았습니다.
개별 계산은 임시 정면 보정을 사용하며 다인 경로는 ID별 계산을 분리합니다. 실제 참여도 정확도나 논문 성능을 달성했다고 주장하지 않습니다.

## 8. 향후 개발 계획

1. 영상 담당자와 원본 회전·보정·추정 품질을 확인하고 좌우 비대칭 원인을 구분합니다.
2. 동일 입력·보류 구간·사람 주석 기준을 고정한 뒤 sigma와 정책을 평가합니다.
3. 추가 시선·필기 특징은 비전 담당자가 제공한 근거와 품질을 확인한 뒤 연결합니다.
4. 프런트 모델 선택·현재 점수 전달은 기존 집계 의미와 공통 계약을 먼저 협의합니다.
5. 다중 얼굴별 보정·연결·집계는 별도 설계·동의·검증이 필요한 미완료 범위로 둡니다.

위 항목은 개발 계획이며 구현 완료나 성능 달성으로 해석하지 않습니다.

## 9. 참고 논문 및 자료

- Huang, Z.; Zhuo, R.; Gao, F. (2024), *How attentive is a student in class? A concentration evaluation method based on head Euler angle*.
  [DOI](https://doi.org/10.1016/j.heliyon.2024.e37365) · [원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC11417239/)
- Li, B.; Liu, P. (2024), *Online Learning State Evaluation Method Based on Face Detection and Head Pose Estimation*, Sensors 24(5), 1365.
  [DOI](https://doi.org/10.3390/s24051365)

원식·온라인 변형·예외 정책의 구분은 [상세 수식](docs/model-formulas.md)에 정리했습니다.
과거 실험 표는 [이력 문서](docs/implementation-history.md)의 당시 조건으로만 해석합니다.
