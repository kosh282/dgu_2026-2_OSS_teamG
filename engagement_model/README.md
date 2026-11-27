# 참여도 모델

## 1. 프로젝트 개요 및 목적

온라인 수업 영상에서 얻은 머리 방향과 관측 시간을 이용해 0~100의 관찰 점수를 계산합니다.
점수는 교수자의 수업 성찰을 위한 참고 지표이며 실제 집중도·이해도·성적의 측정값이 아닙니다.
관측 불가와 판단 보류는 0점 대신 `null`과 사유로 표현합니다.

- 독립 웹캠과 프런트의 기본 모델은 **Gaussian softened**입니다.
- 모델은 Gaussian만 제공합니다. `ModelConfig()`와 모든 웹캠 점수기도 Gaussian을 사용합니다.
- 독립 데모는 Gaussian의 상대각/완화 및 보정 방식을 비교합니다. 프런트에서 재식별을 함께 켜면 다인 Gaussian 경로로 실행합니다.
- 개별 계산기는 단일 얼굴 입력을 사용하며, 프런트는 재식별 ID 또는 임시 위치 트랙마다 계산기를 따로 둡니다.
- 현재 Gaussian 점수와 최근 5초 집계 점수는 다른 값입니다.
- 원본 영상·얼굴 이미지·영구 식별자를 이 모듈의 저장 항목으로 추가하지 않습니다.

상세 문서:

- [모델별 수식·원식과 변형·버전](docs/model-formulas.md)
- [실험 조건·버전별 결과·검증 범위](docs/experiments.md)
- [웹캠 조작·자동 좌우 진단·결과 해석](docs/webcam-diagnostics.md)

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

현재 프런트·Windows 데모의 그룹 provenance는 재식별 사용 시 `multi-face-gaussian` / `multi-gaussian-1.3-head-eye-experimental`,
미사용 시 `position-track-gaussian` / `position-gaussian-1.4-head-eye-experimental`입니다. 머리 70%와 눈 30% 결합값을 집계합니다.
`group_developer_view.group_model_identity()`가 실제 생성기와 세션 설정·상태 이벤트의 모델 정보를 함께 제공합니다.
`WindowPublisher`는 다인 런타임의 `aggregate_window()`를 사용합니다. 각 유효 관측의
같은 시각의 머리 점수와 눈 상태 점수를 70/30으로 결합하고 최근 5초의 결합값을 선형 최근성 가중치로 집계합니다. 양쪽 눈을 2초 연속 감은 이후에는 눈 점수를 초당 20점씩 0점까지 낮추고 눈을 뜨면 100점으로 복귀합니다. 관측 공백은 점수화하지 않고 연속 감김 시간을 다시 계산합니다.
분자는 모든 유효한 사람·구간의 `시간가중치 × 결합 점수` 합,
분모는 같은 사람·구간의 시간가중치 합입니다. 개인의 이동 평균을 다시 평균하지 않습니다.
보류·미검출·보정 중인 사람은 0점을 넣지 않습니다. 이 값은 관측 가능한 사람들에 대한 참고 집계이며
전체 수강생 평균·출석률·집중도는 아닙니다.

임시 설정의 구간은 최근 5초/출력 1초이며 최소 관측·점수화 시간 기준은 각각 60%입니다.
비율은 해당 구간에 적어도 한 사람의 유효 관측/점수가 있는 **벽시계 시간**을 분모로 계산합니다.
`model_extensions.multi-face-gaussian.scoreable_person_duration_ms`는 별도의
점수화 가능한 사람·밀리초 합이므로 구간 길이보다 클 수 있습니다.
예를 들어 두 명을 각각10초 유효 관측하면20000입니다. 이 값은 최근성 가중치 합 자체는 아닙니다.
개별 원시 각도는 집계 메모리에만 있고 저장 이벤트에는 없습니다. 공통 참가자 수 분모는 null입니다.

프런트는 재식별을 끄면 단기 위치 트랙별 다인 경로를 사용합니다. 트랙은 신원이 아닙니다. 같은 트랙이 1초 이내에 안정된 위치로 돌아오면 완료된 정면 기준은 유지하지만, 0.5초를 넘긴 누락 구간의 점수 이력은 초기화합니다. 1초를 넘긴 누락이나 큰 이동은 해당 기준을 다시 수집하며, 겹침·동시 교체·긴 공백은 모든 기준을 폐기할 수 있습니다. 보정 중 일시적인 품질 저하 표본은 버리되, 마지막 유효 표본 이후 1초를 초과하면 수집을 다시 시작합니다.
MobileGaze 시선 실험은 포함하지 않습니다. [프런트 실행 안내](../frontend/README.md)를 참고하세요.

## 2. 전체 데이터 처리 흐름

### 하위 20% 평균 선택 지표

다인 런타임의 `aggregate_window()`와 `WindowPublisher`는 기존 그룹 점수와 별도로
하위 20% 평균을 제공합니다. 기존 `payload.model_output.value`와 그룹 점수식은 변경하지 않습니다.
두 값을 가중 결합하는 기능은 없습니다.

1. 이벤트의 동일한 `[start_ms, end_ms)` 구간(기본 최근 5초)에서 각 ID/위치 트랙의
   유효 원시 점수를 기존 선형 시간 가중식으로 적분합니다. 개인 이동평균을 다시 평균하지 않습니다.
2. 개인별 관측·점수화 시간 기준(기본 각각 60%)을 통과해야 하며, 구간 끝 직전 관측이
   미검출·보류이거나 관측 유지 상한(기본 0.5초)을 넘으면 제외합니다. null을 0점으로 넣지 않습니다.
   머리·눈 중 한 항목의 기존 최대 2초 유지 정책으로 유효한 결합 점수가 나오면 그대로 사용합니다.
3. 유효 점수 N개를 오름차순 정렬하고 `K = max(1, ceil(N * 0.2))`개의 산술평균을 계산합니다.
   N=0이면 평균은 null, 두 인원 필드는 0입니다. 유효한 0점은 제외하지 않습니다.

프런트 담당자는 `service.get_dashboard_snapshot(sid)['engagement_windows']`의 각 이벤트에서
`payload.model_extensions[event.provenance.model_id]`를 읽습니다. 재식별 사용 시 이름 공간은
`multi-face-gaussian`, 미사용 시 `position-track-gaussian`입니다.

| 선택 필드 | 의미 |
|---|---|
| `lower_20_mean_score` | 0~100 하위 평균 또는 null, 소수 첫째 자리 |
| `lower_20_count` | 평균에 포함된 K개 ID/위치 트랙 수 |
| `lower_20_valid_count` | 선정 분모 N: 위 기준을 통과한 ID/위치 트랙 수 |
| `lower_20_fraction` | 0.2 |
| `lower_20_policy` | `individual-window-ceil-v1`: 선택적 확장 정책 버전 |

N=1/2/4/10이면 K=1/1/1/2입니다. 실제 수강생 수·출석률이 아니며 공통
`observation.valid_count`/`eligible_count`는 계속 null입니다. 개인 키와 점수 이력은 저장하지 않고
최대 한 집계 구간의 표본만 메모리에 둡니다. 학생별 정면 재보정 시 해당 표본을 비웁니다.

기존 전체 점수는 유효 사람·시간을 합친 값이고, 새 값은 개인별 시간 집계 후 선정한 평균이므로
현재 미검출자 처리와 분모가 다릅니다. 새 지표를 전체 점수의 대체값으로 사용하지 않습니다.
단일 얼굴 독립 웹캠과 런타임의 `latest`에는 추가하지 않으며, 실제 프런트 표시는 담당자가 연결해야 합니다.
이전 JSON에는 필드가 없으므로 `.get()`으로 읽고 **없음/null을 0점으로 표시하지 마세요.**

```text
video_recognition: 얼굴 특징 → 스무딩 → 정면 보정 → FeatureWindow
  ↓ model_consumer 콜백
input_adapter.py: 형식·단일 얼굴·보정·품질 검사, 상대각/Yaw 완화
  ↓
runtime.py: 방향 상태·연속 관측 시간 관리
  ↓ gaussian.py: Gaussian 점수 계산
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
| [scoring.py](scoring.py) | Gaussian용 각도 검증·방향 상태 분류 |
| [runtime.py](runtime.py) | `HeadOrientationState`: Gaussian용 방향·시간 공통 상태 |
| [gaussian.py](gaussian.py) | `GaussianRuntime`, 방향 점수·시간 가중 집계 |
| [window_publisher.py](window_publisher.py) | 구간 집계와 공통 계약 이벤트 발행 |
| [webcam_demo.py](webcam_demo.py) | 기존 영상 기능을 재사용하는 독립 웹캠·메모리 진단 |
| [tests/](tests/) | 입력·산정식·시간 상태·복귀·계약·실행기 모의 검사 |

## 3. Gaussian 전용 모델

Legacy/Huang 계산식·설정·비교 실행기는 제거했습니다. `--huang-only`, `--model legacy`, `--model huang`은 사용할 수 없습니다. `--model gaussian`과 `--single-model`은 기존 실행 명령의 호환을 위해 유지합니다.

```python
from engagement_model.config import ModelConfig
from engagement_model.gaussian import GaussianRuntime

gaussian = GaussianRuntime()  # softened, angle-time-0.4
reference = GaussianRuntime(ModelConfig.gaussian(gaussian_angle_mode='relative'))
```

상체 분석은 실행하지 않습니다. 어깨 좌표·상체 움직임과 상체 오버레이·분석 주기 옵션을 제거했습니다. 얼굴 머리 방향, 눈·표정 특징, 재식별 및 독립 데모의 나머지 진단·시선 옵션은 유지합니다.

## 4. 현재 Gaussian 산정식과 설정값

논문의 화면 투영 좌표 대신 프로젝트의 정면 보정 상대각을 사용하는 **변형식**입니다.
P/Y는 `head_angles_relative_degrees`의 Pitch/Yaw(도)입니다.
기본 softened 모드는 상태 판정용 원본 상대각을 보존하고 점수 입력만 완화합니다.

```text
h(a; d,k,b) = 0                                              if |a| <= d
             sign(a)*[k*(min(|a|,b)-d)+max(0,|a|-b)]          otherwise
P_eff = h(P; 2, 0.75, 9)
Y_eff = h(Y; 2, 0.75, 15)
S_current = 100*exp(-0.5*((P_eff/15)^2+(Y_eff/16)^2))
```

Yaw 완화는 `soften_angle()`을 사용합니다.
Pitch 완화와 sigma 15/20도는 프로젝트 실험 설정이며 논문의 최적값이 아닙니다.
큰 각도의 기울기는 1을 유지합니다. Roll은 점수에 사용하지 않습니다.
relative 비교 모드는 완화 전 P/Y를 그대로 사용합니다.
스무딩·보정은 영상 모듈에서 수행하며 모델이 얼굴 각도를 다시 추정하지 않습니다.

softened 0.4가 현재 기본이며, relative 모드도 동일한 아래 보기 정책을 사용합니다.
0.1은 과거 설명으로, 현재 코드에 별도 실행 모드가 없습니다. 버전별 실험 표를 혼용하지 마세요.

최근 5초는 실제 시각에 따른 선형 최근성 커널로 유효 구간만 가중평균합니다.
이미 계산된 런타임 이동평균을 publisher가 다시 평균하지 않습니다.
시간 적분은 불규칙 표본을 위한 프로젝트 변형이며 논문의 표본 순서 가중식과 구분합니다.

### 주요 설정값

`ModelConfig`는 불변 설정입니다. 변경하려면 새 설정 객체를 생성합니다.
아래 값은 검증 전 실험값이며 논문 값과 프로젝트 정책의 구분은 상세 수식 문서를 참고하세요.

| 설정 | 기본값 | 적용 |
|---|---:|---|
| `gaussian_sigma_pitch_degrees` / `gaussian_sigma_yaw_degrees` | 15 / 16° | Gaussian |
| `gaussian_angle_mode` | factory 기본 `softened` | 이전 방식은 `relative` |
| `gaussian_pitch_deadband_degrees` / `gaussian_pitch_small_angle_scale` | 2° / 0.75 | Gaussian Pitch 완화 |
| `yaw_neutral_deadband_degrees` / `yaw_small_angle_scale` | 2° / 0.75 | 공통 Yaw 완화 |
| `yaw_full_scale_start_degrees` / `front_pitch_degrees` | 15° / 9° | Yaw/Pitch 완화 경계 |
| `gaussian_down_transition_pitch_degrees` | 9° | Gaussian 하향 전이 |
| `gaussian_down_confirm_seconds` | 0.3초 | 아래 방향 확인 |
| `window_seconds` | 5초 | 임시 집계 구간 |
| `output_interval_seconds` | 1초 | publisher 발행 간격 |
| `max_observation_hold_seconds` | 0.5초 | 관측 유지 상한 |
| `min_observable_ratio` / `min_scoreable_ratio` | 0.6 / 0.6 | 구간 품질 기준 |


### 현재 점수와 JSON 이벤트

| 값 | 의미 | 현재 전달 범위 |
|---|---|---|
| `current_direction_score` | 최신 유효 관측의 Gaussian 점수, 집계 60% 기준과 독립 | 런타임·독립 웹캠 전용 |
| `score` | Gaussian 최근 5초 집계 | 런타임·웹캠 |
| `payload.model_output.value` | publisher가 해당 근거 구간으로 계산한 집계 | `engagement_window` 이벤트 |
| `status`, `reason_codes` | 런타임 집계의 계산·보류 상태 | 현재 방향 점수 품질과 구분 |

현재 점수가 유효해도 구간 내 유효 관측 시간이 부족하면 집계는 보류될 수 있습니다.
5초 이력의 점수화 시간 60%를 다시 채우는 데 약 3초의 유효 관측이 필요합니다.
초기 런타임은 첫 관측 이후 부분 이력으로 계산합니다.
정기 이벤트는 publisher 시작 기준 한 구간이 지난 뒤 처음 발행하고 이후 기본 1초마다 발행합니다.
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
| Pitch ≥ +10°, 또는 직전 아래 보기이고 Pitch > +7° | `looking_down` |
| Yaw 절댓값 ≤ 15°, Pitch 절댓값 ≤ 9° | `front` |
| Yaw 절댓값 ≥ 30°, 또는 Pitch ≤ −20° | `away` |
| 나머지 | `ambiguous` |

Gaussian은 이 상태 분류를 유지하며 별도 `downward_status`로 하향 불확실성을 표시합니다.

- 상대 Pitch > 9°이고 아래 보기 상태가 아직 아니면 `transition_uncertain`으로 점수를 보류합니다.
- 기존 아래 보기 상태: `looking_down_uncertain`, 연속 +10° 이상 0.3초면 `looking_down_confirmed`.
- 아래 보기 진입 전 유효한 머리 집계 점수가 있으면 5초 유지하고, 이후 초당 7점씩 0점까지 감소합니다. 눈 점수는 별도 계산하므로 7:3 결합값의 감소량은 눈 상태와 최근 5초 집계에 따라 달라집니다. 유효한 기준 점수가 없으면 보류합니다. 임시 실험 정책이며 집중도 판단이 아닙니다.
- 확인은 머리 방향의 확인이며 필기·참여·미참여의 확인이 아닙니다.
- 전이 관측은 처음부터 점수화 시간에서 제외합니다. 관측 가능 시간에는 포함할 수 있습니다.
- 미검출·보정 실패·시간/추적 연속성 문제는 사유와 함께 보류 또는 시간 상태 초기화합니다.
- 정면 복귀의 첫 유효 관측부터 Gaussian 현재 점수를 재개합니다.

### 아래 보기 후 복귀 패턴의 규칙 기반 필기 간주

독립 웹캠은 중앙 얼굴의 분할 경계 누락을 줄이도록 전체 화면을 매 분석 시점에 검사합니다
(`FaceAnalyzer(full_frame_interval_ms=1)`). 프런트의 검출 주기는 변경하지 않습니다.

독립 웹캠의 왼쪽 아래 `ACTIVITY`와 `Down episode`에서 판단 상태와 하향 구간 길이를
확인합니다. 기본 화면과 D 디버그 화면 모두 표시하며 `writing_assumed`는 초록색입니다.
정면 복귀 후 표시되는 시간은 완료한 하향 구간의 길이이며 정면 유지시간이 아닙니다.

`GaussianRuntime`은 기존 `looking_down` 분류·연속 시간과 세션 시각을 재사용해
별도의 `activity_state`를 반환합니다. 손·필기구·교재를 시각적으로 확인한 결과가 아니라
**프로젝트 정책상 짧은 하향 후 정면 복귀를 필기로 간주하는 실험 규칙**입니다.

| `activity_state` | 의미 |
|---|---|
| `None` | 판단할 하향 에피소드가 없는 일반 관측 |
| `downward_pending` | 아래 보기 중 또는 정면 복귀 안정성 확인 중 |
| `writing_assumed` | 최초 하향부터 5초 이내 정면 복귀 후 안정성 확인 완료 |
| `downward_prolonged` | 5초를 초과한 하향 에피소드(복귀 후 최근 결과로 남을 수 있음) |
| `activity_unavailable` | 관측·보정·입력 불가 또는 별도 gaze 정책 사용 |

정면 복귀는 기존 `front` 각도 기준으로 판정하고, 기존 `front_seconds`가
`gaussian_down_confirm_seconds`(기본 0.3초)에 도달해야 필기 간주를 확정합니다.
별도 복귀 타이머나 새 임계값은 만들지 않습니다. 5초 경계는 확인 완료 시각이 아니라
**처음 정면으로 돌아온 시각**으로 비교하므로 정확히 5초 복귀도 포함합니다.
한 프레임의 정면 흔들림 후 다시 아래를 보면 에피소드를 끝내지 않습니다.
`activity_down_start_ms`와 `activity_down_duration_seconds`는 메모리 진단용이며,
지속시간은 미확정 정면 흔들림을 포함하는 에피소드 경과시간이고 최초 복귀 시각에서 고정합니다.
관측 불가·추적 변경·재보정은 후보를 취소합니다. 확정 결과는 다음 에피소드나 상태 중단까지 유지합니다.

점수는 5초 유예와 5초 초과분에 대한 초당 7점 감점을 사용합니다. 필기 간주에 대한
추가 점수나 100점 보정, 과거 이벤트의 소급 변경은 없습니다. 정면 복귀 첫 유효 관측의
Gaussian 재계산은 활동 확인 0.3초를 기다리지 않습니다. 눈 감김은 기존 EAR·2초 유예·
70/30 결합을 독립적으로 유지하며 눈 특징 결측을 100점으로 대체하지 않습니다.

다인 모델의 `participants()`와 프런트 `participant_snapshot()`에는 ID/위치 트랙별 선택 필드로
전달됩니다. **현재 프런트 표는 이 필드를 표시하지 않으며 그룹 JSON 이벤트에도 넣지 않습니다.**
자세 상태·점수·집계·기존 JSON 계약은 바뀌지 않습니다. 필기·교재·휴대전화·손동작의
실제 구분 기능은 여전히 없으며, 이 규칙의 행동 판별 정확도는 검증되지 않았습니다.

프런트는 EAR 기반 장시간 양안 감김 후보를 머리 점수와 결합합니다. 깜빡임 횟수·표정은 점수나 필기 근거로 합산하지 않습니다.

## 6. 실행 방법 및 테스트 명령어

### 선택 옵션: 보정 없는 시선 방향 모델

독립 데모의 `--gaze`는 MobileGaze **MobileOne S0 ONNX**로 얼굴 영상에서 시선 방향을
추정합니다. 여러 화면 지점을 바라보는 보정은 없습니다. 카메라 기준 방향 후보이므로
실제 모니터 주시·집중도·필기를 확정하지 않습니다. 단일 얼굴용이며 프런트와
`video_recognition.webcam_test`/`webcam_test_mac`에는 연결하지 않았습니다.
주 모델은 Gaussian 전용입니다.

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
모든 임계값은 검증 전 실험값입니다. `--gaze`를 사용하지 않으면 기본 Gaussian 하향 정책을 적용합니다.

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
# 상세 진단 / 이전 Gaussian 입력 방식
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --debug
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
```

### 합성 입력별 테스트 기대값

| 시연 라벨 | 합성 입력 | 테스트 기대값 |
| --- | --- | --- |
| 정면 | 기준 자세 | 현재 방향 점수 100 |
| 다른 곳 응시 | 상대 yaw 60° | 현재 방향 점수 0.3, 정면 복귀 시 100 |
| 필기 유예 | pitch 20°, 5초 이내 안정적으로 정면 복귀 | 기준 점수 유지, `writing_assumed` |
| 눈 감김 | 양안 EAR 0.1을 5초간 유지 | 눈 후보 점수 70 |
| 지속 하향 | pitch 20°를 10초·15초 유지 | 현재 방향 점수 65·30 |

근거는 [Gaussian](tests/test_gaussian.py), [필기 유예](tests/test_writing_activity.py), [눈 감김](tests/test_eye_candidate.py), [하향 정책](tests/test_downward_policy.py) 테스트입니다.

### 최근 데모 세션에서 관찰한 점수 변화

2026-11-26 로컬 테스트 세션(10분 09초)의 5초 그룹 점수 창 586개 중 581개에 숫자 점수가 있었다. 아래 평균은 세션 시각 기준 30초 단위로 창의 종료 시각을 묶어 계산했다. 보류 창은 평균에서 제외했다.

| 세션 시각 | 평균 | 관측 최저–최고 |
| --- | ---: | ---: |
| 00:00–00:30 | 100.0 | 100.0–100.0 |
| 00:30–01:00 | 68.4 | 57.3–100.0 |
| 03:30–04:00 | 77.2 | 48.1–93.9 |
| 05:30–06:00 | 97.3 | 93.0–99.9 |
| 06:00–06:30 | 98.4 | 94.1–99.9 |
| 08:00–08:30 | 92.4 | 80.9–99.5 |
| 08:30–09:00 | 70.0 | 53.2–97.8 |

초반 기준 구간 이후 평균이 31.6점 떨어졌고, 중반에는 97점 이상으로 회복했다. 후반에도 인접 구간 평균이 92.4점에서 70.0점으로 하락했다. 최근 5초의 관측값을 집계하므로 행동 전환 직후에는 이전 구간 신호가 일부 남을 수 있다.

공유 데모 초안은 4명이 30초씩 행동하고 동일 행동을 2명·3명 조건으로 비교하도록 설계했다. 초안의 총 길이(10분 30초)와 저장 세션의 길이(10분 09초)가 다르고, 최종 행동별 시작 시각과 재생 시작 오프셋은 확인이 필요하다. 위 표는 세션 시간 기준이며, 라벨별 효과와 2명·3명의 차이는 최종 큐시트와 정렬한 뒤 비교한다.

## 7. 현재 구현 상태와 검증 중인 문제

현재 기본 실험 설정은 최근 5초 집계, 아래 보기 5초 유예 뒤 초당 7점 감점, 양쪽 눈 감김 2초 유예 뒤 초당 20점 감점, 머리·눈 70/30 결합입니다. 회귀 테스트에는 이 설정의 기대 동작이 반영되어 있습니다. 설정값과 점수 타당성은 실제 수업 및 사람 주석으로 검증되지 않았습니다.

### 코드로 확인한 구현

구현: Gaussian 전용 실행, 입력 검증, 상태·시간 관리, 하향 유예·감점, Gaussian 현재/집계 분리,
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
4. 다인 얼굴별 보정·연결·집계의 실제 수업 화면 성능을 추가 검증합니다.

위 항목은 개발 계획이며 구현 완료나 성능 달성으로 해석하지 않습니다.

## 9. 참고 논문 및 자료

- Li, B.; Liu, P. (2024), *Online Learning State Evaluation Method Based on Face Detection and Head Pose Estimation*, Sensors 24(5), 1365.
  [DOI](https://doi.org/10.3390/s24051365)

원식·온라인 변형·예외 정책의 구분은 [상세 수식](docs/model-formulas.md)에 정리했습니다.

### 프런트 머리·눈 결합 점수

프런트의 다인 경로는 머리·눈 결합 점수를 익명 사람·시간 가중 점수로 발행합니다. 눈 관측이 부족하면 해당 얼굴의 결합 점수는 보류합니다. 개발자 창의 개인별 결합 점수 평균은 저장 구간 점수와 다를 수 있습니다.

눈 상태는 양쪽 EAR이 0.21 미만으로 2초 이상 지속된 이후 초당 20점씩 감점하며, 짧은 깜빡임은 감점하지 않습니다. 유효한 각 시점의 머리·눈 상태를 70/30으로 결합한 뒤 최근 5초를 시간 가중 집계합니다. 관측 유지 상한은 0.5초, 구간 품질 기준은 60%입니다. 양쪽 EAR 또는 트랙을 얻지 못하면 해당 관측을 제외하며 관측 불량을 감김으로 취급하지 않습니다. 별도 눈 점수 진단은 최근 5초 유효 눈 점수의 시간 평균입니다. `0.21`, 2초, 초당 20점, 60%, 70/30은 검증 전 실험값입니다.

다인 그룹 점수는 머리와 눈을 모두 관측한 사람·시간만 집계합니다. 따라서 전체 수강생의 평균이나 참여율이 아니며, 눈이 보이지 않는 얼굴은 집계에서 빠집니다. 이 점수는 눈 동작의 관찰 후보이며 졸음·집중도·이해도를 뜻하지 않습니다. 얼굴 크기·프레임 주기·안경·조명에 따른 성능과 사람 주석을 이용한 평가가 필요합니다. 얼굴별 눈 특징과 개인 점수는 최종 세션에 저장하지 않습니다. 결합된 그룹 구간 점수와 배점 설정만 저장합니다.

재식별 없는 프런트는 단기 위치 트랙별로 보정을 유지하며, 연결이 모호하거나 장시간 누락되면 해당 기준을 재수집합니다.
# 다인 점수와 보정 유지

머리와 눈은 각 임시 트랙별로 독립 관측합니다. 두 항목 중 하나만 일시 누락되면 같은 트랙에서 마지막으로 측정한 값을 최대 2초 동안 결합 점수에 사용합니다. 개발자 화면에 이전 값과 경과 시간이 표시되며, 이를 포함한 집계는 `partial`, `component_value_held`로 표시합니다. 두 항목이 모두 없거나 얼굴이 사라진 동안에는 점수를 이어서 생성하지 않습니다.

머리 각도 이력의 일시적인 초기화는 눈 관측 이력을 지우지 않습니다. 눈 샘플 사이 허용 간격은 0.8초입니다. 0.5~0.8초 공백이나 짧은 눈 누락 후 같은 트랙에서 양쪽 눈 감김이 다시 확인되면 이전의 확인된 감김 시간은 유지하되 공백은 감김·관측 시간에 더하지 않습니다. 눈을 뜨거나 공백이 0.8초를 넘거나 얼굴 추적이 끊기면 감김 시간은 다시 시작합니다. 눈 점수에 필요한 최근 5초 관측률은 60%입니다. 개발자 진단의 `one_eye_missing`, `both_eyes_missing`, `analysis_gap`, `analysis_gap_partial`, `face_not_detected`는 현재 누락 또는 공백의 단서이며, 횟수는 해당 런타임이 재설정되기 전까지의 누적 프레임 수입니다. 이 값들은 최종 세션에 개인별로 저장하지 않습니다.

아래보기 중 같은 트랙의 관측 공백이 0.8초 이내이고 복귀 후에도 아래보기가 확인되면 이전에 관측한 아래보기 시간과 감점 기준을 유지합니다. 공백은 지속 시간에 더하지 않고, 더 긴 공백이나 트랙 변경 시에는 다시 수집합니다. 아래보기는 5초 유예 후 초당 7점 감점합니다. 개발자 화면에서 아래보기 상태와 관측된 지속 시간을 확인할 수 있습니다.

정면 보정은 각 트랙에 별도로 유지하며, 같은 트랙이 확인되는 짧은 관측 공백은 최대 5초까지 허용합니다. 위치가 크게 변하거나 트랙 연결이 모호한 사람은 보정을 다시 수집합니다. 한 사람의 화면 이탈이나 새 타일 등장으로 계속 추적 중인 다른 사람의 보정을 지우지 않습니다. 위치 추적은 신원 확인이 아니므로 동일한 위치에 다른 사람이 나타나는 경우 구분할 수 없습니다. 영상회의 배치 변경이나 가림이 심하면 재보정이 필요할 수 있습니다.
