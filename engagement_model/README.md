# 참여도 모델

## 1. 프로젝트 개요 및 목적

온라인 수업 영상에서 얻은 머리 방향과 관측 시간을 이용해 0~100의 실험 점수를 계산합니다.
점수는 교수자의 수업 성찰을 위한 참고 지표이며 실제 집중도·이해도·성적의 측정값이 아닙니다.
관측 불가와 판단 보류는 0점 대신 `null`과 사유로 표현합니다.

- 독립 웹캠의 기본 모델은 **Gaussian softened**입니다.
- `EngagementRuntime()`과 현재 프런트의 기본 모델은 **Legacy**입니다.
- Huang·Gaussian은 선택 가능한 실험 모델이며 프런트 모델 선택 UI는 없습니다.
- 입력은 단일 얼굴의 시점별 `FeatureWindow`입니다. 다중 학생별 분석은 지원하지 않습니다.
- 현재 Gaussian 점수와 최근 10초 집계 점수는 다른 값입니다.
- 원본 영상·얼굴 이미지·영구 식별자를 이 모듈의 저장 항목으로 추가하지 않습니다.

상세 문서:

- [최신 main 호환성 검토·커밋 대상 명세](docs/integration-review.md)

- [모델별 수식·원식과 변형·버전](docs/model-formulas.md)
- [실험 조건·버전별 결과·검증 범위](docs/experiments.md)
- [웹캠 조작·자동 좌우 진단·결과 해석](docs/webcam-diagnostics.md)
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
현재 `frontend/live_session.py`는 Legacy 런타임과 publisher를 연결합니다.
독립 Gaussian 웹캠 기본은 기존 Euler 보정입니다. 명시적으로 선택하는 실험용 matrix 경로만
`rotation_calibration.py`에서 Raw 회전행렬 기준으로 보정한 입력 복사본을 사용합니다.
Legacy·Huang은 기존 보정 입력을 유지합니다. 이 행렬 보정은 프런트나 일반 `GaussianRuntime()`에 자동 적용되지 않습니다.

| 파일 | 책임 |
|---|---|
| [config.py](config.py) | `ModelConfig`: 실험 설정과 값 검증 |
| [input_adapter.py](input_adapter.py) | `HeadAngles`, `Observation`, `parse_feature_window()`, `soften_angle()` |
| [scoring.py](scoring.py) | 공통 상태 분류, Legacy 규칙, Huang 순수 함수·구간 계산 |
| [runtime.py](runtime.py) | `EngagementRuntime`, 세 모델을 같은 입력으로 실행하는 `ComparisonRuntime` |
| [gaussian.py](gaussian.py) | `GaussianRuntime`, 방향 점수·시간 가중 집계 |
| [rotation_calibration.py](rotation_calibration.py) | 독립 Gaussian 웹캠의 상대 회전행렬 보정·입력 어댑터 |
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
일반 입력의 스무딩·보정은 영상 모듈이 담당합니다. 독립 Gaussian 웹캠의 matrix 경로는
Raw Euler를 회전행렬로 복원해 보정하며, Euler 축별 스무딩 결과를 사용하지 않습니다.
상대 회전은 기준 Yaw/Roll을 반영한 좌표계에서 해석해 아래 보기의 양의 Pitch 계약을 유지합니다.
이 좌표계 수정은 합성 회전으로 검증했으며 실제 웹캠의 좌우 비대칭 개선 여부는 재검증이 필요합니다.
얼굴 특징이나 solvePnP를 재추정하지 않으며 Gaussian 식·sigma·상태 정책은 유지합니다.

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

MediaPipe 얼굴 변환행렬 비교는 `--compare-face-transform`으로 선택합니다.
주 표시·진단 후보로 사용하려면 `--pose-source face-transform`, PnP 복귀는 `--pose-source pnp`입니다.
행렬 결측을 PnP 점수로 대체하지 않습니다. 전체 기본값은 검증 완료 전까지 PnP로 유지합니다.
한 번의 동일 얼굴 검출에서 PnP와 행렬 각도를 얻고, 같은 설정의 스무딩·Euler 보정·Gaussian을 각각 실행합니다.
기본 주 모델과 공통 영상 코드는 교체하지 않습니다. `--validated-pose`를 함께 쓰면 비교 기준은 검증 PnP입니다.
실행 예시와 좌표계·해석 한계는 [웹캠 진단 문서](docs/webcam-diagnostics.md)를 참고하세요.

원본 PnP의 음의 깊이 해를 제외하는 독립 웹캠 실험은 `--validated-pose --debug`로 선택합니다.
기존 검출·눈·표정 추출을 재사용하고 머리 자세만 SQPnP로 다시 추정합니다.
양의 깊이·재투영 오차·회전 연속성을 검사하며 실패 관측은 `pose_unreliable`로 보류합니다.
공통 영상 모듈과 프런트에는 적용하지 않습니다. 실제 개선 여부는 웹캠 검증이 필요합니다.

저장소 루트에서 실행합니다. 영상 환경은 `video_recognition/requirements-integrated.txt`를 기준으로 합니다.
기존 가상환경이 없다면 해당 requirements로 별도 환경을 준비해야 합니다.

```powershell
# 기본 Gaussian 웹캠: 정면 준비 후 버튼 또는 C/Space로 수동 보정
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo
# 상세 진단 / 선택 모델 단독 / 이전 Gaussian 입력 방식
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --debug
# 새 행렬 보정과 기존 Euler 보정의 Gaussian 점수를 같은 영상에서 비교
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --single-model --calibration-method matrix --compare-calibration
# 기존 보정만 사용하려면 --calibration-method euler
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
같은 회전의 Euler 표현 전환에 대한 보정 오류는 새 행렬 경로에서 회귀 검증했습니다.
실제 solvePnP 회전 자체의 오류나 카메라·얼굴 모형 오차가 해결됐다는 뜻은 아닙니다.
보정·완화 산술의 일치가 원본 추정이나 보정 자세의 정확성을 증명하지는 않습니다.
실제 좌우 회전량의 정답, solvePnP 깊이·재투영 오차, 실제 참여도 정확도는 현재 코드만으로 확인할 수 없습니다.
A/B/C ambiguous 보류·이탈 유예 비교는 분석 실험이며 Gaussian 기본 정책으로 구현하지 않았습니다.
모든 모델은 단일 얼굴·임시 보정 기준이며 실제 참여도 정확도나 논문 성능을 달성했다고 주장하지 않습니다.

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
