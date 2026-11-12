# 웹캠 실행과 좌우 진단

[모델 개요](../README.md) · [수식](model-formulas.md) · [실험 해석](experiments.md) · [과거 기록](implementation-history.md)

## 실행 환경과 기본 조작

### MediaPipe 얼굴 변환행렬과 동일 프레임 비교

`--pose-source face-transform`은 MediaPipe를 주 표시·정면 보정 안내·진단 기준으로 선택합니다.
PnP도 비교 기준으로 계속 실행하며 `PnP reference`로 표시합니다. 행렬 결측 시 PnP 점수로 대체하지 않습니다.
`--pose-source pnp`로 기존 주 화면에 되돌릴 수 있습니다. 옵션 생략 기본값은 아직 PnP입니다.
기본값 전환은 실제 충분한 아래 보기·복귀·동일 회전량 검증 후 별도로 결정합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --pose-source face-transform --validated-pose --yaw-diagnostic --diagnostic-down --front-stability --single-model --diagnostic-move-seconds 2 --diagnostic-hold-seconds 4
```

Gaussian 식·sigma·10초 집계·보정 방식·아래 보기 정책은 경로 선택으로 변경하지 않습니다.
아래 보기 안내 구간에서 상대 Pitch가 +15도 이상이어야 looking_down 진입을 검증할 수 있습니다.
10~15도만 관측됐다면 하향 전이 null만 검증한 것으로 보고합니다.

정면 유지 안정성은 `--front-stability`를 추가해 정면 고정 → 정면을 유지한 작은 몸 이동
→ 정면 복귀 구간을 먼저 측정합니다. 나머지 좌우·하향 안내도 유지합니다.
안내의 전체 구간 수는 실제 단계 수(이 옵션과 아래 보기 사용 시 9)에 맞춰 표시합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --compare-face-transform --validated-pose --yaw-diagnostic --diagnostic-down --front-stability --single-model --diagnostic-move-seconds 2 --diagnostic-hold-seconds 4
```

같은 유지 구간의 Raw·스무딩·Relative Pitch/Yaw 평균·표준편차·최대–최소 폭을 출력합니다.
각도 범위는 원형 평균을 중심으로 펼쳐 ±180도 표현 경계가 가짜 360도 변화가 되지 않게 합니다.
`baseline_changes`는 그 구간에서 관측한 유효 기준값 변경 횟수입니다.
`landmark_motion`은 프레임 간 중심 이동량과 중심·스케일을 제거한 2D 모양 RMS입니다.
이는 추적 변화의 참고 수치이며 실제 머리 회전량이나 랜드마크 오차의 정답이 아닙니다.
랜드마크는 메모리에서만 비교하며 로그에는 집계 수치만 출력합니다.
두 추정기의 동시 변화만으로 실제 움직임을 확정하지 않으며, 보정 기준이 고정돼도
정면 안내 구간과 실제 자세가 달라질 수 있습니다. 확인되지 않은 오차에 점수 완화를 적용하지 않습니다.

설치된 MediaPipe 0.10.35의 `output_facial_transformation_matrixes=True`를 사용합니다.
기존 `FaceAnalyzer`의 검출기를 해당 옵션으로 생성하고, 반환된 한 결과를 기존 특징 추출과
행렬 비교에 함께 사용합니다. 이 선택은 독립 웹캠에서만 적용하며 팀원 파일은 수정하지 않습니다.
기본 실행에서는 이 검출기 교체나 비교 계산을 하지 않습니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --compare-face-transform --validated-pose --yaw-diagnostic --diagnostic-down --single-model --diagnostic-move-seconds 3 --diagnostic-hold-seconds 3
```

주 화면은 기존 PnP이고 `Face transform current`가 비교 점수입니다.
양쪽은 같은 프레임·랜드마크 검출, 설정값, 스무딩, 보정 시작 시각과 아래 보기 정책을 사용합니다.
각자의 원본 각도에 기준값을 독립 수집하며, 한 경로의 보정 실패를 다른 경로로 대체하지 않습니다.
`--validated-pose`를 생략하면 원래 iterative PnP와 비교합니다.
`--compare-face-transform`은 Gaussian+Euler 보정에서만 지원합니다.

MediaPipe canonical→runtime 4×4 행렬에서 SVD로 회전과 균일 스케일을 분리하고,
MediaPipe의 x 오른쪽/y 위/카메라 -z 좌표를 기존 카메라 축으로 `A=diag(1,-1,-1)` 변환합니다.
이 좌표 변환 후 기존 Euler 보정을 적용합니다. 상대 회전행렬 Calibration을 사용하는 기능은 아닙니다.
행렬 누락·반사·퇴화·비유한 값은 보류합니다. 특잇값 비 1.1 초과를 거부하는 기준은 프로젝트 실험값입니다.

`FACE_TRANSFORM_COMPARE`는 같은 프레임의 원본·상대각·현재/집계 점수·상태를 출력합니다.
종료 시 `FACE_TRANSFORM_SUMMARY`에 안내된 유지 구간별 Pitch/Yaw 평균·표준편차,
현재 점수 평균·null 수·looking_down 수를 출력합니다. 원본 미디어나 좌표·수치 파일은 저장하지 않습니다.
실제 회전량 정답은 없으므로 좌우 편차 감소·정면 안정성은 비교할 수 있어도 각도 정확도를 확정할 수 없습니다.
안내와 실제 자세가 일치하지 않거나 정면 복귀가 기준에서 벗어난 회차는 대칭성 평가에서 제외합니다.
공식 자료: [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker),
[Face Geometry 좌표계](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md).

### unknown·점수 보류 원인 집계

행동 상태에 `unknown`은 없습니다. `activity uncertain`은 아래 보기 의미를 알 수 없다는
표시이고 실제 상태·현재 점수 보류·10초 집계 보류와 구분합니다.
웹캠 종료 시 `Uncertainty comparison`에서 상태, 현재/집계 null, 원인별 횟수·비율을 출력합니다.
C 재보정 시 집계를 새로 시작하며 보정 수집 구간도 분모에 포함합니다.
`null_cause`는 우선순위로 분리한 상호배타적 현재 점수 결측 원인이고,
`reason`은 중복 가능한 사유이므로 합계가 100%를 넘을 수 있습니다. 모든 비율은 프레임 기준입니다.

기본 분류에서 완화 후 Yaw 15~30도 사이, 위쪽 Pitch -20~-10도 사이 등은 `ambiguous`입니다.
정확한 경계는 `scoring.classify_orientation()`의 조건 순서를 따릅니다.
기본 Yaw 완화에서는 원래 상대 Yaw 약 20.25~35.25도가 그 사이 구간입니다.
Gaussian은 유효한 ambiguous 관측도 점수화합니다. 따라서 ambiguous 비율과 null 비율은 다릅니다.
Pitch >10도의 하향 전이와 아래 보기 히스테리시스는 별도의 보류 정책입니다.

같은 입력에 정면 Yaw 20도 후보를 메모리에서만 계산해 기본 15도와 비교합니다.
20도 후보가 현재 정면보다 넓고 이탈 임계값보다 작은 경우에만 실행합니다.
조건을 만족하지 않으면 후보만 비활성화하고 `enabled=false`와 `disabled_reason`을 출력합니다.
예를 들어 이탈 임계값이 20도이면 기본 모델은 그대로 실행하고 20도 후보는 생략합니다.
이는 범위 완화의 실험 후보이며 기본값·Legacy·Huang·JSON·sigma·하향 정책을 변경하지 않습니다.
품질 부족·관측 실패를 정면으로 대체하지 않습니다. 사람 주석 없이 후보를 최적값으로 확정하지 않습니다.

### 양의 깊이 자세 추정 실험

좌우와 아래 보기를 한 회차에서 안내하려면 다음 명령을 사용합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --validated-pose --yaw-diagnostic --diagnostic-down --single-model --diagnostic-move-seconds 3 --diagnostic-hold-seconds 5
```

수동 보정 후 정면 → 왼쪽 → 정면 → 오른쪽 → 정면 → 아래 보기 → 정면을 안내합니다.
종료 요약은 Raw·스무딩·상대·사용 각도의 평균/편차, 상태별 관측 수와 현재 점수 null 횟수를 구분합니다.
유지 안내는 실제로 그 자세를 유지했다는 증명이 아닙니다. 좌우 회전량 정답이 없으면 대칭 정확도를 보고하지 않습니다.
첫 정면 구간이 보정 자세와 크게 다르거나 좌우가 정면의 같은 쪽이면 좌우 비교 조건을 충족하지 못한 것입니다.
보정 기준을 바꾸는 시험은 한 회차 완료 후 C로 재보정해 별도 요약끼리 비교합니다.
동일 회차 중 임의로 기준을 바꾸지 않습니다. 수집한 수치는 메모리에만 유지합니다.

검증 경로는 SQPnP와 양의 깊이 iterative 후보, 각 후보의 LM 정제 결과를 픽셀 재투영 RMSE로 비교합니다.
음의 깊이 후보는 선택하지 않고 가장 작은 오차의 양의 깊이 후보를 품질·연속성 검사에 전달합니다.
이는 좌우 각도 보정 계수가 아닙니다. RMSE가 작아도 개인 얼굴 모형·카메라 근사의 각도 편향이 남을 수 있습니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --validated-pose --model gaussian --calibration-method euler --debug
```

공통 영상 파일을 변경하지 않고 `ValidatedFaceAnalyzer`가 기존 얼굴 분석기를 감쌉니다.
같은 6개 랜드마크·3D 모형·가상 카메라 행렬에 SQPnP를 적용합니다.
기존 iterative PnP가 음의 tvec 깊이 해를 반환해도 성공으로 처리하던 사례는
고정 난수의 204개 합성 입력 중 14건에서 재현됐고 새 경로에서는 0건이었습니다.
이는 실제 영상 정확도나 좌우 대칭 보장이 아닙니다.

`PoseValidationConfig`의 품질 기준은 조정 가능한 프로젝트 실험값입니다.
모든 3D 점의 깊이가 양수여야 하며, 재투영 RMSE는 영상 대각선의 2% 이하여야 합니다.
300ms 이내 연속 관측의 회전 변화 상한은 max(45도, 720도/초 × 간격)입니다.
이 기준들은 논문의 최적값이 아닙니다. 빠른 정상 회전도 보류될 수 있습니다.
실패 시 각도를 보간·고정하지 않고 `pose_unreliable`로 표시합니다.
미검출·복수 얼굴은 자세 연속성 기준을 다시 시작하며 임시 얼굴 배열을 신원으로 보지 않습니다.

같은 프레임의 기존/검증 Raw 각도, 두 해의 최소 깊이·재투영 오차, 새 회전·이동 벡터와
모델 상대각·현재 점수·집계·away 시간이 터미널에 표시됩니다. 원본 미디어와 파일은 저장하지 않습니다.
기존 Raw와 새 Raw는 같은 검출점을 쓰지만 점수 모델은 새 자세 입력을 공유합니다.
기존 자세로 계산한 점수를 동시에 재현하는 기능은 아닙니다.

타이머의 기존 규칙은 유지합니다. 0.3초 이내 짧은 자세 결측에는 점수를 보류하고 시간 상태를 유지합니다.
0.3초 초과 관측 공백·추적 키 변경·시간 역전·상태 변경은 초기화 사유입니다.
관측 부재를 지속 이탈로 확정하지 않으며 긴 실패 후 타이머 연속성은 보장하지 않습니다.
Gaussian의 아래 보기 진입 15도/해제 10도와 하향 전이 보류는 유지합니다.
Legacy는 기존 아래 보기 유예·감점을 유지하므로 아래 보기 null 정책과 다릅니다.

### Gaussian 상대 회전행렬 보정

독립 웹캠의 기본 보정은 기존 **Euler**입니다. 행렬 방식은 실제 웹캠에서
옆 보기의 하향 오분류가 해결되지 않아 `--calibration-method matrix` 실험 경로로 분리했습니다.
`rotation_calibration.py`는 Raw P/Y/R에서 `R=Rz(roll)Ry(yaw)Rx(pitch)`를 복원하고
보정 표본의 회전행렬 평균을 SVD로 정규화합니다. 상대 회전은 기준 Yaw/Roll의
좌표계 F에서 `R_relative=F.T @ R_current @ R_baseline.T @ F`로 계산합니다.
그 결과를 정준 Euler 각도로 변환합니다. 이 경로의 합성 검증은 실제 영상 해결을 뜻하지 않습니다.
Yaw는 ±90도 범위이며 특이점에서는 Roll을 0으로 고정해 같은 회전에 일관된 표현을 택합니다.
큰 회전의 Euler 특이성과 실제 solvePnP 추정 오류까지 제거하는 방법은 아닙니다.

기존 보정과 같은 3초/10표본을 요구하며, 0.98 응집도는 행렬 평균의 특잇값에 적용합니다.
동일 Euler 축별 안정성 검사와 수학적으로 같은 검사는 아니며 프로젝트 실험 정책입니다.
표현 전환이 섞인 축별 스무딩을 피하기 위해 새 경로는 Raw 회전행렬을 사용합니다.
따라서 잡음 평활 특성도 기존 scalar 1-Euro 경로와 다르며 실제 영상에서 함께 평가해야 합니다.
Gaussian 식·sigma·하향 보류 정책은 변경하지 않습니다.

```powershell
# 새 보정과 기존 보정을 같은 영상으로 계산: Current direction과 Old Euler current 비교
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --single-model --calibration-method matrix --compare-calibration
# 이전 보정만 실행
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --single-model --calibration-method euler
```

비교 모드에서 수동 보정을 시작하면 두 경로가 함께 수집합니다. 비교 화면의 Current direction은 새 행렬 보정,
Old Euler current는 기존 축별 보정으로 계산한 점수입니다. 한 경로의 보정이 미완료이면 null입니다.
Legacy/Huang에는 원본 FeatureWindow를 전달하고 Gaussian에만 입력 복사본의 상대각을 대체합니다.
팀원 영상 파일·프런트·FeatureWindow 계약을 수정하지 않으며 행렬·추가 진단값도 JSON에 저장하지 않습니다.

matrix 모드의 `relative_formula_error_max`는 상대 행렬의 Euler 변환과 실제 입력의 일치도입니다.
`baseline_minus_initial_front_yaw`는 서로 다른 Euler 표현의 평균을 빼지 않도록 null로 둡니다.
정면 복귀는 `front_return_yaw`로 확인합니다. 보정 분포의 Raw Euler 축 분산이 커도 행렬이 같을 수 있습니다.

검증 명령: `video_recognition/.venv/Scripts/python.exe -B -m unittest engagement_model.tests.test_rotation_calibration -v`.

저장소 루트에서 실행합니다. 환경은 `video_recognition/requirements-integrated.txt`를 기준으로 합니다.
독립 실행기는 기존 얼굴·상체 분석, `VisionPipeline`, 웹캠 어댑터와 렌더러를 import합니다.
팀원 영상 데모·프런트를 수정하거나 별도 시선 추정기를 실행하지 않습니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --debug
```

1. 얼굴 한 명과 고정된 카메라로 정면을 준비합니다.
2. 웹캠 창에서 CALIBRATE 버튼 또는 영문 C/Space를 한 번 누릅니다.
3. collecting 동안 약 3초 정면을 유지합니다. ready 이후 점수를 확인합니다.
4. D는 상세 진단, V는 오버레이, Q/Esc는 종료입니다.

키 반복은 억제하며 수집 중 시작 요청으로 표본을 초기화하지 않습니다.
수집 중 관측 불가·불안정이 계속되면 표본 수집이 다시 시작될 수 있으므로 진행률과 사유를 봅니다.
자동 보정이 필요한 일반 데모는 `--calibrate`를 추가합니다. 아래 좌우 진단과는 병용하지 않습니다.

기본 화면의 Relative P/Y는 완화 전 상대각이며, 디버그 USED P/Y는 실제 Gaussian 점수 입력입니다.
현재 점수와 10초 집계, 방향 상태와 집계 보류 상태를 서로 구분합니다.

## 자동 구간 안내 진단

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --yaw-diagnostic --single-model
```

수동 보정이 완료되면 다음 다섯 구간이 자동 진행됩니다.

```text
정면 → 본인 왼쪽 회전 → 정면 복귀 → 본인 오른쪽 회전 → 정면 복귀
```

구간마다 MOVE/SETTLE 2초 후 HOLD STILL 3초이며 전체 25초입니다.
LEFT/RIGHT는 영상의 좌우가 아니라 본인의 좌우입니다. 좌우를 비슷한 크기로 돌립니다.
화면 카운트다운을 따르고 유지 구간 중에는 자세를 유지합니다. 구간 사이에 재보정하지 않습니다.

더 긴 이동·유지 시간은 다음과 같이 선택합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --yaw-diagnostic --single-model --diagnostic-move-seconds 3 --diagnostic-hold-seconds 5
```

이 설정은 총 40초입니다. 안내 시간은 실험 편의 설정이며 모델 판정 임계값을 바꾸지 않습니다.
진단 종료 시 결과를 터미널에 자동 출력합니다. 별도의 결과 입력 명령은 없습니다.
API 실행 환경에서 시작했다면 결과는 IDE 터미널이 아니라 해당 실행 프로세스 출력에 나타날 수 있습니다.

## 기록 범위와 집계

`GuidedYawDiagnostic`의 메모리에 최대 10,000개 숫자 관측을 보관합니다.

- 관측 시각, 안내 구간, 이동/유지 구분
- raw/smoothed/relative/used Pitch·Yaw·Roll, 최종 baseline
- 관측 품질과 사유, 단일 얼굴 여부, 추적 가능·변경 상태, 입력 출처
- Gaussian 현재 점수, 방향 상태와 하향 보류 상태
- 실제 채택된 보정 표본의 분포

used의 Roll은 null입니다. Gaussian 점수는 Roll을 사용하지 않습니다.
영구 식별자나 임시 추적 키 자체를 구간 기록에 추가하지 않습니다.
원본 영상·얼굴 이미지·랜드마크를 기록하거나 로그/세션 JSON 파일을 저장하지 않습니다.
재보정은 이전 진단을 중단·요약한 뒤 기록을 새로 시작합니다.
보정 무효화·기준 변경이나 조기 종료는 완료와 구분해 출력합니다.

안내상 유지 구간 중 유효/부분 관측만 각도 평균에 포함합니다.
각도 평균은 원형평균이며 표준편차는 평균에 대한 최단 각도 차이로 계산합니다.
이동 구간은 유지 평균에서 제외합니다. 점수 평균은 실제 현재 점수가 있는 표본만 사용합니다.
값이 없으면 null로 남기고 표본 수·점수 유효 비율도 함께 출력합니다.
메모리 상한에 도달하면 `truncated=true`로 보고합니다.

## 요약 해석

| 필드 | 의미 |
|---|---|
| `calibration_distribution` | 실제 보정 표본의 축별 평균·편차·최솟값·최댓값 |
| `yaw_comparison` | 단계별 Yaw 평균 및 최초 정면을 기준으로 한 좌우 편차 |
| `front_return_yaw` | 정면/복귀의 상대 Yaw와 0도 대비 평균 절대 오차 |
| `score_mean_difference` | 좌우 유효 점수 평균의 절댓값 차이 |
| `relative_formula_error_max` | 상대각과 wrap(입력-baseline)의 최대 산술 차이 |
| `used_formula_error_max` | 실제 점수 입력과 설정된 완화 함수 결과의 최대 차이 |
| `baseline_minus_initial_front_yaw` | 보정 기준과 최초 정면 유지 구간의 스무딩 Yaw 차이 |

산술 오차 0은 계산이 구현과 일치한다는 뜻이며, 올바른 자세로 보정됐다는 증명이 아닙니다.
raw/smoothed의 0도가 실제 정면이라는 보장도 없습니다. 최초 정면 대비 편차로 비교합니다.
`hold`는 유지 안내 구간이지 실제 안정성 검출 결과가 아닙니다.
완료·유효 표본 존재 플래그도 물리적 각도 정확성을 보장하지 않습니다.
표준편차가 크거나 좌우 결측 비율이 다르면 점수 평균 차이만으로 모델을 평가하지 않습니다.

## 급락 디버그

D 모드의 `GaussianDiagnostics`는 현재 입력과 같은 입력의 sigmaYaw 20/30/40 후보 점수를 표시합니다.
후보 계산은 실제 설정·점수를 변경하지 않습니다. 축별 제곱 정규화 항으로 감쇠 기여도를 봅니다.
최근 5초의 점수 감소와 최대 8초 유지되는 `HISTORICAL DROP`은 과거 관측으로 표시합니다.
미검출·하향 보류·관측 공백·추적 변경 때 연속 변화량 비교를 끊고 재보정 시 기록을 지웁니다.

## 확인된 현상과 남은 검증

실제 웹캠에서 좌우 상대 Yaw가 크게 달랐고, 회전 시 Pitch/Roll도 크게 변하는 사례가 있었습니다.
자동 진단에는 원본 각도와 유지 구간의 큰 분산, 최초 정면과 보정 기준의 차이가 나타났습니다.
한편 정면 복귀 점수가 회복되고 보정·완화 산술 오차가 0인 실행도 확인했습니다.
이 기록만으로 실제 좌우 회전량, solvePnP 오류, 안내와 자세의 일치 여부를 확정하지 못합니다.

현재 입력에는 rvec/tvec·깊이·재투영 오차가 없습니다.
이 정보와 회전 연속성을 확인하려면 비전 담당자와 진단 연결을 협의해야 합니다.
오른쪽에만 다른 sigma를 주거나 좌우 측정값의 중간으로 baseline을 임의 교체하지 않습니다.
이 문서는 각도 추정 문제를 해결했다거나 참여도 정확도를 검증했다고 주장하지 않습니다.

## 보정 기준 ±20도 재현 검사

```powershell
video_recognition/.venv/Scripts/python.exe -B -m unittest engagement_model.tests.test_yaw_calibration -v
```

`tests/test_yaw_calibration.py`는 실제 보정기·모델을 호출해 다음을 분리합니다.

| 합성 입력 조건 | 확인 내용 |
|---|---|
| 같은 Raw 종점 -40/+40도, 기준 -20도 | Relative -20/+60도: Gaussian 점수 약 76.2/2.4 |
| 같은 Raw 종점 -40/+40도, 기준 +20도 | Relative -60/+20도: Gaussian 점수 약 2.4/76.2 |
| 기준과 종점을 같은 양만큼 이동 | 상대각·점수는 동일; 기준 부호 자체가 Gaussian 비대칭을 만들지 않음 |
| 같은 Pitch·±동일 Relative Yaw | relative/softened 모드 모두 대칭 |
| Raw와 smoothed가 다른 보정 표본 | smoothed가 있으면 baseline과 상대각은 smoothed 기준 |
| 이상적 6점 투영 → 실제 head_angles → 보정 → Gaussian | 잡음 없는 고정 합성 좌표계의 ±20도 기준에서 대칭 |
| 같은 회전행렬의 두 Euler 표현 | 현재 축별 차이 보정의 표현 의존성 재현 |

첫 두 행은 사용자가 보고한 기준에 따른 점수 차이의 방향을 수학적으로 재현합니다.
같은 절대 Raw 종점을 썼으므로 물리적으로 같은 크기의 좌우 회전을 재현한 것은 아닙니다.
기준을 바꾸면서 종점은 그대로 둔 경우와, 정면 기준에서 실제로 동일하게 회전한 경우를 구분해야 합니다.

### 코드에서 확인한 한계

고정 합성 좌표계의 Euler P/Y/R=(180,30,0)과 (0,150,180)은 같은 회전행렬을 표현합니다.
현재 `head_calibration.py`의 축별 최단 각도 차이는 이를 Pitch=-180도, Yaw=120도로 계산합니다.
이 상대각을 Gaussian에 넣으면 같은 회전인데도 낮은 점수가 나옵니다.
이는 축별 Euler 보정의 표현 의존성을 재현한 검사이며, 해결됐음을 검증하는 검사가 아닙니다.
실제 웹캠에서 해당 표현 전환이 발생했는지는 회전행렬·rvec/tvec 자료 없이는 확정할 수 없습니다.
이상적 투영 테스트가 실제 얼굴 모형 오차·가림·잡음·재검출을 검증하지도 않습니다.

### 다음 실제 측정과 수정 판단

1. 자동 구간 이름을 실제 자세의 정답으로 사용하지 말고, 정면 유지 중 Raw/smoothed와 baseline부터 확인합니다.
2. 보정 때와 복귀 때 정면 자세·카메라가 같았는지 확인하고 각도와 Roll의 변화도 비교합니다.
3. 실제 Raw 회전 자료에서 표현 전환이 확인되면 비전 담당자와 상대 회전행렬 기반 보정을 검증합니다.
4. 추정 회전행렬 자체가 불안정하면 `head_angles()`의 깊이·재투영 오차·추정 연속성부터 점검합니다.

상대 회전행렬만으로 잘못된 solvePnP 추정이 고쳐지는 것은 아닙니다.
실제 원인 확인 전에는 좌우별 보정값, 중간값 baseline 교체, sigma 확대를 적용하지 않습니다.
이 검사 추가는 Gaussian 식·기본값·상태·하향 정책이나 팀원 영상 코드를 변경하지 않습니다.
# 현재 점수와 집계 점수 구분 진단

행렬 보정의 상대각은 카메라 축에서 직접 추출하지 않습니다.
`R0 = Rz(r0) Ry(y0) Rx(p0)`, `F = Rz(r0) Ry(y0)`로 정의하고
`R_relative = F.T @ R_current @ R0.T @ F`에서 Euler 각도를 추출합니다.
정준 기준 자세에서 Pitch 증가가 아래 보기로 전달되도록 Yaw/Roll 기준 축을 맞춥니다.
단순 `R_current @ R0.T`는 기준 Roll이 약 180도이면 Pitch 부호가 반전될 수 있습니다.
기준과 같은 실제 회전의 동치 Euler 표현은 같은 행렬과 같은 결과를 만듭니다.
카메라 고정 축의 회전과 이 보정 좌표계의 회전은 구분해야 합니다.

상대 Euler는 Rz-Ry-Rx 순서이며 Yaw는 ±90도 정준 범위입니다.
Yaw 특이점에서는 Pitch/Roll을 독립적으로 식별할 수 없습니다.
작은 정상 상대 회전에 대해 큰 Pitch/Roll이 생기지 않는 것은 합성 테스트로 확인하지만,
실제 원본 회전행렬 자체가 뒤집히면 이 변환으로 복구할 수 없습니다.
행렬이 다른 원본 자세를 단순 Euler 표현 전환으로 간주하지 마세요.
기존 경로는 스무딩, 새 경로는 raw 회전을 사용하므로 순간값의 차이가 전부 보정 차이는 아닙니다.

실제 확인: `--calibration-method matrix --compare-calibration --single-model`로 실행하고 C 보정 후
아래 보기 유지 → 정면 → 같은 크기 좌/우 회전을 진행합니다.
아래 보기에서는 새 경로도 양의 Pitch와 보류가 나오는지 확인합니다.
새 경로에 큰 Pitch/Roll이 남으면 D의 원본 각도와 비교하고 영상 담당자와
원본 solvePnP 회전·재투영 오차를 추가 확인해야 합니다. 실제 웹캠 개선은 재검증 전입니다.

`--calibration-method matrix --compare-calibration --single-model` 실행에서는 디버그 표시 여부와 관계없이
2초마다 행렬 보정/기존 Euler 보정의 `current_direction_score`, `score`,
상대각(`source_angles`), 실제 점수 입력(`gaussian_observation`), 행동·하향 상태와
집계 사유를 터미널에 출력합니다. 두 보정 경로는 같은 영상 관측을 사용합니다.
기록은 최근 최대 6,000개 관측을 메모리에만 보관하며 재보정 시 비웁니다.
종료 시 보관된 관측의 점수 범위·null 비율·하향 상태별 관측 수를 출력합니다.
이는 프레임 기준 진단이며 실제 행동 정확도나 시간 비율이 아닙니다.

정면 보정 후 정면 5초 → 아래 보기 유지 5초 → 정면 5초를 시험합니다.
아래 보기에서 현재 점수까지 높다면 상대 Pitch와 하향 상태를 확인합니다.
현재 점수는 null이고 집계만 높다면 두 점수의 의미를 구분해야 합니다.
현재 정책은 아래 보기를 감점으로 확정하지 않고 보류하므로, 아래 보기만으로
점수가 반드시 감소해야 한다는 기준으로 평가하지 않습니다.
이 진단 추가는 산정식·sigma·하향 정책·JSON 이벤트 계약을 변경하지 않습니다.
