# 웹캠 실행과 좌우 진단

[모델 개요](../README.md) · [수식](model-formulas.md) · [실험 해석](experiments.md) · [과거 기록](implementation-history.md)

## 실행 환경과 기본 조작

### 독립 웹캠 자세 경로 선택

기본값 `--pose-source pnp`와 기존 gaze·yaw-log·디버그 페이지 기능은 유지합니다.
`--pose-source face-transform`은 같은 얼굴 검출의 변환행렬을 별도 스무딩·Euler 보정에 전달하고,
같은 설정의 Gaussian을 주 화면·진단에 표시합니다. PnP는 비교 기준으로 계속 계산합니다.
`--validated-pose`는 비교 PnP에 양의 깊이·재투영 오차·연속성 검사를 적용합니다.
실패 관측은 null이며 다른 경로 점수로 채우지 않습니다. 프런트나 다인 runtime에는 주입하지 않습니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --pose-source face-transform --validated-pose --yaw-diagnostic --diagnostic-down --front-stability --single-model --diagnostic-move-seconds 2 --diagnostic-hold-seconds 4
```

수동 C/Space 보정 후 정면 고정 → 정면을 유지한 작은 몸 이동 → 정면 → 좌우·복귀 → 아래 보기·복귀를 안내합니다.
전체 구간 수는 동적으로 표시합니다. 실제 자세가 안내 구간과 일치한다는 보장은 없습니다.
아래 보기의 상대 Pitch가 +15도 이상일 때 looking_down, +10도 초과는 기존 하향 전이 보류입니다.
Gaussian 식·sigma·10초 집계·히스테리시스를 경로 선택 때문에 변경하지 않습니다.

`FACE_TRANSFORM_COMPARE`는 같은 프레임의 Raw·스무딩·Relative 각도, 보정 기준, 상태·현재/집계 점수를 출력합니다.
종료 요약에는 구간별 각도 평균·표준편차·최대–최소 폭, null/하향 수와 첫 정면 복귀가 포함됩니다.
MediaPipe의 canonical→runtime 행렬에서 스케일을 SVD로 분리하고 `diag(1,-1,-1)`로 기존 카메라 축을 맞춥니다.
행렬 결측·반사·퇴화는 보류합니다. 실제 각도 정답 없이 좌우 정확도를 검증했다고 표현하지 않습니다.
`--calibration-method matrix`는 별도의 이전 실험 보정이며 기본값은 euler입니다. 두 행렬 기능은 서로 다릅니다.

기본 진단 수치는 메모리와 터미널 전용입니다. 기존 `--yaw-log`를 함께 명시하면 main의 집계 저장 기능을 사용합니다.
원본 미디어·얼굴/랜드마크·개인 ID·회전/이동 벡터는 저장 요약에 추가하지 않습니다.
20도 정면 후보 진단은 현재 정면보다 넓고 이탈 임계값보다 작은 경우만 실행하며, 아니면 후보만 비활성화합니다.

기존 gaze의 하향 정책을 보존하기 위해 `--gaze`와 검증 PnP/변환행렬/실험 행렬 보정의 동시 사용은 거부합니다.
gaze 단독·yaw-log 단독과 live/debug_page 호출은 계속 지원합니다.
MediaPipe 지원 환경과 통합 검증 범위는 [통합 검토](integration-review.md)를 참고하세요.

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

영상 오른쪽의 현재 점수·10초 집계·머리 각도·시선 후보·보정 상태는 고정된 위치에 표시됩니다. 상세 진단은 `D`로 켜고 `[` / `]`로 페이지를 넘깁니다. `--gaze`가 없거나 관측할 수 없는 값은 `--` 또는 `OFF`로 표시하며, 현재 점수와 10초 집계는 서로 다릅니다.

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

결과를 파일로도 남기려면 다음처럼 `--yaw-log`를 명시합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --yaw-diagnostic --yaw-log --single-model --diagnostic-move-seconds 3 --diagnostic-hold-seconds 5
```

진단 완료·중단 또는 재보정 때 `engagement_model/diagnostic_logs/`에 각 시행별 JSON을 만듭니다. 단계별 요약, 1초 단위 원본·스무딩·상대각·모델 입력 Yaw의 평균·표준편차·최소·최대, 관측 수와 추적 변경 횟수를 기록합니다. 세부 프레임, 원본 영상·음성·랜드마크, 얼굴 ID와 세션 ID는 기록하지 않습니다. 파일에는 개인의 머리 방향 통계가 있으므로 동의한 테스트에만 사용하고, 필요가 끝나면 직접 삭제하세요. 저장 경로는 Git에서 제외됩니다. `--yaw-log` 없이 실행하면 기존처럼 파일을 남기지 않습니다.

## 기록 범위와 집계

`GuidedYawDiagnostic`의 메모리에 최대 10,000개 숫자 관측을 보관합니다.

- 관측 시각, 안내 구간, 이동/유지 구분
- raw/smoothed/relative/used Pitch·Yaw·Roll, 최종 baseline
- 관측 품질과 사유, 단일 얼굴 여부, 추적 가능·변경 상태, 입력 출처
- Gaussian 현재 점수, 방향 상태와 하향 보류 상태
- 실제 채택된 보정 표본의 분포

used의 Roll은 null입니다. Gaussian 점수는 Roll을 사용하지 않습니다.
영구 식별자나 임시 추적 키 자체를 구간 기록에 추가하지 않습니다.
기본 실행은 원본 영상·얼굴 이미지·랜드마크나 로그/세션 JSON 파일을 저장하지 않습니다. `--yaw-log`는 위에 설명한 집계 통계만 별도 저장합니다.
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

`--yaw-log`를 사용하면 `timeline`의 초별 `pose_input`에 기존 6개 기준점을 사용한
재투영 평균 오차(px), 눈 사이 거리 대비 오차 비율, 눈 사이 거리(px), 직전 프레임 대비
얼굴 중심과 크기를 정규화한 기준점 평균 변화량을 집계합니다. 좌표·영상·회전/이동 벡터는 저장하지 않습니다.
눈 사이 거리는 얼굴 전체 너비가 아니라 크기 비교용 근사치입니다. 기준점 변화에는 실제 고개 회전도 포함되며,
샘플 간격이 일정하지 않아 값 자체가 속도나 인식 오류를 의미하지 않습니다. 직전 단일 얼굴이 없으면 변화량은 null입니다.
재투영 오차가 작아도 실제 각도가 정확하다는 뜻은 아니며, 카메라 초점거리·3D 모형이 근사치인 점에 유의합니다.
각 수치는 임계값 없이 진단용으로만 쓰이며 세션 계약이나 점수에는 반영되지 않습니다.
오른쪽에만 다른 sigma를 주거나 좌우 측정값의 중간으로 baseline을 임의 교체하지 않습니다.
이 문서는 각도 추정 문제를 해결했다거나 참여도 정확도를 검증했다고 주장하지 않습니다.
