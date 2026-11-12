# 웹캠 실행과 좌우 진단

[모델 개요](../README.md) · [수식](model-formulas.md) · [실험 해석](experiments.md) · [과거 기록](implementation-history.md)

## 실행 환경과 기본 조작

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
