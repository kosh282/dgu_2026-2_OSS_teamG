# 실험 조건과 검증 기록의 해석

[현재 구현](../README.md) · [상세 수식](model-formulas.md) ·
[웹캠 진단](webcam-diagnostics.md) · [정리 전 README 보관본](implementation-history.md)

현재 실행 가능한 코드, 합성 입력 비교, 실제 웹캠 관찰, 과거 기록을 구분합니다.
문서만 정리하는 이번 작업에서 전체 테스트나 실제 웹캠 측정을 새로 수행하지 않았습니다.
과거 통과 개수는 현재 실행 결과로 다시 표기하지 않습니다.

## 1. 현재 실행 모드와 과거 버전

| 구분 | 현재 코드에서의 의미 | 재현 시 주의 |
|---|---|---|
| Gaussian 0.1 | 보관 문서의 과거 개발 단계 | 별도 실행 옵션이 없으므로 현재 결과라고 표시하지 않음 |
| Gaussian relative 0.2 | 완화 전 상대각을 점수에 사용 | 현재 하향 전이 보류와 현재/집계 분리를 유지하는 비교 모드 |
| Gaussian softened 0.3 | 기본 모드, Pitch/Yaw 중립 완화 | sigma는 15/20도, 하향 정책과 시간 집계는 유지 |
| Huang 온라인 0.2 | 현재 선택 가능한 Huang 온라인 변형 | 원식 전체 영상 재현이나 실제 집중도 확률이 아님 |
| Legacy | 기본 `EngagementRuntime()` | 프런트·팀원 웹캠 기본, 상태·시간 규칙으로 점수화 |

현재 relative 모드는 과거 모든 시점의 Gaussian을 그대로 재현하는 옵션이 아닙니다.
모델 버전·설정 버전·publisher 확장 버전은 서로 다른 항목입니다.
설정 버전은 호출자가 덮어쓸 수 있으므로 기록된 각도 모드·실제 설정도 함께 확인합니다.
상세 버전 표와 수식은 [설계 문서](model-formulas.md)에 있습니다.

```powershell
# 같은 현재 전처리/보류 구조에서 점수 입력의 완화 여부 비교
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --single-model --gaussian-angle-mode relative
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.webcam_demo --single-model --gaussian-angle-mode softened
```

## 2. 합성 입력 비교

`tests/test_huang.py::comparison_report()`는 Legacy·Huang 및 선택적으로 Gaussian에 같은
시점별 `FeatureWindow`를 전달합니다. 모델마다 시간 상태는 독립적으로 관리합니다.
준비 구간 뒤 평가 구간의 유효 점수 평균과 보류 비율을 출력합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m engagement_model.tests.test_huang --compare-all
```

이 명령은 현재 `ModelConfig.gaussian()`을 사용하므로 기본 결과는 softened 0.3입니다.
보관본의 상대 Pitch -2도 점수 99.10 등 완화 전 결과를 그대로 기대하면 안 됩니다.
현재 softened 모드에서는 ±2도 이내 입력이 중립으로 완화됩니다.

현재 회귀 테스트에서 비교하는 대표 수치는 다음과 같습니다.

| 합성 조건 | Legacy | relative Gaussian 현재 | softened Gaussian 현재 |
|---|---:|---:|---:|
| 정면 10초 뒤 Yaw +10도 첫 관측 | 100 | 88.2 | 95.6 |
| 큰 방향 이탈 뒤 정면 복귀 첫 관측 | 90 | 100 | 100 |

근거는 `tests/test_gaussian.py::test_legacy_reference_and_improvement_on_same_sequence()`입니다.
큰 이탈의 softened 현재 점수 2.4(Yaw 60도)와 발행 집계 26.8(앞 5초 정면·뒤 5초 Yaw 60도)도
같은 파일에 기대값으로 정의되어 있습니다. 현재 점수와 시간 집계의 수치를 혼동하지 마세요.
이 값은 구현·계산의 회귀 기준이지 실제 영상 정확도나 사람 주석과의 일치율이 아닙니다.

보관본에는 다른 버전의 작은 움직임·하향 복귀·sigma 후보 비교 표도 있습니다.
버전·입력·준비 시간·평가 시간·유효 구간이 같은지 확인한 뒤 비교해야 합니다.
누락값은 0점으로 바꾸지 않고, 점수 평균과 함께 보류율을 보고합니다.

## 3. A/B/C 보류·유예 분석의 상태

대화 중 다음 정책을 같은 합성 관측으로 비교한 분석이 있었습니다.

- A: 현재 Gaussian의 즉시 점수.
- B: ambiguous 상태를 추가 보류하고 집계에서 제외.
- C: B에 짧은 away 유예를 추가하고 그 구간도 집계에서 제외.

B/C는 현재 런타임이나 CLI에 구현된 선택 옵션이 아닙니다.
현재 Gaussian은 ambiguous의 유효 입력을 점수화하며 Legacy away 3초 유예를 사용하지 않습니다.
표시용 과거 점수를 현재 관측으로 간주하거나 집계에 삽입하지 않는다는 조건이 중요합니다.
보류·유예로 부드러워 보이는 것과 실제 이탈을 놓치지 않는 것은 별도 평가 대상입니다.

## 4. 실제 웹캠 관찰과 제한

이전 진단에서 다음 현상이 관찰됐습니다.

- 같은 보정 기준에서 좌우 상대 Yaw가 약 -32도와 +80도로 다르게 출력된 화면이 있었습니다.
- 회전 구간에서 Pitch/Roll도 크게 변하고 안내상 유지 구간의 표준편차가 큰 실행이 있었습니다.
- 보정·완화 산술 차이는 0이면서 정면 복귀 점수는 회복되는 실행도 있었습니다.
- 보정 당시의 안정성, 최초 정면의 재현성, 좌우 안내와 실제 움직임 일치는 별개입니다.

이 수치들은 사람의 실제 좌우 회전량을 측정한 정답 데이터가 아닙니다.
완료 플래그나 유효 얼굴 표본의 존재를 안정적인 자세·정확한 머리 각도의 증거로 쓰지 않습니다.
원본 추정, Euler 표현, 스무딩 이력, 보정 자세, 안내 타이밍 중 원인을 아직 확정하지 못했습니다.
기존 solvePnP 합성 진단에서 확인한 가능성을 실제 웹캠 오류의 확정 원인으로 승격하지 않습니다.
영상 담당자 수정과 추가 회전 품질 자료가 필요한지 실제 관측으로 판단해야 합니다.

자동 구간의 원시 수치·영상은 파일에 저장하지 않습니다.
문서의 과거 관찰 요약만으로 원본 기록 전체를 재생하거나 재검증할 수는 없습니다.
재현 절차와 출력 필드는 [진단 문서](webcam-diagnostics.md)를 참고하세요.

## 5. 테스트 범위와 결과 보고

| 파일 | 코드로 확인한 검증 범위 |
|---|---|
| `tests/test_model.py` | 설정·어댑터·Legacy 점수 규칙 |
| `tests/test_runtime.py` | 연속 관측·상태 시간·유예·하향·공백·초기화 |
| `tests/test_huang.py` | 원식 손계산·온라인 변형·복귀·계약·웹캠 모의 검사 |
| `tests/test_gaussian.py` | Gaussian 대칭·민감도·시간 적분·하향 전이·기존 모델 회귀·JSON 왕복 |
| `tests/test_webcam_recovery.py` | 합성 파이프라인 복귀·보정 유지·시각화·진단 스냅샷·입력 요청 |

```powershell
video_recognition/.venv/Scripts/python.exe -B -m unittest discover -s engagement_model/tests -v
```

테스트 코드가 있다는 사실과 해당 환경에서 방금 통과했다는 사실은 구분합니다.
카메라·GUI 모의 검사는 실제 장치의 포커스·재검출·장시간 안정성을 보장하지 않습니다.
자동 구간 안내의 구현은 `webcam_demo.py::GuidedYawDiagnostic`이며 별도 영구 테스트 파일은 없습니다.
이전 세션의 메모리 검증을 현재 저장소 테스트 개수에 더하지 않습니다.

새 결과를 보고할 때는 명령, 모델·설정, 입력 종류, 시간 간격, 준비/평가 구간,
실행 환경, 실제 통과·실패·건너뜀, 보류율과 미검증 범위를 함께 기록합니다.
사람 주석이 없다면 MAE·참여도 정확도·논문 성능 재현을 달성했다고 보고하지 않습니다.

## 6. 원문 이력 보존

[implementation-history.md](implementation-history.md)는 정리 전 README의 상세 설명과
버전별 수정·실험 표를 보존합니다. 이 문서 작업에서 과거 수치를 다시 측정하지 않았습니다.
오래된 테스트 수·미구현 목록·provenance를 최신 구현 설명으로 복사하지 않습니다.
현재 실행 안내는 상위 README와 실제 코드가 기준입니다.
