# 참여도 모델 — 1단계 구현

## 1. 개요

개인별 정면 기준에 대한 머리 방향 차이와 자세 유지 시간을 이용해 0~100점의 행동 점수를 계산하는 실험용 모듈입니다.

**점수는 실제 집중도나 이해도의 직접 측정값이 아닙니다. 모든 각도·시간·점수 계수는 검증 전 실험 후보입니다.**

[video_recognition](../video_recognition/README.md)이 얼굴 검출과 정면 보정을 담당하고, 이 모듈은 그 결과를 검증하고 점수화합니다. 현재는 단일 학생용 함수 단계이며 영상 파이프라인, integration, frontend에는 연결되지 않았습니다.

- 입력: VisionPipeline의 시점별 `FeatureWindow` 사전과 호출자가 계산한 유지 시간·이전 안정 점수
- 내부 자료: `Observation`과 정면 기준 상대 각도 `HeadAngles`
- 출력: `behavior_score()`의 0~100 숫자 또는 판단 보류·관측 불가를 나타내는 `None`
- EAR, 깜빡임, blendshape, 상체 위치는 현재 점수 계산에 사용하지 않습니다.

## 2. 전체 처리 흐름

```text
video_recognition: FeatureWindow 생성
  ↓ 호출자가 parse_feature_window()에 전달
input_adapter.py: 형식·얼굴 개수·보정·품질·상대 각도 검사
  ↓ Observation 반환
scoring.py: classify_orientation()으로 자세 후보 분류
  ↓ 정면 / 이탈 / 아래 보기 / 불명확 / 관측 불가
scoring.py: behavior_score()로 점수 계산
  └─ 이탈 상태에서는 orientation_severity()로 각도 강도 계산
  ↓
0~100점 또는 None 반환
```

각 단계는 `config.py`의 설정을 사용합니다. 위 흐름을 자동 실행하는 연결 모듈은 아직 없습니다. `front_seconds`, `away_seconds`, `previous_stable_score`는 호출자가 전달해야 하며, 현재 함수는 시간이나 이전 상태를 기억하지 않습니다.

## 3. 파일별 역할

| 파일 | 주요 구성 | 구현 내용 |
|---|---|---|
| [config.py](config.py) | `ModelConfig`, `__post_init__()` | 실험 설정 보관과 값 검증 |
| [input_adapter.py](input_adapter.py) | `HeadAngles`, `Observation`, `parse_feature_window()` | 기존 영상 출력 검증·변환 |
| [scoring.py](scoring.py) | `classify_orientation()`, `orientation_severity()`, `behavior_score()` | 자세 후보·이탈 강도·행동 점수 계산 |
| [tests/test_model.py](tests/test_model.py) | `ConfigTests`, `AdapterTests`, `ScoringTests` | 기본 단위 테스트 15개 |
| `__init__.py`, `tests/__init__.py` | 패키지 설명 | 별도 실행 로직 없음 |

### config.py

`ModelConfig`는 생성 후 값을 바꾸지 못하는 자료 클래스입니다. 다른 설정으로 실험하려면 새 객체를 만듭니다. 각도 단위는 도, 시간 설정 단위는 초입니다.

- 기본 버전: 설정 `experimental-0.1`, 입력 특징 `experimental-0.7`
- 정면: yaw·pitch 각각 ±15도
- 좌우 이탈: 30도부터, 60도에서 강도 상한
- 위쪽 이탈: 25도부터, 45도에서 강도 상한
- 아래 보기: 상대 pitch +20도 이상
- 이탈 유예 3초, 시간 감점 상한 10초, 정면 보너스 상한 5초
- 정면 기본 90점·보너스 10점, 이탈 기본 70점·각도 감점 30점·시간 감점 40점

`__post_init__()`는 음수·무한대·참/거짓 값, 잘못된 각도 순서, 0으로 나누게 되는 설정 등을 거부합니다.

**10초 집계, 2초 출력, 2초 복귀 확인, 최대 관측 유지 0.3초, 최소 관측률·점수화 비율 0.7은 설정만 있으며 현재 계산 흐름에는 아직 적용되지 않습니다.**

### input_adapter.py

- `HeadAngles`: 상대 `pitch`, `yaw`, `roll` 저장
- `Observation`: 세션 ID, 시각, 특징 버전, 각도, 사유 코드, 임시 추적 키 저장
- `Observation.observable`: 각도가 있으면 `True`, 없으면 `False`
- `parse_feature_window(data, config=None)`: 원본 입력을 수정하지 않고 관측 객체 반환

현재 입력 조건은 다음과 같습니다.

| 입력 | 검사·처리 |
|---|---|
| `session_id` | UUID 문자열 검사 |
| `start_ms`, `end_ms` | 0 이상의 정수, 두 값이 같은 시점 입력만 지원 |
| `feature_version` | 설정의 특징 버전과 일치해야 함 |
| `features.faces` | 단일 얼굴만 지원 |
| `head_calibration_status` | `ready`여야 각도 사용 |
| 얼굴별 `quality.status` | `valid`여야 각도 사용 |
| `head_angles_relative_degrees` | 세 축 모두 유한한 숫자이며 −180~180도 |

잘못된 버전·시간·자료 구조는 `ValueError`로 알립니다. 미검출·복수 얼굴·보정 미완료·품질 부족·각도 결측은 `angles=None`과 사유 코드를 반환합니다. 사유에는 `face_not_found`, `multiple_faces_unsupported`, `baseline_unavailable`, `landmark_low_quality`, `relative_angles_unavailable` 등이 있습니다.

상체 결측으로 최상위 품질이 `partial`이어도 얼굴이 유효하면 사용할 수 있습니다. 최상위가 `not_observable`이면 사용하지 않습니다. 임시 추적 키는 보관만 하며 학생 식별·추적 기능은 없습니다.

### scoring.py

- `classify_orientation()`: 한 시점의 자세 후보 분류. 지속적인 비참여 판정은 아님
- `orientation_severity()`: 좌우·위쪽 이탈 강도를 0~1로 반환. 아래 보기·결측은 `None`
- `behavior_score()`: 관측과 전달받은 유지 시간으로 행동 점수 반환
- 내부 함수 `_number()`, `_validate_angles()`, `_clip()`: 숫자·각도 검사와 범위 제한

세 함수는 같은 `ModelConfig`를 사용할 수 있습니다. 어댑터가 품질을 검사하므로 실제 연결 시에는 어댑터를 거친 관측을 점수 함수에 전달해야 합니다.

## 4. 점수 계산 알고리즘

### 상대 각도와 자세 후보

영상 모듈이 계산한 개인별 정면 대비 상대 각도를 사용합니다. yaw는 좌우 회전, pitch는 위아래 회전입니다. 현재 환경에서 pitch 양수는 아래쪽입니다. roll은 좌우 기울기이며 현재 점수에는 사용하지 않고 값의 유효성만 검사합니다.

기본 설정에서 아래 순서로 분류합니다.

| 순서 | 조건 | 상태 |
|---|---|---|
| 1 | 각도 없음 | `not_observable` |
| 2 | pitch ≥ +20도 | `looking_down` |
| 3 | yaw 절댓값 ≤ 15도, pitch 절댓값 ≤ 15도 | `front` |
| 4 | yaw 절댓값 ≥ 30도 또는 pitch ≤ −25도 | `away` |
| 5 | 나머지 | `ambiguous` |

아래 보기를 먼저 분류하므로 아래+옆 보기 역시 즉시 감점하지 않습니다. 필기를 직접 인식하는 것은 아니며, 필기 가능성 때문에 판단을 보류하는 규칙입니다.

### 계산식

`clip(x)`는 0~1, `clip100(x)`는 0~100으로 값을 제한합니다. 아래는 기본 설정을 대입한 식입니다.

정면 유지 점수:

```text
S_front = 90 + 10 × clip(front_seconds / 5)
```

방향 이탈 강도와 지속 시간 비율:

```text
A_yaw = clip((abs(yaw) − 30) / 30)
A_up  = clip((−pitch − 25) / 20)
A     = max(A_yaw, A_up)
D     = clip((away_seconds − 3) / 7)
```

이탈 시간이 3초를 초과했을 때:

```text
S_away = clip100(70 − 30 × A − 40 × D)
```

3초 이내의 이탈은 `previous_stable_score`를 그대로 반환합니다. 이전 안정 점수가 없으면 `None`입니다. 정면 복귀 확인과 시간 자동 누적은 아직 구현하지 않았습니다.

| 입력 예시 | 결과 |
|---|---|
| 정면 범위, 정면 유지 2.5초 | 95점 |
| yaw 45도·pitch 0도, 이탈 6.5초 | A=0.5, D=0.5 → 35점 |
| yaw 60도, 이탈 3초, 이전 안정 점수 95점 | 95점 유지 |
| pitch +20도, 이탈 시간 인자 100초 | `None` — 장시간 아래 보기도 현재 감점하지 않음 |

### None과 0점의 구분

`behavior_score()`는 아래 보기·불명확·관측 불가에 `None`을 반환합니다. Python의 `None`은 JSON 직렬화 시 `null`이 되지만, 현재 이벤트 생성·저장 기능은 없습니다.

0점은 유효한 입력으로 계산된 낮은 행동 점수입니다. 미검출이나 보정 실패를 0점으로 바꾸지 않습니다. 아래 보기 보류와 관측 불가는 자세 분류 결과 및 관측 사유를 함께 확인해 구분해야 합니다.

현재 계산은 3초 유예 직후 이탈 점수식으로 전환하므로 점수가 급격히 변할 수 있습니다. 전환의 적절성과 계수는 후속 실험 대상입니다.

## 5. 테스트

저장소 루트에서 실행합니다. 단위 테스트는 실제 카메라 없이 Python 표준 라이브러리로 실행합니다.

```powershell
python -B -m unittest discover -s engagement_model/tests -v
```

2026-11-04 문서 작성 시 Python 3.14.5 환경에서 재실행한 결과:

```text
Ran 15 tests in 0.006s
OK
```

| 테스트 클래스 | 개수 | 검증 항목 |
|---|---:|---|
| `ConfigTests` | 2 | 설정 변경, 잘못된 설정 거부 |
| `AdapterTests` | 5 | 원본 유지, 상체 결측 허용, 미검출·복수 얼굴·보정·품질·각도·형식 오류 |
| `ScoringTests` | 8 | 정면 보너스, 유예, 강도 경계, 지속 감점, 아래 보기, 불명확·결측, 잘못된 인자, 사용자 가중치 |

15개 모두 통과했습니다. 이는 준비된 입력에서 함수가 규칙대로 동작한다는 검증입니다. 실제 영상의 각도 정확도, 필기 오판율, 참여도 예측 성능, 실시간 전체 연결은 검증하지 않았습니다.

## 6. 현재 구현 상태와 향후 계획

### 완료

- 실험 설정과 유효성 검사
- 시점별 영상 특징을 단일 학생 관측으로 변환
- 단일 얼굴·정면 보정 상태·품질·상대 각도 검사
- 자세 후보와 방향 이탈 강도 계산
- 전달받은 유지 시간을 이용한 행동 점수와 None 처리
- 기본 단위 테스트 15개

### 미구현

- 실제 시간 누적, 시간 역전·추적 단절·보정 무효화에 따른 상태 초기화
- 상태 전환과 정면 복귀 확인
- 10초 이동 구간 집계·2초 간격 출력, 시간 가중 평균
- 관측률·점수화 가능 비율에 따른 구간 결과 보류
- 장시간 아래 보기 감점 정책 비교
- `model_consumer` 연결, `engagement_window` 이벤트 생성, frontend 표시
- 다중 학생별 보정·연결·집계
- 정답 데이터 기반 MAE·Macro F1·필기 오판율·관측률 평가

### 향후 순서

1. 단일 학생의 시간 누적·상태 전환·관측 공백 처리 구현
2. 구간 집계와 관측률 적용
3. integration 이벤트 및 frontend 연결
4. 실제 영상에서 필기·촬영 각도별 평가와 설정 비교
5. 안정적인 학생/타일 연결과 개인별 보정 기반을 마련한 뒤 다중 학생 확장

정답 점수가 없는 동안 MAE를 실제 성능으로 보고하지 않습니다. 먼저 독립적으로 작성한 행동·참여 상태 레이블과 예외 상황 평가를 사용하고, 점수 평가 데이터가 준비되면 MAE를 추가합니다.
