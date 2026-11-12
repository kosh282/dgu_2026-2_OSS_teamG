# 모델 수식과 프로젝트 변형

[현재 구현·실행 안내](../README.md) · [실험 해석](experiments.md) · [과거 실험·변경 기록](implementation-history.md)

수식의 구현 근거는 `scoring.py`, `input_adapter.py`, `runtime.py`, `gaussian.py`입니다.
점수는 방향·시간에 대한 실험 지표이며 실제 집중도 확률이 아닙니다.

## 공통 입력과 각도 완화

`parse_feature_window()`는 `head_angles_relative_degrees`를 읽습니다.
원본/스무딩 각도를 다시 추정하지 않습니다. 상대각은 `source_angles`로 보존합니다.
상태 분류용 Pitch는 그대로이고 Yaw에만 `soften_angle()`을 적용합니다.

```text
h(a; d,k,b) = 0                                              if |a| <= d
             sign(a) * [k*(min(|a|,b)-d) + max(0,|a|-b)]      otherwise
```

- 상태 분류·Legacy Yaw: `d=2°, k=0.75, b=15°`.
- Gaussian softened 점수 Pitch: `d=2°, k=0.75, b=10°`.
- Gaussian softened 점수 Yaw: 상태 분류와 같은 설정.
- 큰 각도에서는 기울기가 1이며 완화 구간에서 생긴 오프셋만 남습니다.
- 홀함수이므로 동일 Pitch에서 ±동일 Yaw의 점수는 같습니다.
- Pitch 완화는 프로젝트가 추가한 실험 정책이며 기존 Legacy Pitch 처리나 논문 원식이 아닙니다.
- 아래 보기 판정·전이 보류에는 완화 전 Pitch를 사용합니다.

## Legacy

`classify_orientation()`의 상태 순서는 상위 README를 참고합니다.
`behavior_score()`의 기본 설정 대입식은 다음과 같습니다.

```text
clip(x) = max(0, min(1, x))
S_front = 90 + 10 * clip(front_seconds / 5)
A_yaw = clip((abs(model_yaw) - 30) / 30)
A_up  = clip((-pitch - 20) / 20)
A = max(A_yaw, A_up)
D = clip((away_seconds - 3) / 7)
S_away = max(0, min(100, 70 - 30*A - 40*D))
```

`away_seconds <= 3`이면 새 감점 대신 이전 안정 점수를 반환합니다.
이전 점수가 없으면 null입니다. ambiguous·관측 불가 역시 null입니다.
아래 보기 시작 점수가 있으면 5초 유예 뒤 10초 동안 최대 20점을 감점합니다.

```text
S_down = clip100(down_start_score - 20*clip((looking_down_seconds-5)/10))
```

이 점수 유지·감점 규칙은 프로젝트의 실험 규칙입니다.
화면용 `display_score` 유지와 계산식의 `previous_stable_score`는 별개입니다.
현재 점수가 null일 때의 화면 캐시는 최대 2초이며 집계 입력으로 사용하지 않습니다.

## Huang 온라인 모델

출처: Huang, Zhuo, Gao (2024),
*How attentive is a student in class? A concentration evaluation method based on head Euler angle*.
[DOI](https://doi.org/10.1016/j.heliyon.2024.e37365) ·
[Method 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC11417239/)

원문의 큰 구간 Pitch 시퀀스는 `SP_i`, 전체 시퀀스는 `S`, 작은 구간 평균은 `p_ij`입니다.
원문 Pitch는 위쪽 양수입니다. `scoring.py`의 순수 함수가 다음 식을 구현합니다.

```text
(1) q_i = median(SP_i) if |Δ(SP_i)| >= θ else median(S)
(2) a_ij = |(p_ij-q_i)/(max(SP_i)-q_i)| if p_ij >= q_i
           |(p_ij-q_i)/(min(SP_i)-q_i)| otherwise
(3) P_ij = 1/(1+exp(-a_ij*(p_ij-q_i)))
(4) β = |yaw|-cs if x >= W/2 else |yaw|+cs
(5) cs = atan(|x-W/2|/y)
(6) Y = 1-β/O
(7) C = 1-sqrt((Y-1)^2+(P-1)^2)/sqrt(2)
```

식 (5)의 각도 단위는 식 전체에서 일치시킵니다. 구현은 도로 변환합니다.
`Δ`는 구현에서 max-min 진폭으로 해석하며, 백분율은 0~1 비율로 표현합니다.

`huang_head_up()`, `huang_front_facing()`, `huang_concentration()`이 대응합니다.
온라인 `huang_window_score()`는 다음 변형을 포함하며 논문 전체의 재현이 아닙니다.

- 프로젝트의 아래쪽 양수 Pitch를 반전하고 Yaw 압축 전 상대각을 사용합니다.
- 미래의 전체 중앙값은 사용할 수 없어 낮은 진폭 구간의 기준을 보정 정면 0도로 둡니다.
- 교실 공간좌표가 없어 `Y_local=clip(1-abs(relative_yaw)/90)`로 대체합니다.
- θ=5도, 큰 구간 30초, 작은 구간 3초를 기본으로 사용합니다. 시간 선택은 논문 범위에 기초한 프로젝트 설정입니다.
- 유효 관측 시간으로 작은 구간 P/Y를 가중하고 마지막에 `100*C`를 계산합니다.
- 미완료 작은 구간·아래 보기·결측은 제외하고 관측/점수화 시간 비율 70%를 요구합니다.
- 순수 함수는 분모 0에서 null입니다. 온라인에서만 `p=q`인 퇴화 구간을 P=0.5로 처리합니다.
- 이 예외는 프로젝트 중립 정책이며 논문에 명시된 예외식이 아닙니다.

정지 정면에서 P=0.5, Y=1이면 약 64.6점입니다. 정면을 100점으로 보정한 모델이 아닙니다.
아래 보기 후 정면 복귀 시 실시간 이력을 새로 시작하고 기본3초를 수집합니다.
publisher는 전체 근거 구간을 별도로 유지하므로 같은 시점에 이벤트 점수가 보류될 수 있습니다.

## Gaussian 각도 공간 모델

프런트의 재식별 다인 경로는 각 ID에 독립적인 정면 기준과 Gaussian 계산기를 둡니다.
그룹 구간은 각 사람의 유효 관측 점수 `s_i(t)`에 최근성 선형 시간 커널 `w(t)`를 적용해
`Σ_i ∫valid_i w(t)s_i(t)dt / Σ_i ∫valid_i w(t)dt`로 집계합니다.
개인별 이동 평균을 다시 평균하지 않습니다. 재식별 미확정·보정 중·미검출·기준 점수가 없는 아래 보기 구간은
적분 대상에서 제외하며 0점으로 넣지 않습니다. 관측/점수화 시간 gate는 해당 구간에
적어도 한 사람의 유효 관측/점수가 존재하는 벽시계 시간 기준입니다.
저장은 ID 없는 그룹 이벤트만 사용합니다. 분모와 한계는 [다인 모델 안내](../README.md#프런트-다인-gaussian)를 참고하세요.

독립 데모에서 `--gaze`를 켜면 사전 학습 MobileGaze 모델로 하향 관측 선택을 대체합니다.
Gaussian 수식 자체는 유지하며, 머리만 숙인 경우 점수를 계산합니다.
상대 머리 Pitch≥15°와 모델 시선 Pitch≤−30°/|Yaw|<35°가 0.4초 이상 함께
확인될 때 보류합니다. 머리 각도와 모델 시선 각도는 서로 다른 좌표 의미이며,
시선 불명확만으로 아래 보기나 0점으로 바꾸지 않습니다. 기본 실행 정책은 유지합니다.
보정·실행 범위는 [README](../README.md#선택-옵션-보정-없는-시선-방향-모델)를 참고하세요.

출처: Li, B.; Liu, P. (2024),
*Online Learning State Evaluation Method Based on Face Detection and Head Pose Estimation*,
Sensors 24(5), 1365, §3.3 식(8)~(10).
[DOI](https://doi.org/10.3390/s24051365)

논문 Gaussian은 화면 중심과 머리 방향 투영점의 좌표 차이·유효 시야 크기를 사용합니다.
현재 입력에는 해당 투영 좌표가 없어 상대 Pitch/Yaw로 변형했습니다.
논문의 시야 크기 계수를 각도 sigma로 그대로 사용하지 않습니다.

```text
relative 모드: P_eff=P, Y_eff=Y
softened 모드: P_eff=h(P;2,0.75,10), Y_eff=h(Y;2,0.75,15)
S_angle=100*exp(-0.5*((P_eff/sigma_pitch)^2+(Y_eff/sigma_yaw)^2))
```

`gaussian_score_angles()`가 전처리를 선택하고 `gaussian_angle_score()`가 순수 Gaussian을 계산합니다.
sigma 기본15/20도와 전처리 계수는 검증 전 프로젝트 실험값입니다.
이 각도 sigma를 최적값으로 선택했다는 실험 근거는 현재 코드만으로 확인할 수 없습니다.
설정·CLI의 기본값과 후보 비교 기능이 있다는 사실을 통계적 최적화 결과로 해석하지 않습니다.
ambiguous도 유효 각도면 점수화하고 away에 Legacy3초 유예를 적용하지 않습니다.
하향 전이와 미검출은 현재 점수를 null로 둡니다. 아래 보기는 직전 유효 점수가 있으면 5초 유지 후 10초간 최대 20점을 선형 감점합니다.

### 최근10초의 시간 가중

논문의 기록 순서 선형 가중평균은 `li_liu_rank_average()`에 보존합니다.

```text
S_rank = sum(k*S_k, k=1..K)/sum(k, k=1..K)  # 오래된 순서
```

실행 모델은 표본 간격이 불규칙할 수 있어 실제 시간에 적분하는 프로젝트 변형입니다.

```text
T=집계 끝, W=10초, a=T-W
w(t)=(t-a)/W
S_time=sum(S_angle(segment)*integral_segment(w(t)dt))
       /sum(integral_segment(w(t)dt))
```

관측을 다음 시각 또는 최대0.3초까지만 유지하고 유효 구간만 분모에 포함합니다.
시간 커널·70% 기준·관측 유지 상한은 논문 원식 그대로가 아닙니다.
균일 표본에서도 시간 적분과 정수 순서 가중식은 정확히 같은 식이 아닙니다.
하향 전이는 진입 시점부터 제외하며 이후 확인됐다고 과거 이벤트를 수정하지 않습니다.
아래 보기 확인은 방향 확인이며 필기나 참여·미참여의 확인이 아닙니다.

## 버전 구분

| 선택 | 모델 버전 | 설정 버전 | publisher 확장 |
|---|---|---|---|
| Legacy | experimental-0.5 | experimental-0.4 | window-1 |
| Huang | huang-online-0.2 | huang-online-0.2 | window-1 |
| Gaussian relative | angle-time-0.4 | gaussian-angle-time-0.4-relative | angle-time-window-4 |
| Gaussian softened | angle-time-0.4 | gaussian-angle-time-0.4 | angle-time-window-4 |

설정 버전은 호출자가 재정의할 수 있으므로 실제 실행 provenance도 확인합니다.
Gaussian 0.1은 현재 별도 실행 모드가 없습니다. relative 0.4는 같은 아래 보기 정책을 공유하는 비교 모드입니다.
보관본의 과거 0.2 결과를 현재 relative 모드의 완전한 재현 결과라고 간주하지 않습니다.
과거 버전 설명·실험 표는 [이력 원문](implementation-history.md)에 별도로 보존합니다.
