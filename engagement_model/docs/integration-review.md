# 최신 main 기반 통합 검토

통합 브랜치 `integration/engagement-mediapipe-main`은 main `5292724`에서 생성했다.
기존 작업은 feature 브랜치의 `5bc6ccc`로 백업됐으며 이 통합 작업은 별도 worktree에서 수행한다.
통합 브랜치에서는 Commit·Push·PR 생성·main Merge를 수행하지 않았다.
main의 파일을 통째로 덮어쓰지 않고 승인된 engagement_model 변경만 반영했다.

## 수정한 설정 호환성 문제

`UncertaintyDiagnostics`는 20도 정면 후보가 `현재 정면 < 20 < 이탈`을 만족할 때만 생성한다.
이탈 20도 이하에서는 `candidate_reaches_away_boundary`, 현재 정면 20도 이상에서는
`candidate_not_wider_than_baseline`으로 후보를 비활성화한다.
원래 모델 설정·분류 임계값·점수식은 변경하지 않는다. 비활성 후보의 통계를 실제 비교 결과로 표시하지 않는다.
이탈 16/19/20도, 정면 20/25도, 20도 직상단 부동소수점 경계와 기본 설정을 회귀 검사한다.
reset·입력 처리·기준 모델 전체 출력이 후보 사용 여부와 무관하게 유지되는지도 확인한다.

## 최신 main과의 중첩

| 대상 | main의 변경 | 통합 시 보존할 내용 |
|---|---|---|
| `webcam_demo.py` | 고정 레이아웃, 디버그 페이지, live 렌더링, gaze, yaw-log | main 렌더러와 공개 인자 유지 후 독립 pose-source/비교 옵션을 추가 |
| `README.md` | 프런트 Gaussian·ID별 다인 집계·MobileGaze 설명 | main의 현재 제품 설명 유지, 단일 얼굴 제한은 독립 자세 비교 실험에만 명시 |
| `docs/webcam-diagnostics.md` | yaw-log 저장 정책·진단 지표·페이지 안내 | main의 명시적 저장 옵션 설명과 이번 기본 메모리 비교 정책을 구분 |
| `gaussian.py` | 선택적 `downward_policy` | 기본 기존 정책과 선택적 gaze 정책 모두 보존 |
| `multi_face.py`, `window_publisher.py` | ID별 상태·`aggregate_window()`·사람/시간 집계 | 현재 작업으로 교체하지 않음. 주 웹캠 비교기는 단일 얼굴 전용으로 유지 |

기준 HEAD에서 양쪽 수정 구간을 대조하면 실행기와 README에 실제 줄 중첩이 있다.
진단 문서는 줄 중첩이 없더라도 저장 정책과 제품 설명의 의미를 함께 확인해야 한다.
이 검사는 실제 Git 병합을 수행한 결과가 아니므로 충돌 마커 해결을 완료한 상태는 아니다.

**해결한 호출 비호환:** main의 `render_panel(..., debug_page=..., live=True)` 시그니처와
고정 레이아웃·디버그 페이지·live 표시를 보존했다. 해당 API에 MediaPipe 모델을 전달하는 검사가 있다.
main의 PoseInputDiagnostics·yaw-log 집계 저장·gaze 연결을 유지했다.
기본/gaze 경로는 콜백에 기존 런타임 객체를 그대로 전달하고, 구 보정 비교만 추가 콜백을 사용한다.
`--gaze`와 실험용 자세 선택의 동시 사용은 하향 정책 혼합을 피하기 위해 명시적으로 거부한다.
gaze 단독 실행과 기존 저장 옵션은 유지한다.

## 적용한 통합 방법

1. 백업 Commit을 보존하고 main에서 별도 브랜치를 생성했다.
2. main의 다인 런타임·publisher·gaze·live 렌더러·저장 정책을 보존했다.
3. 독립 웹캠에 자세 분석기·동일 프레임 비교·보정 진단을 작은 변경 단위로 추가했다.
4. live/debug_page와 gaze/yaw-log CLI 옵션을 유지하면서 pose-source 표시를 추가했다.
5. ID별 특징을 단일 얼굴 비교기에 전달하거나 얼굴 배열 순서를 ID로 사용하지 않는다.
6. main의 테스트와 신규 회귀·통합 경로 검사를 함께 실행했다.
7. 실제 웹캠 시작·검출·보정·MediaPipe 점수 계산을 짧게 확인했다.

실제 다인 얼굴 재식별 성능·장시간 수업·시선 모델 실측은 이번에 검증하지 않았다.
이번 독립 웹캠의 실측 개선을 프런트/다인 추정 정확도로 표현하지 않는다.

## 검사 범위

- 통합 worktree의 모델·frontend 테스트와 별도 사전학습 gaze 연결 테스트를 실행한다.
- `tests/test_main_integration.py`는 live 렌더러, MediaPipe 주 표시·양쪽 보정,
  기존 gaze/yaw-log CLI, 프런트 재식별→ID별 점수→익명 구간 발행을 검사한다.
- 재식별 모델·검출 프레임은 모의 입력이고 모델·publisher·계약 서비스는 실제 구현을 호출한다.
- 문서 상대 링크, Python 구문, diff 공백 검사와 팀원 영역 diff를 확인한다.
- Gaussian 식·sigma·Legacy·다인 runtime·publisher 및 팀원 파일은 main과 동일하게 유지한다.

## 추후 커밋 대상 명세

아래 파일만 명시적으로 추가한다. 이 명세는 staging·Commit을 실행한 것이 아니다.

```text
engagement_model/README.md
engagement_model/docs/webcam-diagnostics.md
engagement_model/docs/integration-review.md
engagement_model/input_adapter.py
engagement_model/webcam_demo.py
engagement_model/face_transform.py
engagement_model/pose_validation.py
engagement_model/rotation_calibration.py
engagement_model/tests/test_face_transform.py
engagement_model/tests/test_pose_validation.py
engagement_model/tests/test_rotation_calibration.py
engagement_model/tests/test_yaw_calibration.py
engagement_model/tests/test_main_integration.py
```

`test_yaw_calibration.py`의 공통 합성 입력은 다른 회귀 검사도 사용하므로 누락하지 않는다.
`rotation_calibration.py`는 기본 비활성 실험 경로이며 관련 회귀 검사와 함께 보존한다.

제외: `.comparison-pre-gaussian/` worktree, 참조되지 않는 `frontend_pose_comparison.py`,
가상환경·캐시·원본 미디어·로컬 진단 로그·세션 JSON.
`git add .`로 worktree나 미완료 파일을 함께 추가하지 않는다. 제외 대상은 요청 없이 삭제하지 않는다.

## 현재 판단

진단기 설정 오류와 공개 렌더러/콜백 충돌을 해결한 로컬 통합 변경이다.
단위·통합 경로 검사를 통과해 리뷰할 수 있지만 실제 다인 정확도나 모든 옵션 조합의 성능을 보장하지 않는다.
기존 PnP가 기본이며 MediaPipe는 `--pose-source face-transform`으로 선택한다.
사용자 확인 후 통합 브랜치 Commit·Push·PR을 별도 진행한다. main Merge는 별도 검토가 필요하다.
