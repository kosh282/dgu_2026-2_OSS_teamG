# Push 전 호환성 검토

이 문서는 병합 결과가 아니라 현재 작업과 최신 main의 비교 결과다.
검토 기준: 작업 HEAD `569a5da`, origin/main `5292724`, 원격 작업 브랜치 `ef2d9c6`.
Merge·Rebase·Commit·Push는 수행하지 않았다. 최신 main의 파일을 통째로 덮어쓰지 않는다.

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

**확인된 호출 비호환:** 최신 프런트는 `render_panel(..., debug_page=..., live=True)`를 사용한다.
현재 작업 파일의 함수에 같은 호출을 바인딩하면 `unexpected keyword argument 'debug_page'`가 발생한다.
최신 main을 통합할 때 main의 렌더러/함수 인자를 우선 보존해야 한다.
인자를 무시하는 임시 래퍼로 테스트만 통과시키거나 현재 파일 전체로 덮어쓰지 않는다.

## 통합 순서 제안 — 별도 승인 후 수행

1. 현재 작업을 별도로 보존한 뒤 승인된 브랜치에서 최신 main을 통합한다.
2. main의 다인 런타임·publisher·gaze·live 렌더러·저장 정책을 그대로 보존한다.
3. 독립 웹캠에만 이번 자세 분석기와 동일 프레임 비교 경로를 작은 변경 단위로 적용한다.
4. `render_panel`의 live/debug_page와 기존 main CLI 옵션을 유지하면서 pose-source 표시를 추가한다.
5. ID별 특징을 단일 얼굴 비교기에 전달하거나 얼굴 배열 순서를 ID로 사용하지 않는다.
6. 최신 main의 모델·다인·publisher·프런트·yaw-log 검사와 이번 신규 회귀 검사를 함께 실행한다.
7. 현재 점수·시간 집계·아래 보기 정책·재보정·다인 ID 보존을 실제 통합 환경에서 확인한다.

현재는 1~7단계 통합을 수행하지 않았다. 이번 독립 웹캠의 실측 개선을 프런트/다인 검증으로 표현하지 않는다.

## 검사 범위

- 현재 체크아웃의 engagement_model·frontend 테스트를 각각 실행한다.
- 최신 main의 `multi_face.py`와 `test_multi_face.py`만 메모리에 읽어
  현재 config/adapter/Gaussian 코어와 조합한 7개 테스트는 통과했다.
  이는 다인 계산 코어의 부분 호환 검사이며 최신 프런트·publisher 통합 통과를 뜻하지 않는다.
- 문서 상대 링크, Python 구문, diff 공백 검사와 팀원 영역 diff를 확인한다.
- 최신 main을 checkout하거나 파일을 복사해 작업 코드에 섞지 않는다.

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
```

`test_yaw_calibration.py`의 공통 합성 입력은 다른 회귀 검사도 사용하므로 누락하지 않는다.
`rotation_calibration.py`는 기본 비활성 실험 경로이며 관련 회귀 검사와 함께 보존한다.

제외: `.comparison-pre-gaussian/` worktree, 참조되지 않는 `frontend_pose_comparison.py`,
가상환경·캐시·원본 미디어·로컬 진단 로그·세션 JSON.
`git add .`로 worktree나 미완료 파일을 함께 추가하지 않는다. 제외 대상은 요청 없이 삭제하지 않는다.

## 현재 판단

진단기 설정 오류는 수정했지만 최신 main과의 전체 호환성 확인은 완료되지 않았다.
현재 브랜치의 독립 실험 변경으로 리뷰할 수는 있으나, 최신 main/팀 기능과 안전하게 통합됐다고 보고하지 않는다.
최종 통합 검사 전 main 반영·전체 호환 보장을 전제로 한 Push는 보류한다.
