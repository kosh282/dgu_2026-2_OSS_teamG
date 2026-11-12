# Frontend Multi-face Gaussian Implementation Plan

**Goal:** 재식별된 ID별 Gaussian 점수를 프런트에 표시하고 그룹 집계만 저장한다.
**Architecture:** 기존 FaceReidentifier를 추론/렌더링으로 분리하고 ID별 HeadCalibration/GaussianRuntime을
별도 런타임에 둔다. WindowPublisher는 익명 관측 집계 훅을 통해 그룹 이벤트를 발행한다.
**Tech Stack:** Python, Streamlit, MediaPipe, InsightFace, 기존 ONNX Runtime.
**Spec:** `docs/superpowers/specs/2026-11-12-frontend-multi-gaussian-design.md` (사용자 승인)

## Constraints / Review Focus

- 기존 작업 브랜치에서 수행하며 사용자 파일/미추적 파일을 정리하지 않는다. 커밋/푸시하지 않는다.
- 개인 ID·임베딩·개인별 이력은 메모리 전용, JSON과 성능 로그에는 넣지 않는다.
- 기본 프런트 점수는 Gaussian, 다인 점수는 재식별을 함께 켠 경우에만 사용한다.
- 개별 미검출·재식별 미확정·보정 중·아래 보기 보류는 0점이 아니다.
- 같은 프레임의 얼굴/경계만 연결하며 프리뷰 비활성화로 추론이 멈추면 안 된다.
- 추적 변경·ID 복귀·보정 초기화·모델 오류·종료 이후에 과거 현재점수를 노출하지 않는다.

## Task 1: ID별 계산과 익명 집계

Files: `engagement_model/multi_face.py`, `engagement_model/tests/test_multi_face.py`.
Interfaces: `MultiFaceGaussianRuntime(config=None)`; `.update_identities(mapping)`;
`.__call__(feature)`; `.participants()` (메모리 UI 표); `.recalibrate()`;
`.reset()`; `.aggregate_window(segments,start,end)`.
`.config`는 Gaussian 설정, `.model_id='multi-face-gaussian'`, `.model_version='multi-gaussian-0.1'`.
입력 mapping은 후처리 transient_track_key→임시 재식별 ID이며 특징에는 추가하지 않는다.
최신 결과는 기존 런타임 형식과 `.gaussian_observation`에 익명 [(pitch,yaw)]만 제공한다.
사람별 실제 시간/최근성 가중치로 그룹 점수를 계산하고, 최종 summary는 기존 Gaussian summary 키를 제공한다.

- [x] 두 얼굴 독립 보정/각도·이탈·복귀·유효 시간 가중 그룹 집계 실패 검사 작성/확인
- [x] ID별 런타임과 120초/최대32 상태 수명, null 및 재보정 구현
- [x] 동일 단일 얼굴 Gaussian 및 입력 특징 불변성 검사 통과

## Task 2: 재식별 추론과 표시 분리

Files: `video_recognition/face_reidentification.py`, `experimental_vision/test_reidentification.py`.
Interfaces: `.identify(frame,timestamp_ms,faces,*,boxes)`는 표시를 하지 않고 connections를 갱신한다.
`.render(frame,faces,*,boxes)`는 복사본에 마지막 결과만 표시하며 추론/시간 갱신하지 않는다.
기존 `.annotate(...)`는 identify→render 호환 래퍼로 유지한다. `.close()`는 새 캐시도 지운다.

- [x] 추론 없이 render, 원본 불변, annotate 호환 실패 검사 작성/확인
- [x] API 분리하고 기존 재식별 회귀 검사 통과

## Task 3: LiveSession·publisher·UI 연결

Files: `frontend/live_session.py`, `frontend/app.py`, `engagement_model/window_publisher.py`, 관련 tests/README/계약 문서.
LiveSession `reidentification_enabled=False` 옵션. 활성 시 모델은 MultiFaceGaussianRuntime,
비활성 시 GaussianRuntime. 점수 disabled일 때도 선택한 재식별은 영상과 독립 실행 가능.
`.participant_snapshot()`은 lock으로 UI용 복사만 반환한다. 종료 시 개인 상태/캐시를 지운다.
WindowPublisher는 aggregate_window 훅이 있으면 기존 Gaussian 집계를 대체하고 provenance를 유지한다.

- [x] 프런트 옵션·같은 프레임/ID 전달·publisher 훅·개인정보 없는 JSON 실패 검사 작성/확인
- [x] 모델 생성/재보정/오류재시도/자원해제와 ID 프리뷰·라이브 표 연결
- [x] 계약 확장 분모와 README의 설치/사용/한계 설명 갱신
- [x] frontend/engagement_model/재식별 검사, diff 검사, 마지막 독립 리뷰

## Baseline

최신 main 01bb61d. Mac의 frontend 30개 중 Windows 창 선택 모의 검사 1개가 기존 실패한다.
검사에서 Windows 창 목록을 seed하는 방식으로 플랫폼 의존 없이 검증할 수 있게 보완한다.

## 실행 기록

기존 사용자 작업 브랜치에서 수정했다. 개인별 모델은 7개 새 단위 검사, 재식별 분리는 4개 새 검사로 검증했다.
프런트 결선·모델 provenance·직접 VisionPipeline 실행·저장/재열기·UI·개인정보 경계를 추가 검사했다.
독립 리뷰에서 새 참가자 보정이 전체 그룹 구간을 비우던 문제를 발견해 다인 집계 훅의 초기화 경로를
분리했고, 신규 참가자와 재보정 후 상태 복귀 회귀 검사를 추가했다.
실제 Windows 캡처·다인 인식 정확도·지연·발열은 검증하지 않았다. 커밋/푸시하지 않았다.
