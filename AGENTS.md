# 저장소 작업 지침

## 프로젝트 목표

이 프로젝트는 교수자가 온라인 수업 프로그램의 창을 선택해 수업 중 관찰 신호와 수업 기록을 함께 확인하는 로컬 실행형 보조 도구다. 아래는 제출 제안서의 제품 목표이며, 현재 구현 완료를 뜻하지 않는다.

- 선택한 수업 창 전체를 캡처한다. 화면 타일의 고정 위치나 참가자 배치에 의존하지 않는다.
- 영상에서 얼굴 방향, 머리 움직임, 표정 변화, 상체 움직임 등 관찰 가능한 특징을 추출한다.
- 영상 분석과 음성 전사를 하나의 수업 시간축에 맞추고, 구간별 참여 흐름·분석 가능 여부·전사·교수자 메모를 함께 검토할 수 있게 한다.
- 확정 전사와 교수자가 제공한 강의 자료를 근거로 자료 제안과 수업 후 피드백을 제공한다. 제안에는 근거 구간과 출처를 연결하고 최종 판단은 교수자에게 둔다.
- 세션 결과는 공통 계약에 맞는 로컬 JSON으로 저장·재열기·삭제할 수 있게 한다.

## 해석과 개인정보 원칙

- 참여 점수와 관찰 신호는 교수자의 수업 성찰을 돕는 참고 자료다. 실제 집중도·이해도·성적·심리 상태를 확정하거나 학습자 평가에 직접 사용하지 않는다.
- 얼굴 방향이나 표정만으로 낮은 참여를 단정하지 않는다. 가림, 작은 얼굴, 조명, 창 크기, 카메라 각도 등으로 관측이 어려운 경우는 점수 0이 아니라 관측 불가와 사유로 표현한다.
- 동의는 수업 분석의 전제다. 동의 거부나 철회로 수업 참여에 불이익을 주는 기능을 만들지 않는다.
- 원본 영상·음성은 저장하지 않는 것을 기본으로 한다. 필요한 프레임 특징은 분석·집계에만 쓰고, 지속 저장은 계약에 정의된 최소 결과(예: 집계, 전사, 메모, 설정)로 제한한다.
- 이름, 얼굴 이미지, 영구 개인 식별자, 개인별 행동 이력을 새 저장 항목으로 추가하지 않는다. 필요성이 생기면 저장 범위와 동의·삭제 방식을 먼저 명시한다.
- 진단용 로그가 명시적으로 필요할 때도 원본 미디어를 기록하지 않는다. 민감 특징은 로컬 임시 로그에 최소한으로 남기고, 최종 세션 JSON이나 커밋에 섞이지 않게 한다.

## 구조와 경계

- `frontend/app.py`: Streamlit 화면과 사용자 동작
- `frontend/live_session.py`: 선택 창, 수업 세션, 분석 스레드와 개발자 프리뷰 연결
- `video_recognition/window_capture.py`: Windows 창 캡처
- `video_recognition/integrated_analysis.py`: 프레임 샘플링, 얼굴·상체 분석, 후처리, `FeatureWindow` 생성
- `video_recognition/face_analysis.py`, `pose_analysis.py`, `face_postprocessing.py`, `head_calibration.py`: 영상 특징과 임시 보정 상태
- `engagement_model/`: 특징 어댑터, 규칙 기반 실험 점수와 상태 런타임
- `integration/contracts.py`, `integration/service.py`: 이벤트 계약 검증, 공통 세션 시계, 이벤트와 세션 JSON 관리
- `video_recognition/webcam_test.py`: 웹캠 데모 및 프런트 개발자 모드가 공유하는 출력기

영상 분석은 `FeatureWindow`를 모델 런타임에 전달한다. 모델 계산 결과가 `latest`에 보관되는 것과 그 결과를 `engagement_window` 이벤트로 계약 저장소에 발행하는 것은 별도 단계다. 코드를 확인하기 전 두 단계가 모두 연결됐다고 말하지 않는다.

제안서의 목표는 여러 참가자가 함께 보이는 화면을 처리하는 것이다. 현재 모델 어댑터가 단일 얼굴만 점수화하거나 보정할 수 있다면 다인 분석은 미완료로 설명한다. 얼굴 배열 순서나 화면 좌표를 영구 참가자 ID로 간주하지 않는다.

제안서의 sherpa-onnx 음성 입력·한국어 실시간 전사, 강의 자료 검색, 실시간 자료 제안, 수업 후 피드백은 각각의 실제 모듈과 이벤트 연결을 확인하기 전까지 구현된 것으로 표시하지 않는다. 화면 API 연동은 제안서상 필수 조건이 아니다.

## 데이터 계약과 시간

- 이벤트 필드, payload, 품질 상태, provenance는 `공유용_데이터_계약/README.md`와 `integration/contracts.py`를 기준으로 한다.
- 계약 변경은 관련 문서와 검증 코드를 함께 확인한다. 임의 필드를 조용히 추가하거나 기존 필드의 의미·단위를 바꾸지 않는다.
- 수업 시작을 기준으로 한 공통 상대 시간축을 쓴다. 화면 캡처, 영상 특징, 전사, 메모의 시간 기준을 혼용하지 않는다.
- 값이 없거나 관측할 수 없는 상태는 `null`/품질 상태/사유를 유지한다. 이를 0 또는 정상 상태로 대체하지 않는다.
- 저장·재열기·삭제 시 세션 ID, 이벤트 연결, 근거 구간이 일관되게 유지되는지 확인한다.

## 개발 환경과 실행

- 통합 영상 분석 환경은 `video_recognition/requirements-integrated.txt`, 화면 추가 패키지는 `frontend/requirements.txt`를 기준으로 한다.
- 기본 프런트 실행은 저장소 루트에서 다음 명령을 사용한다.

  ```powershell
  video_recognition/.venv/Scripts/python.exe -m streamlit run frontend/app.py
  ```

- 개발자 웹캠 데모는 다음 명령을 사용한다.

  ```powershell
  video_recognition/.venv/Scripts/python.exe -m video_recognition.webcam_test
  ```

- 창 캡처는 현재 Windows 구현이다. 실제 카메라·수업 창·최소화·화면 배율 동작은 해당 장치에서 실행하지 않았다면 검증됐다고 보고하지 않는다.
- 종속성이나 Python 지원 범위는 추측하지 말고 requirements와 실제 설치 결과를 확인한다.

## 작업 방식

- 작업 전 적용되는 `AGENTS.md`, 해당 기능의 README, 계약 문서와 구현 코드를 확인한다. 제안서의 목표, 현재 코드, 추론한 내용을 구분한다.
- 코드는 영어 식별자를 유지하고, 사용자 문서와 필요한 설명 주석은 한국어로 쓴다. 기존 모듈의 책임과 이름을 존중한다.
- 사용자가 요청한 범위 밖의 파일, 로컬 세션 JSON, 로그, 캐시, 미추적 변경을 덮어쓰거나 정리하지 않는다.
- 검증을 요청받으면 관련 범위의 검사만 실행하고, 실행 명령과 관찰한 결과를 보고한다. 실행하지 않은 검증은 했다고 쓰지 않는다.
- 막힘이나 불확실성이 생기면 조용히 진행하거나 완성된 것처럼 표현하지 말고 가능한 한 빨리 알린다.
- 커밋·푸시는 사용자가 요청할 때만 한다. 외부 사람에게 메시지를 보내지 않는다.
- 동작·계약·설치 방법이 바뀌면 해당 README와 현재 상태 설명도 함께 검토한다. 완료 상태나 날짜가 자주 바뀌는 일정은 이 파일에 고정하지 않는다.

## Proposal evaluation targets (tentative)

The submitted proposal names the following initial evaluation targets. They are goals, not verified results or final acceptance criteria; do not claim them as achieved. Confirm the labels, sample, annotation method, and thresholds with the team before using them as release gates.

- At least 80% agreement between the participation-level model and human annotations is proposed as an initial target. If a continuous score is used, define a separate error metric against annotated scores.
- A Korean streaming transcription character error rate (CER) below 20% is an initial target.
- Average delay of at most 5 seconds from analysis-window end to visible result is an initial target.
- Verify at least 10 minutes of continuous operation, timeline updates, transcript/note retrieval, and JSON save/reload.
- Record the annotation criteria, evaluated intervals, machine/environment, and deferred or unobservable intervals. Treat public datasets such as DAiSEE as research references, not ground truth for real-class engagement.

The proposal's `session`, `transcript/notes`, `aggregates/events`, `assistant`, and `config` are logical data categories, not the current JSON schema. Map them through the shared contract; do not add top-level fields by assumption.
