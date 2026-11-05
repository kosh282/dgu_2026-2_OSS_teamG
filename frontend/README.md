# 교수자 대시보드

기존 Windows 영상 분석 작업(`LiveSession`)과 공통 `IntegrationService`를 사용하는 Streamlit 화면입니다. Figma의 세션 시작 준비, 라이브 세션, 사후 검토를 연결합니다.

## 실행

저장소 루트에서 같은 Python 환경에 영상 분석, 프런트, STT 의존성을 설치하고 한국어 모델을 한 번 내려받습니다.

```powershell
python -m pip install -r video_recognition/requirements-integrated.txt
python -m pip install -r frontend/requirements.txt
python -m pip install -r sherpa_stt/requirements.txt
python -m sherpa_stt.download_model
python -m streamlit run frontend/app.py
```

실시간 창 캡처는 Windows 전용입니다. Windows 외 환경에서도 준비 화면과 저장 JSON의 사후 검토는 사용할 수 있습니다.

## 화면과 데이터 연결

- **세션 시작 준비**: 수업 제목, 창 선택/목록 갱신, 일회성 창 미리보기, 음성 입력 방식, 분석·저장 동의 확인 후 시작합니다. 창 미리보기는 저장하지 않습니다. 음성 전사는 선택 사항이며 출력 루프백 또는 마이크 중 하나를 선택합니다.
- **라이브 세션**: 통합부 스냅샷을 1초마다 읽어 타이머, 최신 참여 상태, 실제 이벤트 시각의 그래프, 확정/부분 전사, 자료 제안, 분석 상태, 메모를 갱신합니다. 메모와 자료 제안에서 저장한 메모는 같은 통합부에 기록합니다. 기존 개발자 프리뷰와 종료 기능을 유지합니다.
- **사후 검토**: 종료 수업을 다시 열거나 공통 계약 JSON을 가져옵니다. 타임라인 박스 선택 또는 구간 슬라이더로 전사·메모·자료·피드백을 함께 필터링합니다. 전사 검색/수정, JSON 다운로드, 선택한 수업 삭제를 지원합니다.
- 전사 수정은 기존 발화의 구간과 ID를 유지하는 `transcript_edited` 이벤트로 전달합니다. 기존 근거가 바뀐 자료/피드백에는 재확인 안내를 표시합니다.
- 세션 저장은 통합부의 원자적 저장을 사용합니다. 메모 목록을 별도로 다시 합치지 않으므로 불러오기/재저장 시 메모가 중복되지 않습니다.

### 로컬 기록 위치

기본 경로는 사용자 홈의 `lecturer_dashboard/sessions/`입니다. `LECTURER_SESSION_DIR` 환경변수로 변경할 수 있습니다. 수업 종료 시 자동 저장하며, 재실행 뒤 지난 수업 목록에서 열 수 있습니다. JSON 가져오기는 공통 계약 검사를 통과한 파일만 저장하며, 동일 세션 ID가 있으면 기존 파일을 유지합니다. 삭제는 이 앱의 기록만 제거하고 별도 다운로드 파일에는 영향을 주지 않습니다. 브라우저를 닫거나 프로세스가 비정상 종료되었을 때 자동 종료·복구하는 기능은 없습니다.

### 모듈 연결 상태와 제한

- 영상 캡처·MediaPipe는 기존 `LiveSession` 경로를 사용합니다. 이 변경의 Linux 검증에서는 Windows 캡처·실제 영상 처리를 실행하지 않았습니다.
- 라이브 세션은 `engagement_model.EngagementRuntime`을 `VisionPipeline.model_consumer`로 호출해 최신 실험 행동 점수를 표시합니다. 이는 단일 얼굴의 머리 방향·유지 시간 기반 참고값이며 집중도·이해도를 나타내지 않습니다. 모델 출력은 세션 메모리에만 두며, 10초 구간 집계와 `engagement_window` 이벤트 발행은 아직 연결되지 않아 공통 타임라인·JSON에 저장되지 않습니다.
- 라이브 세션은 `sherpa_stt.LiveTranscription`을 같은 `IntegrationService`와 세션 ID로 실행합니다. 부분·확정 전사는 계약 이벤트로 라이브 화면에 반영되며 확정 전사만 세션 JSON에 저장합니다. 종료 시 STT를 먼저 멈춰 마지막 발화를 확정한 뒤 세션을 종료합니다.
- STT 설치에는 `sherpa_stt/requirements.txt`와 기본 한국어 모델(압축 약 420MB)이 필요합니다. 준비 화면은 선택한 입력에 필요한 패키지와 모델 파일 유무를 표시하며, 누락되면 세션은 영상 분석을 계속하고 STT 상태를 `unavailable`로 기록합니다.
- 출력 루프백은 기본 출력 장치의 모든 소리를 받아 선택한 Zoom 창 소리만 분리하지 않습니다. 마이크와 루프백 동시 수집은 아직 지원하지 않습니다. Windows 실제 장치에서의 동작 검증은 환경별로 확인해야 합니다.
- 자료 검색·사후 피드백 생성은 아직 연결되지 않습니다. 업로드 자료는 현재 브라우저 세션 메모리에 보관되며 검색 모듈에 전달하거나 파일로 영구 저장하지 않습니다.
- 참여 점수는 `payload.score`, `payload.engagement_score`, `payload.model_output.score`, `payload.model_output.engagement_score`를 지원합니다. 모델 출력 키는 아직 최종 확정되지 않았으므로 최종 계약에 맞춰 `dashboard_data.score()`를 조정해야 할 수 있습니다. 숫자가 없거나 품질이 `not_observable`이면 0점으로 대체하지 않습니다.
- 수업 전체 점수와 분석 가능률은 제공된 각 이벤트 구간 길이로 가중 평균합니다. 누락된 구간을 0점으로 간주하지 않습니다. 중첩하지 않는 집계 구간을 전제로 합니다.
- 신호 레인과 변화 알림은 세부 이벤트 형식이 확정되지 않아 시안의 정적 예시를 표시하지 않습니다. 현재 사후 요약 지표의 건수는 실제 자료 제안 수입니다. 강의 요약은 생성 모듈 연동 전입니다.
- 자료 제안의 숨김/메모 저장 후 숨김은 현재 브라우저 세션의 표시 상태입니다. 제안 자체의 계약상 상태를 변경하지 않습니다.
- 전사는 같은 `LiveSession.service.receive_event()`에 전달되며 세션 JSON에 저장됩니다. 별도 프로세스와의 HTTP/WebSocket 연결은 없습니다.

## 검증

```bash
python -m unittest discover -s frontend/tests -v
```

회귀 검증: 실제 시각/2시간 세션 축, 같은 시각의 숫자와 null 점수, 잘못된 점수, 구간 교차, 전사 수정·메모 저장 후 반복 저장/재열기, 저장 경로, 준비/검토 화면, 공통 스냅샷을 읽는 라이브 화면.
