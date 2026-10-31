# 통합부 구현 내용 (2026-10-31)

## 구현 범위

팀 공통 데이터 계약 v0.1.0을 바탕으로 실시간 수업 데이터를 한 프로세스에서 관리하는 Python 최소 버전입니다. 표준 라이브러리만 사용하며 Python 3.10 이상을 권장합니다.

- `integration/contracts.py`: 공통 이벤트 필드와 주요 이벤트별 payload 검사. 알려지지 않은 이벤트도 공통 형식에 맞으면 보존
- `integration/service.py`: 세션 생성·공통 시계·이벤트 수신, 현재 상태, 프런트용 스냅샷, 시간 구간 조회, JSON 저장/재열기
- `integration/demo.py`: 합성 전사·메모를 연결하고 스냅샷과 저장/재열기를 보여 주는 실행 예제. 세션 JSON은 임시 폴더에 저장됐다가 예제 종료 시 지워짐

평가 모델은 아직 선택하지 않았습니다. `participation_model`이 없으면 시작 시 `analysis_status`에 `model_not_configured`를 기록하고, 스냅샷의 `status.model`을 `unavailable`로 표시합니다. `engagement_window`의 공통 틀과 모델 ID/버전은 검사하되 모델별 출력·점수의 세부 형식은 고정하지 않았습니다.

## 시작 방법

PowerShell에서 저장소 루트 폴더로 이동한 후 실행합니다.

```powershell
python -m integration.demo
```

Python 실행 명령은 팀 환경에 따라 `py -3 -m integration.demo`로 바꿀 수 있습니다. 별도의 pip 설치는 필요하지 않습니다.

## 각 파트 연결 예시

**중요:** 같은 앱 프로세스에서 하나의 `IntegrationService` 인스턴스와 `session_id`를 공유해야 합니다. 각 파트에서 인스턴스를 새로 만들면 서로 다른 메모리를 보게 됩니다. 스레드에서 동일 인스턴스를 호출할 수 있습니다. 다른 프로세스나 다른 기기에서 연결하는 API·메시지 큐는 아직 없습니다.

```python
from uuid import uuid4
from integration import IntegrationService

service = IntegrationService()
session_id = service.start_session("공개SW 수업")
utterance_id = str(uuid4())

# 수집 시각은 캡처 시점에 service.elapsed_ms(session_id)로 기록.
start_ms = service.elapsed_ms(session_id)
# ... 오디오 수집과 STT ...
end_ms = service.elapsed_ms(session_id)
event = service.make_event(
    session_id=session_id,
    event_type="transcript_final",
    payload={"utterance_id": utterance_id, "revision": 1,
             "text": "예시 전사", "is_edited": False},
    start_ms=start_ms, end_ms=end_ms, module="speech",
    model_id="실제-STT-모델-ID", model_version="실제-모델-버전",
)
service.receive_event(event)
snapshot = service.get_dashboard_snapshot(session_id)
```

기존 계약대로 파트가 공통 외피를 직접 만들어 `receive_event(event)`로 전달해도 됩니다. `make_event()`는 편의 함수이며 이벤트를 자동 등록하지 않습니다. 이벤트의 공통 시간은 **같은 서비스 세션의 단조 시계**를 사용합니다. `end_ms`가 수신 당시 세션 시각보다 미래이면 오류를 반환합니다.

| 함수 | 설명 |
|---|---|
| `start_session(title, config=None)` | 세션 생성 후 UUID 문자열 반환. `config` 지정 시 `config_version` 필수 |
| `elapsed_ms(session_id)` | 공통 수업 상대 시각. 종료 후에는 세션 길이 반환 |
| `make_event(...)` | 공통 이벤트 객체 생성. 수신은 별도 호출 |
| `receive_event(event)` | 검사 후 반영. 신규 이벤트는 `True`, 같은 ID 재전달은 `False` |
| `get_dashboard_snapshot(session_id)` | 상태·부분/확정 전사·모델 결과·메모·조교 결과의 복사본 |
| `get_events_in_range(session_id, start_ms, end_ms)` | `[start_ms, end_ms)`와 겹치는 저장 대상 이벤트 조회 |
| `stop_session(session_id)` | 세션 시계 정지, 부분 전사 메모리 정리. 호출 전에 각 캡처/STT 작업을 중지하고 마지막 확정 전사를 수신해야 함 |
| `save_session(session_id, file_path)` | UTF-8 JSON 저장. 임시 파일 작성 후 대상 파일을 교체하므로 지정한 기존 파일은 덮어씀 |
| `load_session(file_path)` | 세션 JSON을 종료된 상태로 불러오고 ID 반환. 같은 서비스에 같은 세션이 있으면 오류 |

## 처리 기준과 프런트 연동

- `transcript_partial`은 발화 ID별 최신 `revision`만 메모리에 유지하고 저장하지 않습니다. `transcript_final`을 받으면 부분 전사를 지우고 확정합니다.
- `transcript_edited`는 기존 확정본의 ID를 `replaces_event_id`에 넣고 더 높은 `revision`과 동일 시간 구간으로 전달합니다. 현재 확정본만 저장합니다.
- `resource_suggestion`과 `post_session_feedback`은 현재 존재하는 근거 이벤트 ID를 받아야 합니다. 전사 수정으로 근거가 사라진 경우 스냅샷에 `evidence_needs_refresh: true`가 붙습니다. 이 필드는 화면용이며 저장 이벤트를 바꾸지 않습니다.
- 스냅샷의 `status`, `partial_transcripts`, `transcripts`, `engagement_windows`, `notes`, `suggestions`, `feedback`을 프런트가 읽습니다. 화면용 데이터는 복사본입니다.
- 이벤트는 수신 순서로 저장하고 타임라인과 구간 조회는 발생 시각으로 정렬합니다. 저장 파일의 최상위 구조는 `schema_version`, `session`, `config`, `events`입니다.
- 원본 프레임·오디오·얼굴 이미지·개인 트랙 등을 저장 이벤트에 넣지 마세요. 일부 대표 키를 차단하지만 이 기능은 모든 민감 데이터를 찾아내는 도구가 아닙니다. 전사와 메모에는 개인 식별 내용이 들어갈 수 있습니다.
- 다른 모듈은 통합부에 원본 미디어가 아닌 결과 이벤트를 전달합니다. 캡처 프레임·오디오 청크와 영상 모델용 중간 특징은 각 모듈 사이의 메모리 경로에서 처리합니다.

## 현재 제한과 다음 작업

실제 화면/음성 캡처, MediaPipe, sherpa-onnx, 참여 모델, 강의 조교, Streamlit 화면은 연결하지 않았습니다. 프런트는 하나의 서비스 인스턴스를 유지하면서 스냅샷을 읽어야 합니다. 장치 오류 감지와 모듈 시작/종료, 프런트 갱신 주기, 멀티프로세스 전달, 자동 저장, 파일 삭제·접근 권한, 모델별 세부 검증은 후속 통합에서 결정해야 합니다. 실행 중 중복 ID는 방지하지만 재열기 시 현재 저장된 이벤트 이외의 과거 부분 전사·교체 전 이벤트 ID까지 기억하지 않습니다. 저장 파일을 읽으면 종료 상태로 복원하며 실시간 세션 재개 기능은 없습니다.

`EventValidationError`는 계약 형식 오류, `IntegrationError`는 세션 상태나 전사 순서 등의 처리 오류입니다. 새 이벤트의 전달 방식·필수 payload가 정해지면 계약 문서와 검사를 함께 갱신해 주세요.
