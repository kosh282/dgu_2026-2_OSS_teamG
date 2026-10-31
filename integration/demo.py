"""통합부 연결 방법을 보여 주는 합성 데이터 예제."""
import json
from pathlib import Path
import tempfile
from uuid import uuid4
from . import IntegrationService


def main():
    service = IntegrationService()
    session_id = service.start_session("합성 수업")
    utterance_id = str(uuid4())
    partial = service.make_event(session_id, "transcript_partial", {
        "utterance_id": utterance_id, "revision": 1, "text": "오픈소스"
    }, 0, 0, "speech")
    service.receive_event(partial)
    final = service.make_event(session_id, "transcript_final", {
        "utterance_id": utterance_id, "revision": 2,
        "text": "오픈소스 라이선스를 살펴봅니다.", "is_edited": False
    }, 0, 0, "speech")
    service.receive_event(final)
    note = service.make_event(session_id, "note_created", {
        "text": "비교 예제를 추가하기", "linked_event_ids": [final["event_id"]]
    }, 0, 0, "ui")
    service.receive_event(note)
    print("프런트용 스냅샷:")
    print(json.dumps(service.get_dashboard_snapshot(session_id), ensure_ascii=False, indent=2))
    service.stop_session(session_id)
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        path = Path(directory) / "session.json"
        service.save_session(session_id, path)
        reader = IntegrationService()
        restored_id = reader.load_session(path)
        print("저장 후 재열기:", restored_id, "/ 전사", len(reader.get_dashboard_snapshot(restored_id)["transcripts"]), "건")


if __name__ == "__main__":
    main()
