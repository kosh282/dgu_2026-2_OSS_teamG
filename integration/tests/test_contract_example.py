"""공유 계약의 합성 세션 예제를 저장 경계에서 검증한다."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from integration import IntegrationService
from integration.contracts import EventValidationError, validate_event
from integration.service import IntegrationError


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "공유용_데이터_계약" / "examples" / "session.json"


class SharedContractExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_shared_example_loads_and_restores_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            path.write_text(json.dumps(self.document, ensure_ascii=False), encoding="utf-8")
            service = IntegrationService()
            session_id = service.load_session(path)
        self.assertEqual(session_id, self.document["session"]["session_id"])
        reloaded = Path(path.parent) / "round_trip.json"
        service.save_session(session_id, reloaded)
        round_trip = json.loads(reloaded.read_text(encoding="utf-8"))
        self.assertEqual(len(round_trip["events"]), len(self.document["events"]))
        self.assertEqual({e["event_type"] for e in round_trip["events"]},
                         {e["event_type"] for e in self.document["events"]})

    def test_example_events_have_valid_shared_time_and_identity(self):
        session_id = self.document["session"]["session_id"]
        for event in self.document["events"]:
            with self.subTest(event_type=event["event_type"]):
                validate_event(event, session_id)
                self.assertLessEqual(event["start_ms"], event["end_ms"])
                self.assertLessEqual(event["end_ms"], self.document["session"]["duration_ms"])

    def test_event_rejects_missing_fields_and_invalid_timestamps(self):
        event = self.document["events"][1]
        for mutation in (
            lambda value: value.pop("quality"),
            lambda value: value.update(start_ms=True),
            lambda value: value.update(end_ms=value["start_ms"] - 1),
            lambda value: value.update(created_at_utc="2026-01-01T00:00:00"),
        ):
            changed = copy.deepcopy(event)
            mutation(changed)
            with self.subTest(changed=changed), self.assertRaises(EventValidationError):
                validate_event(changed, changed["session_id"])

    def test_import_rejects_invalid_session_utc_timestamp(self):
        changed = copy.deepcopy(self.document)
        changed["session"]["started_at_utc"] = "2026-10-29T01:00:00"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(IntegrationError):
                IntegrationService().load_session(path)

    def test_import_rejects_created_time_after_session_duration(self):
        changed = copy.deepcopy(self.document)
        changed["events"][0]["created_ms"] = changed["session"]["duration_ms"] + 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(IntegrationError):
                IntegrationService().load_session(path)


if __name__ == "__main__":
    unittest.main()
