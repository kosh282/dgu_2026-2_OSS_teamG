"""메모리 기반 세션·이벤트 통합 서비스. 외부 의존성 없음."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from uuid import uuid4, UUID

from .contracts import SCHEMA_VERSION, EventValidationError, validate_event, _check_forbidden


class IntegrationError(Exception):
    """세션 상태나 이벤트 처리 문제."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ordered(events):
    return sorted(events, key=lambda e: (e["start_ms"], e["end_ms"], e["event_id"]))


class IntegrationService:
    """한 프로세스에서 여러 생산자와 프런트가 공유하는 통합부.

    모든 공개 메서드는 스레드 안전하며 반환값은 복사본이다.
    프로세스를 넘는 이벤트 전달·자동 저장·사용자 인증은 제공하지 않는다.
    """
    def __init__(self):
        self._sessions = {}
        self._lock = threading.RLock()

    def _get(self, session_id):
        try:
            return self._sessions[session_id]
        except KeyError:
            raise IntegrationError("존재하지 않는 세션입니다") from None

    def start_session(self, title, config=None):
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title은 비어 있지 않은 문자열이어야 합니다")
        if config is None:
            config = {"config_version": "initial", "participation_model": None, "stt_model": None}
        if not isinstance(config, dict) or not isinstance(config.get("config_version"), str) or not config["config_version"].strip():
            raise ValueError("config.config_version이 필요합니다")
        try:
            json.dumps(config, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("config는 JSON으로 직렬화할 수 있어야 합니다") from None
        sid = str(uuid4())
        with self._lock:
            self._sessions[sid] = {
                "session": {"session_id": sid, "title": title, "started_at_utc": _utc_now(), "duration_ms": 0},
                "config": deepcopy(config), "started_ns": time.monotonic_ns(),
                "running": True, "events": {}, "seen": set(), "partials": {}, "transcripts": {},
            }
        if config.get("participation_model") is None:
            event = self.make_event(sid, "analysis_status", {"component": "model", "state": "unavailable", "message": "참여 모델이 설정되지 않았습니다."}, 0, 0, "model", quality_status="not_observable", reason_codes=["model_not_configured"])
            self.receive_event(event)
        return sid

    def elapsed_ms(self, session_id):
        with self._lock:
            state = self._get(session_id)
            if state["running"]:
                return (time.monotonic_ns() - state["started_ns"]) // 1_000_000
            return state["session"]["duration_ms"]

    def make_event(self, session_id, event_type, payload, start_ms, end_ms,
                   module, quality_status="valid", reason_codes=None, model_id=None, model_version=None):
        """호출 측의 payload를 공통 외피에 담는다. 수신·검사는 별도 호출한다."""
        with self._lock:
            state = self._get(session_id)
            running = state["running"]
            return {
                "schema_version": SCHEMA_VERSION, "session_id": session_id,
                "event_id": str(uuid4()), "event_type": event_type,
                "start_ms": start_ms, "end_ms": end_ms,
                "created_ms": self.elapsed_ms(session_id) if running else None,
                "created_at_utc": None if running else _utc_now(),
                "payload": deepcopy(payload),
                "quality": {"status": quality_status, "reason_codes": list(reason_codes or [])},
                "provenance": {"module": module, "config_version": state["config"]["config_version"],
                               "model_id": model_id, "model_version": model_version},
            }

    def receive_event(self, event):
        """이벤트를 반영한다. 중복 ID는 False, 새 이벤트는 True를 반환한다."""
        if not isinstance(event, dict) or not isinstance(event.get("session_id"), str):
            raise EventValidationError("session_id: 필수 필드가 없습니다")
        with self._lock:
            state = self._get(event["session_id"])
            validate_event(event, state["session"]["session_id"])
            eid, kind, payload = event["event_id"], event["event_type"], event["payload"]
            if eid in state["seen"]:
                return False
            if state["running"] and event["end_ms"] > self.elapsed_ms(event["session_id"]):
                raise IntegrationError("이벤트 구간이 현재 세션 시각보다 미래입니다")
            if state["running"] and event["created_ms"] is None:
                raise IntegrationError("진행 중 이벤트에는 created_ms가 필요합니다")
            if not state["running"]:
                if event["created_ms"] is not None:
                    raise IntegrationError("종료 후 이벤트에는 created_at_utc가 필요합니다")
                if kind not in {"transcript_edited", "note_created", "resource_suggestion", "post_session_feedback"}:
                    raise IntegrationError("종료된 세션에는 실시간 이벤트를 추가할 수 없습니다")
                if event["end_ms"] > state["session"]["duration_ms"]:
                    raise IntegrationError("이벤트 시각이 수업 종료 시각을 초과합니다")
            item = deepcopy(event)
            if kind in {"transcript_partial", "transcript_final", "transcript_edited"}:
                uid, revision = payload["utterance_id"], payload["revision"]
                current = state["transcripts"].get(uid)
                partial = state["partials"].get(uid)
                if kind == "transcript_partial":
                    if current is not None:
                        raise IntegrationError("이미 확정된 발화의 부분 전사는 받을 수 없습니다")
                    if partial and revision <= partial["payload"]["revision"]:
                        raise IntegrationError("부분 전사 revision은 증가해야 합니다")
                    state["partials"][uid] = item
                elif kind == "transcript_final":
                    if current is not None:
                        raise IntegrationError("이미 확정된 발화입니다. 수정 이벤트를 사용하세요")
                    if partial and revision < partial["payload"]["revision"]:
                        raise IntegrationError("확정 전사 revision이 최신 부분 전사보다 이전입니다")
                    state["partials"].pop(uid, None)
                    state["transcripts"][uid] = item
                    state["events"][eid] = item
                else:
                    if not current or payload["replaces_event_id"] != current["event_id"]:
                        raise IntegrationError("수정 대상은 현재 확정 전사여야 합니다")
                    if revision <= current["payload"]["revision"]:
                        raise IntegrationError("수정 전사 revision은 증가해야 합니다")
                    if (event["start_ms"], event["end_ms"]) != (current["start_ms"], current["end_ms"]):
                        raise IntegrationError("수정 전사의 시간 구간은 유지해야 합니다")
                    del state["events"][current["event_id"]]
                    state["transcripts"][uid] = item
                    state["events"][eid] = item
            else:
                if kind in {"resource_suggestion", "post_session_feedback"}:
                    if any(ref not in state["events"] for ref in payload["evidence_event_ids"]):
                        raise IntegrationError("근거 이벤트가 현재 세션에 없습니다")
                if kind == "note_created" and any(ref not in state["events"] for ref in payload["linked_event_ids"]):
                    raise IntegrationError("연결할 이벤트가 현재 세션에 없습니다")
                state["events"][eid] = item
            state["seen"].add(eid)
            return True

    def get_dashboard_snapshot(self, session_id):
        with self._lock:
            state = self._get(session_id)
            events = _ordered(list(state["events"].values()))
            active_ids = set(state["events"])
            statuses = {"model": "unavailable"} if state["config"].get("participation_model") is None else {}
            # 도착 순서상 마지막 상태가 현재 상태다.
            for event in state["events"].values():
                if event["event_type"] == "analysis_status":
                    statuses[event["payload"]["component"]] = event["payload"]["state"]
            def select(kind):
                return [e for e in events if e["event_type"] == kind]
            def with_evidence_flag(event):
                result = deepcopy(event)
                result["evidence_needs_refresh"] = any(ref not in active_ids for ref in event["payload"]["evidence_event_ids"])
                return result
            session = deepcopy(state["session"])
            session["duration_ms"] = self.elapsed_ms(session_id)
            session["status"] = "running" if state["running"] else "stopped"
            return deepcopy({
                "session": session, "status": statuses,
                "partial_transcripts": _ordered(list(state["partials"].values())),
                "transcripts": _ordered(list(state["transcripts"].values())),
                "engagement_windows": select("engagement_window"), "notes": select("note_created"),
                "suggestions": [with_evidence_flag(e) for e in select("resource_suggestion")],
                "feedback": [with_evidence_flag(e) for e in select("post_session_feedback")],
            })

    def get_events_in_range(self, session_id, start_ms, end_ms):
        """[start, end)와 겹치는 저장 대상 이벤트. 시점 이벤트는 해당 구간 안이면 포함."""
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start_ms, end_ms)) or start_ms < 0 or end_ms <= start_ms:
            raise ValueError("0 <= start_ms < end_ms인 정수 구간이 필요합니다")
        with self._lock:
            events = self._get(session_id)["events"].values()
            selected = [e for e in events if (start_ms <= e["start_ms"] < end_ms if e["start_ms"] == e["end_ms"] else e["start_ms"] < end_ms and start_ms < e["end_ms"])]
            return deepcopy(_ordered(selected))

    def stop_session(self, session_id):
        with self._lock:
            state = self._get(session_id)
            if state["running"]:
                state["session"]["duration_ms"] = self.elapsed_ms(session_id)
                state["running"] = False
                state["started_ns"] = None
                state["partials"].clear()
            return deepcopy(state["session"])

    def save_session(self, session_id, file_path):
        """현재 스냅샷을 같은 디렉터리의 임시 파일을 거쳐 JSON으로 교체한다."""
        path = Path(file_path)
        with self._lock:
            state = self._get(session_id)
            session = deepcopy(state["session"])
            session["duration_ms"] = self.elapsed_ms(session_id)
            # 수신 순서를 보존하여 최근 상태를 복원한다. 타임라인 표시는 별도 정렬한다.
            data = {"schema_version": SCHEMA_VERSION, "session": session,
                    "config": deepcopy(state["config"]), "events": list(deepcopy(state["events"]).values())}
            encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, prefix=".session-", suffix=".tmp", delete=False) as handle:
                    temporary = handle.name
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                temporary = None
            finally:
                if temporary is not None:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
        return str(path)

    def load_session(self, file_path):
        """세션을 종료된 상태로 불러온다. 실시간 캡처의 재개는 지원하지 않는다."""
        with open(file_path, encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise IntegrationError("지원하지 않는 저장 스키마입니다")
        session, config, events = data.get("session"), data.get("config"), data.get("events")
        if not isinstance(session, dict) or not isinstance(config, dict) or not isinstance(events, list):
            raise IntegrationError("세션 파일 형식이 올바르지 않습니다")
        try:
            sid = session["session_id"]
            UUID(sid)
            duration = session["duration_ms"]
            if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
                raise ValueError
            if not isinstance(session["title"], str) or not isinstance(session["started_at_utc"], str):
                raise ValueError
            if not isinstance(config.get("config_version"), str) or not config["config_version"]:
                raise ValueError
        except (KeyError, ValueError, TypeError, AttributeError):
            raise IntegrationError("세션 메타데이터가 올바르지 않습니다") from None
        state = {"session": deepcopy(session), "config": deepcopy(config), "started_ns": None,
                 "running": False, "events": {}, "seen": set(), "partials": {}, "transcripts": {}}
        for event in events:
            validate_event(event, sid)
            eid, kind = event["event_id"], event["event_type"]
            if eid in state["seen"] or kind == "transcript_partial":
                raise IntegrationError("중복 이벤트 또는 저장할 수 없는 부분 전사가 있습니다")
            if event["end_ms"] > duration:
                raise IntegrationError("이벤트가 저장된 수업 길이를 초과합니다")
            if kind in {"transcript_final", "transcript_edited"}:
                uid = event["payload"]["utterance_id"]
                if uid in state["transcripts"]:
                    raise IntegrationError("한 발화에 여러 확정 전사가 있습니다")
                state["transcripts"][uid] = deepcopy(event)
            state["events"][eid] = deepcopy(event)
            state["seen"].add(eid)
        with self._lock:
            if sid in self._sessions:
                raise IntegrationError("같은 ID의 세션이 이미 메모리에 있습니다")
            self._sessions[sid] = state
        return sid
