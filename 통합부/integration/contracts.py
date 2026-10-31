"""공통 이벤트 검사. 모델의 세부 출력 형식은 팀 합의 전까지 열어 둔다."""
import json
import re
from datetime import datetime
from uuid import UUID

SCHEMA_VERSION = "0.1.0"
MODULES = {"capture", "vision", "model", "speech", "ui", "assistant", "system"}
QUALITY = {"valid", "partial", "not_observable"}
# 원본 입력·개인별 중간 데이터의 대표 키. 포괄적인 개인정보 탐지기는 아니다.
FORBIDDEN_KEYS = {"image", "frame", "frames", "samples", "audio", "video", "landmarks", "face_image", "track_id", "participant_id"}


class EventValidationError(ValueError):
    """이벤트의 필수 계약을 만족하지 못한 경우."""


def _fail(field, message):
    raise EventValidationError(f"{field}: {message}")


def _object(value, field):
    if not isinstance(value, dict):
        _fail(field, "객체여야 합니다")
    return value


def _string(value, field):
    if not isinstance(value, str) or not value.strip():
        _fail(field, "비어 있지 않은 문자열이어야 합니다")


def _uuid(value, field):
    _string(value, field)
    try:
        UUID(value)
    except (ValueError, AttributeError):
        _fail(field, "UUID 문자열이어야 합니다")


def _int(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(field, f"{minimum} 이상의 정수여야 합니다")


def _utc(value, field):
    _string(value, field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError
    except ValueError:
        _fail(field, "UTC ISO 8601 시각이어야 합니다")


def _id_list(value, field):
    if not isinstance(value, list):
        _fail(field, "배열이어야 합니다")
    for i, item in enumerate(value):
        _uuid(item, f"{field}[{i}]")


def _check_forbidden(value, field="event"):
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(field, "객체 키는 문자열이어야 합니다")
            if key.lower() in FORBIDDEN_KEYS:
                _fail(f"{field}.{key}", "원본 또는 개인별 중간 데이터 키는 허용하지 않습니다")
            _check_forbidden(child, f"{field}.{key}")
    elif isinstance(value, list):
        for child in value:
            _check_forbidden(child, field)


def validate_event(event, session_id):
    _object(event, "event")
    required = ("schema_version", "session_id", "event_id", "event_type", "start_ms", "end_ms", "created_ms", "created_at_utc", "payload", "quality", "provenance")
    for key in required:
        if key not in event:
            _fail(key, "필수 필드가 없습니다")
    version = event["schema_version"]
    if not isinstance(version, str) or not re.fullmatch(r"0\.\d+\.\d+", version):
        _fail("schema_version", "지원하는 0.x 버전이어야 합니다")
    _uuid(event["session_id"], "session_id")
    if event["session_id"] != session_id:
        _fail("session_id", "현재 세션과 다릅니다")
    _uuid(event["event_id"], "event_id")
    _string(event["event_type"], "event_type")
    _int(event["start_ms"], "start_ms")
    _int(event["end_ms"], "end_ms")
    if event["end_ms"] < event["start_ms"]:
        _fail("end_ms", "start_ms보다 작습니다")
    if event["created_ms"] is None:
        _utc(event["created_at_utc"], "created_at_utc")
    else:
        _int(event["created_ms"], "created_ms")
        if event["created_at_utc"] is not None:
            _utc(event["created_at_utc"], "created_at_utc")
    payload = _object(event["payload"], "payload")
    quality = _object(event["quality"], "quality")
    if quality.get("status") not in QUALITY:
        _fail("quality.status", "valid, partial, not_observable 중 하나여야 합니다")
    reasons = quality.get("reason_codes")
    if not isinstance(reasons, list) or any(not isinstance(x, str) or not x for x in reasons):
        _fail("quality.reason_codes", "문자열 배열이어야 합니다")
    provenance = _object(event["provenance"], "provenance")
    if provenance.get("module") not in MODULES:
        _fail("provenance.module", "알 수 없는 모듈입니다")
    _string(provenance.get("config_version"), "provenance.config_version")
    for key in ("model_id", "model_version"):
        if key not in provenance or (provenance[key] is not None and (not isinstance(provenance[key], str) or not provenance[key])):
            _fail(f"provenance.{key}", "문자열 또는 null이어야 합니다")
    kind = event["event_type"]
    if kind in {"transcript_partial", "transcript_final", "transcript_edited"}:
        _uuid(payload.get("utterance_id"), "payload.utterance_id")
        _int(payload.get("revision"), "payload.revision", 1)
        if not isinstance(payload.get("text"), str):
            _fail("payload.text", "문자열이어야 합니다")
        if kind != "transcript_partial" and payload.get("is_edited") is not (kind == "transcript_edited"):
            _fail("payload.is_edited", "전사 유형과 일치하는 불리언이어야 합니다")
        if kind == "transcript_edited":
            _uuid(payload.get("replaces_event_id"), "payload.replaces_event_id")
    elif kind == "analysis_status":
        _string(payload.get("component"), "payload.component")
        if payload.get("state") not in {"ready", "running", "unavailable", "stopped"}:
            _fail("payload.state", "허용되지 않은 상태입니다")
        if not isinstance(payload.get("message"), str):
            _fail("payload.message", "문자열이어야 합니다")
    elif kind == "engagement_window":
        if provenance["model_id"] is None or provenance["model_version"] is None:
            _fail("provenance", "참여 모델 ID와 버전이 필요합니다")
        for key in ("model_output", "observation", "signals"):
            if key not in payload:
                _fail(f"payload.{key}", "필수 필드가 없습니다")
        if payload["model_output"] is not None:
            _object(payload["model_output"], "payload.model_output")
        obs = _object(payload["observation"], "payload.observation")
        for key in ("valid_count", "eligible_count", "observable_ratio", "count_unit"):
            if key not in obs:
                _fail(f"payload.observation.{key}", "필수 필드가 없습니다")
        for key in ("valid_count", "eligible_count"):
            if obs[key] is not None:
                _int(obs[key], f"payload.observation.{key}")
        ratio = obs["observable_ratio"]
        if ratio is not None and (isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1):
            _fail("payload.observation.observable_ratio", "0~1 또는 null이어야 합니다")
        valid_count, eligible_count = obs["valid_count"], obs["eligible_count"]
        if eligible_count is None and (valid_count is not None or ratio is not None):
            _fail("payload.observation", "분모가 미정이면 수치와 비율은 모두 null이어야 합니다")
        if eligible_count == 0 and ratio is not None:
            _fail("payload.observation.observable_ratio", "분모가 0이면 null이어야 합니다")
        if valid_count is not None and eligible_count is not None and valid_count > eligible_count:
            _fail("payload.observation.valid_count", "유효 관측 수가 전체 관측 수보다 클 수 없습니다")
        if obs["count_unit"] is not None:
            _string(obs["count_unit"], "payload.observation.count_unit")
        _object(payload["signals"], "payload.signals")
        if "model_extensions" in payload:
            _object(payload["model_extensions"], "payload.model_extensions")
    elif kind == "note_created":
        if not isinstance(payload.get("text"), str):
            _fail("payload.text", "문자열이어야 합니다")
        _id_list(payload.get("linked_event_ids"), "payload.linked_event_ids")
    elif kind in {"resource_suggestion", "post_session_feedback"}:
        _id_list(payload.get("evidence_event_ids"), "payload.evidence_event_ids")
        if kind == "resource_suggestion":
            _string(payload.get("title"), "payload.title")
            if not isinstance(payload.get("description"), str):
                _fail("payload.description", "문자열이어야 합니다")
            source = _object(payload.get("source"), "payload.source")
            _uuid(source.get("document_id"), "payload.source.document_id")
            _string(source.get("document_name"), "payload.source.document_name")
            if "page" not in source:
                _fail("payload.source.page", "필수 필드가 없습니다")
            if source["page"] is not None:
                _int(source["page"], "payload.source.page", 1)
            if payload.get("status") not in {"suggested", "accepted", "dismissed"}:
                _fail("payload.status", "허용되지 않은 제안 상태입니다")
        else:
            if not isinstance(payload.get("text"), str):
                _fail("payload.text", "문자열이어야 합니다")
            if payload.get("status") not in {"draft", "reviewed"}:
                _fail("payload.status", "허용되지 않은 피드백 상태입니다")
    _check_forbidden(event)
    try:
        json.dumps(event, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        _fail("event", "유효한 JSON으로 직렬화할 수 없습니다")
