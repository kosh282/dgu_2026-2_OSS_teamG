"""강의 조교: 통합부의 확정 전사를 읽어 자료 제안과 사후 피드백 이벤트를 만든다.

입출력은 모두 `IntegrationService`의 공통 계약 이벤트다.
- 입력: `transcript_final`/`transcript_edited`, `note_created`, `engagement_window`, `resource_suggestion`
- 출력: `resource_suggestion`, `post_session_feedback`, `analysis_status`(component `assistant`)
제안·피드백에는 근거 전사 `event_id`와 자료 출처(문서·페이지)를 연결하고, 상태는 항상
`suggested`/`draft`로 만들어 최종 판단을 교수자에게 둔다.
"""
import threading

from integration.service import IntegrationError
from .writer import LocalWriter, section_keywords

COMPONENT = "assistant"
MODULE = "assistant"


def _model_ids(writer, result):
    if result.get("fallback"):
        return LocalWriter.model_id, LocalWriter.model_version
    return writer.model_id, writer.model_version


def _suggestion_payload(result, evidence):
    page = result["hit"].page
    return {
        "title": result["title"],
        "description": result["description"],
        "source": {"document_id": page.document_id, "document_name": page.document_name, "page": page.page},
        "evidence_event_ids": [event["event_id"] for event in evidence],
        "status": "suggested",
    }


def _evidence(transcripts, hit):
    """제안 근거 전사: 검색 키워드가 나온 문장. 없으면 검색에 쓴 문장 전체."""
    matched = [event for event in transcripts
               if any(keyword in event["payload"]["text"].lower() for keyword in hit.keywords)]
    return matched or list(transcripts)


class LiveAssistant:
    """수업 중 주기적으로 최근 확정 전사와 강의 자료를 비교해 자료 제안을 만든다.

    세션이 종료되면 스스로 멈춘다. 같은 자료 페이지는 `cooldown_ms` 동안 다시 제안하지 않는다.
    """

    def __init__(self, service, session_id, index, writer=None, *, interval_s=10.0,
                 window_ms=60_000, cooldown_ms=600_000):
        self.service = service
        self.session_id = session_id
        self.index = index
        self.writer = writer or LocalWriter()
        self.interval_s = interval_s
        self.window_ms = window_ms
        self.cooldown_ms = cooldown_ms
        self._last_end_ms = -1
        self._suggested = {}  # (document_id, page) -> 마지막 제안 구간 끝
        self._reported_error = None
        self._stop = threading.Event()
        self._thread = None
        self.last_error = None
        self.suggestion_count = 0

    # 상태 -----------------------------------------------------------------
    def _status(self, state, message, quality="valid"):
        try:
            now = self.service.elapsed_ms(self.session_id)
            event = self.service.make_event(
                self.session_id, "analysis_status", {"component": COMPONENT, "state": state, "message": message},
                now, now, MODULE, quality_status=quality,
                model_id=self.writer.model_id, model_version=self.writer.model_version)
            self.service.receive_event(event)
        except IntegrationError:
            pass  # 세션이 이미 끝났으면 상태를 남기지 않는다

    def _session_running(self):
        return self.service.get_dashboard_snapshot(self.session_id)["session"]["status"] == "running"

    # 실행 -----------------------------------------------------------------
    def start(self):
        if self._thread is not None:
            raise RuntimeError("이미 시작했습니다")
        if not self.index:
            self._status("stopped", "강의 자료가 없어 실시간 자료 제안을 하지 않습니다.")
            return False
        where = "Claude 작성(전사·자료 일부를 Anthropic API로 전송)" if self.writer.uses_external_service else "로컬 검색"
        self._status("running", f"강의 자료 {len(self.index.materials)}개·{self.index.page_count}쪽으로 실시간 자료 제안 중 ({where})")
        self._thread = threading.Thread(target=self._loop, name="lecture-assistant", daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            try:
                if not self._session_running():
                    return
                self.step()
            except Exception as error:  # 조교 오류가 수업 세션을 멈추지 않게 한다
                self.last_error = str(error)
                self._status("unavailable", f"자료 제안 중 오류: {error}", quality="not_observable")
                return

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        if self._thread is not None and self.last_error is None:
            try:
                if self._session_running():
                    self._status("stopped", f"실시간 자료 제안을 종료했습니다. 제안 {self.suggestion_count}건")
            except IntegrationError:
                pass

    def step(self):
        """새 확정 전사가 있으면 한 번 검색해 제안을 만든다. 만든 이벤트(없으면 None)를 돌려준다."""
        snapshot = self.service.get_dashboard_snapshot(self.session_id)
        transcripts = snapshot["transcripts"]
        newest = max((event["end_ms"] for event in transcripts), default=-1)
        if newest <= self._last_end_ms:
            return None
        self._last_end_ms = newest
        window = [event for event in transcripts if event["end_ms"] >= newest - self.window_ms]
        query = " ".join(event["payload"]["text"] for event in window)
        hits = [hit for hit in self.index.search(query)
                if newest - self._suggested.get((hit.page.document_id, hit.page.page), -10**12) > self.cooldown_ms]
        if not hits:
            return None
        result = self.writer.suggest([event["payload"]["text"] for event in window], hits)
        self._report_writer_error()
        if result is None:
            return None
        evidence = _evidence(window, result["hit"])
        model_id, model_version = _model_ids(self.writer, result)
        event = self.service.make_event(
            self.session_id, "resource_suggestion", _suggestion_payload(result, evidence),
            min(e["start_ms"] for e in evidence), max(e["end_ms"] for e in evidence), MODULE,
            model_id=model_id, model_version=model_version)
        try:
            self.service.receive_event(event)
        except IntegrationError:
            return None  # 생성 직후 세션이 종료된 경우
        page = result["hit"].page
        self._suggested[(page.document_id, page.page)] = newest
        self.suggestion_count += 1
        return event

    def _report_writer_error(self):
        error = self.writer.last_error
        if error != self._reported_error:
            self._reported_error = error
            if error:
                self._status("running", f"Claude 작성 실패, 로컬 검색 결과로 대체: {error}", quality="partial")
            else:
                self._status("running", "Claude 작성이 다시 동작합니다.")


# ---------------------------------------------------------------------------
# 사후 피드백


def _overlaps(event, start, end):
    return event["start_ms"] < end and event["end_ms"] > start or event["start_ms"] == event["end_ms"] and start <= event["start_ms"] < end


def _engagement_line(windows):
    """실험 점수 요약. 관측 불가를 0점으로 바꾸지 않는다."""
    if not windows:
        return "실험 참여 점수 기록이 없습니다."
    values = []
    unobservable = 0
    for event in windows:
        output = event["payload"].get("model_output") or {}
        value = output.get("value") if isinstance(output, dict) else None
        if event["quality"]["status"] == "not_observable" or not isinstance(value, (int, float)) or isinstance(value, bool):
            unobservable += 1
        else:
            values.append(float(value))
    if not values:
        return f"실험 참여 점수: 이 구간은 관측 불가({unobservable}개 구간)로 점수가 없습니다."
    average = sum(values) / len(values)
    return (f"실험 참여 점수(참고): 평균 {average:.0f}점, 점수 있는 구간 {len(values)}개·관측 불가 {unobservable}개. "
            "집중도·이해도를 뜻하지 않습니다.")


def _existing_assistant_feedback(snapshot):
    return [event for event in snapshot["feedback"] if event["provenance"]["module"] == MODULE]


def generate_post_session(service, session_id, index, writer=None, *, section_ms=300_000):
    """종료된(또는 진행 중인) 세션에 사후 자료 제안과 구간별 피드백을 추가한다.

    - 자료 제안이 없던 구간은 강의 자료에서 관련 페이지를 찾아 수업 시각에 맞춘 제안을 만든다.
    - 확정 전사가 있는 구간마다 피드백(`draft`)을, 마지막에 수업 전체 피드백을 만든다.
    같은 세션에 이미 조교 피드백이 있으면 중복을 막기 위해 새로 만들지 않는다.
    반환값: {"suggestions": n, "feedback": n, "skipped": 사유 또는 None, "writer_error": 오류 또는 None}
    """
    writer = writer or LocalWriter()
    snapshot = service.get_dashboard_snapshot(session_id)
    if _existing_assistant_feedback(snapshot):
        return {"suggestions": 0, "feedback": 0, "skipped": "이미 조교 피드백이 있습니다.", "writer_error": None}
    duration = snapshot["session"]["duration_ms"]
    transcripts = snapshot["transcripts"]
    if not transcripts:
        return {"suggestions": 0, "feedback": 0, "skipped": "확정 전사가 없어 피드백을 만들 수 없습니다.", "writer_error": None}
    has_materials = bool(index)
    created_suggestions = 0
    sections = []
    matched_pages = set()
    for start in range(0, max(duration, 1), section_ms):
        end = min(start + section_ms, duration) if duration else section_ms
        section_transcripts = [event for event in transcripts if start <= event["start_ms"] < end]
        if not section_transcripts:
            continue
        texts = [event["payload"]["text"] for event in section_transcripts]
        hits = index.search(" ".join(texts), limit=3) if has_materials else []
        matched_pages.update((hit.page.document_id, hit.page.page) for hit in hits)
        existing = [event for event in service.get_dashboard_snapshot(session_id)["suggestions"]
                    if _overlaps(event, start, end)]
        # 수업 중 제안이 없던 구간만 타임라인에 맞춘 사후 제안을 추가한다.
        if hits and not existing:
            result = writer.suggest(texts, hits)
            if result is not None:
                evidence = _evidence(section_transcripts, result["hit"])
                model_id, model_version = _model_ids(writer, result)
                event = service.make_event(
                    session_id, "resource_suggestion", _suggestion_payload(result, evidence),
                    min(e["start_ms"] for e in evidence), max(e["end_ms"] for e in evidence), MODULE,
                    model_id=model_id, model_version=model_version)
                service.receive_event(event)
                existing.append(event)
                created_suggestions += 1
        current = service.get_dashboard_snapshot(session_id)
        sections.append({
            "start_ms": start, "end_ms": end, "transcripts": section_transcripts,
            "keywords": section_keywords(texts), "pages": hits, "has_materials": has_materials,
            "notes": [e["payload"]["text"] for e in current["notes"] if start <= e["start_ms"] < end],
            "suggestions": [f"{e['payload']['source']['document_name']} {e['payload']['source']['page'] or '?'}쪽" for e in existing],
            "suggestion_ids": [e["event_id"] for e in existing],
            "engagement": _engagement_line([e for e in current["engagement_windows"] if _overlaps(e, start, end)]),
        })
    uncovered = [f"{page.document_name} {page.page}쪽" for page in (index.pages if has_materials else [])
                 if (page.document_id, page.page) not in matched_pages]
    overall = {"title": snapshot["session"]["title"], "duration_ms": duration,
               "transcript_count": len(transcripts), "note_count": len(snapshot["notes"]),
               "uncovered_pages": uncovered, "has_materials": has_materials}
    texts, overall_text = writer.feedback(sections, overall)
    writer_error = writer.last_error
    model_id = LocalWriter.model_id if writer_error else writer.model_id
    model_version = LocalWriter.model_version if writer_error else writer.model_version
    for section, text in zip(sections, texts):
        evidence = [e["event_id"] for e in section["transcripts"]] + section["suggestion_ids"]
        event = service.make_event(
            session_id, "post_session_feedback", {"text": text, "evidence_event_ids": evidence, "status": "draft"},
            section["start_ms"], section["end_ms"], MODULE, model_id=model_id, model_version=model_version)
        service.receive_event(event)
    overall_event = service.make_event(
        session_id, "post_session_feedback", {"text": overall_text, "evidence_event_ids": [], "status": "draft"},
        0, duration, MODULE, model_id=model_id, model_version=model_version)
    service.receive_event(overall_event)
    return {"suggestions": created_suggestions, "feedback": len(sections) + 1, "skipped": None, "writer_error": writer_error}
