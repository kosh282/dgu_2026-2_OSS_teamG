"""자료 제안·사후 피드백 문장 작성기.

- `LocalWriter`: 외부 전송 없이 검색 결과와 수업 기록으로 템플릿 문장을 만든다(기본).
- `ClaudeWriter`: 교수자가 켠 경우에만 Claude API로 문장을 작성한다. 전송 범위는
  해당 요청에 필요한 확정 전사·메모·관련 자료 페이지 텍스트로 제한하며 원본 영상·음성,
  얼굴 특징은 보내지 않는다. 호출이 실패하면 `LocalWriter` 결과로 대체하고 오류를 남긴다.
"""
import json
import threading

from .retrieval import words

LOCAL_MODEL_ID = "local-bm25-ko"
LOCAL_MODEL_VERSION = "0.1.0"
CLAUDE_MODEL = "claude-opus-5-5"
PROMPT_VERSION = "prompt-v1"


def _clock(ms):
    seconds = int(ms // 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _snippet(text, limit=60):
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return line if len(line) <= limit else line[:limit] + "…"


class LocalWriter:
    model_id = LOCAL_MODEL_ID
    model_version = LOCAL_MODEL_VERSION
    uses_external_service = False
    last_error = None

    def suggest(self, evidence_texts, hits):
        """가장 점수가 높은 후보로 제안 문장을 만든다. 후보가 없으면 None."""
        if not hits:
            return None
        hit = hits[0]
        keywords = ", ".join(f"'{word}'" for word in hit.keywords[:3]) or "관련 표현"
        title = f"{hit.page.document_name} {hit.page.page}쪽"
        description = (f"최근 설명의 {keywords} 내용이 자료 {hit.page.page}쪽과 겹칩니다. "
                       f"페이지 첫 줄: {_snippet(hit.page.text)}")
        return {"hit": hit, "title": title, "description": description}

    def feedback(self, sections, overall):
        """구간별 피드백 문장 목록과 수업 전체 문장을 돌려준다."""
        return [self._section_text(section) for section in sections], self._overall_text(overall)

    @staticmethod
    def _section_text(section):
        parts = [f"{_clock(section['start_ms'])}–{_clock(section['end_ms'])} 구간, 확정 전사 {len(section['transcripts'])}문장."]
        top = [word for word, _ in section["keywords"][:4]]
        if top:
            parts.append("주요 표현: " + ", ".join(top) + ".")
        if section["pages"]:
            pages = ", ".join(f"{hit.page.document_name} {hit.page.page}쪽" for hit in section["pages"])
            parts.append(f"관련 자료: {pages}. 수업 뒤 자료와 설명 순서가 맞았는지 확인해 보세요.")
        elif section["has_materials"]:
            parts.append("올린 자료에서 이 설명과 뚜렷하게 겹치는 페이지를 찾지 못했습니다. 자료 밖 설명이었다면 보충 자료를 고려할 수 있습니다.")
        if section["suggestions"]:
            parts.append("이 구간의 자료 제안: " + ", ".join(section["suggestions"]) + ".")
        if section["notes"]:
            parts.append("교수자 메모: " + " / ".join(section["notes"]) + ".")
        parts.append(section["engagement"])
        return " ".join(parts)

    @staticmethod
    def _overall_text(overall):
        parts = [f"수업 {_clock(overall['duration_ms'])}, 확정 전사 {overall['transcript_count']}문장, 메모 {overall['note_count']}개."]
        if overall["uncovered_pages"]:
            pages = ", ".join(overall["uncovered_pages"][:10])
            more = f" 외 {len(overall['uncovered_pages']) - 10}쪽" if len(overall["uncovered_pages"]) > 10 else ""
            parts.append(f"전사와 겹치는 설명을 찾지 못한 자료 페이지: {pages}{more}. 다음 수업에서 다룰지 확인해 보세요.")
        elif overall["has_materials"]:
            parts.append("올린 자료의 텍스트 페이지는 모두 수업 설명과 겹치는 부분이 있었습니다.")
        parts.append("이 피드백은 전사·메모·자료 검색 결과를 모은 참고 자료이며, 최종 판단은 교수자에게 있습니다.")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Claude

_SUGGEST_SYSTEM = """You assist a university lecturer during a live online lecture.
You receive the most recent confirmed transcript lines of the lecturer's speech (Korean, from a
speech recognizer, so expect recognition errors) and up to three candidate pages from the
lecturer's own uploaded lecture materials.

Pick the one candidate page that the lecturer could refer to or show right now because it is
clearly about what is being explained. If no candidate is clearly related, answer choice -1.
Write the title (at most 40 characters) and a one-sentence description in Korean. The description
explains the connection using only content that appears in the transcript and the chosen page.
Do not evaluate the lecturer or students. The lecturer decides whether to use the suggestion."""

_SUGGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "integer", "description": "0-based candidate index, or -1 for none"},
        "title": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["choice", "title", "description"],
    "additionalProperties": False,
}

_FEEDBACK_SYSTEM = """You help a university lecturer reflect on their own online lecture after class.
You receive the lecture split into time sections. Each section has the confirmed transcript of
the lecturer's speech (Korean speech recognition, expect recognition errors), the lecturer's own
notes, matched pages from the lecturer's uploaded materials, material suggestions shown during the
lecture, and an engagement line.

Write feedback in Korean for each section (2 to 4 sentences) and one overall paragraph.
- Base every statement only on the provided content. Do not invent topics or student reactions.
- Engagement values come from an experimental single-face video model. They are reference values
  only: never infer students' attention, understanding, grades or emotions from them, and treat a
  missing or unobservable value as "not observable", not as low engagement.
- Do not name, identify or evaluate individual students.
- Offer concrete, optional ideas (e.g. revisit a material page, add an example, reorder
  explanations). The lecturer makes the final decision."""

_FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"section_index": {"type": "integer"}, "feedback": {"type": "string"}},
                "required": ["section_index", "feedback"],
                "additionalProperties": False,
            },
        },
        "overall": {"type": "string"},
    },
    "required": ["sections", "overall"],
    "additionalProperties": False,
}


class ClaudeUnavailable(RuntimeError):
    """Claude 호출 실패. 호출 측은 로컬 결과로 대체한다."""


def claude_ready():
    """anthropic 패키지 설치 여부. 자격 증명은 실제 호출에서 확인한다."""
    import importlib.util
    return importlib.util.find_spec("anthropic") is not None


class ClaudeWriter:
    """Claude가 문장을 쓰고, 실패하면 로컬 문장으로 대체한다."""

    model_id = CLAUDE_MODEL
    model_version = PROMPT_VERSION
    uses_external_service = True

    def __init__(self, client=None, local=None, model=CLAUDE_MODEL):
        self._client = client
        self._local = local or LocalWriter()
        self._model = model
        self._lock = threading.Lock()
        self.last_error = None

    def _get_client(self):
        with self._lock:
            if self._client is None:
                try:
                    import anthropic
                except ImportError as error:
                    raise ClaudeUnavailable("anthropic 패키지가 없습니다. ai_agent/requirements.txt를 설치하세요") from error
                # 자격 증명은 ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, `ant auth login` 프로필 순으로 찾는다.
                self._client = anthropic.Anthropic(timeout=60.0)
            return self._client

    def _ask(self, system, content, schema, effort):
        import anthropic
        try:
            response = self._get_client().beta.messages.create(
                model=self._model,
                max_tokens=16000,
                # 안전 분류기가 거절하면 서버가 대체 모델로 다시 실행한다.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                system=system,
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.AuthenticationError as error:
            raise ClaudeUnavailable("Claude 인증 실패: API 키 또는 로그인 정보를 확인하세요") from error
        except anthropic.RateLimitError as error:
            raise ClaudeUnavailable("Claude 사용량 한도에 걸렸습니다. 잠시 뒤 다시 시도하세요") from error
        except anthropic.APIStatusError as error:
            raise ClaudeUnavailable(f"Claude API 오류 {error.status_code}: {error.message}") from error
        except anthropic.APIConnectionError as error:
            raise ClaudeUnavailable("Claude API에 연결할 수 없습니다(네트워크 확인)") from error
        except anthropic.AnthropicError as error:  # 자격 증명 없음 등
            raise ClaudeUnavailable(f"Claude를 사용할 수 없습니다: {error}") from error
        if response.stop_reason == "refusal":
            raise ClaudeUnavailable("Claude가 요청을 거절했습니다")
        if response.stop_reason == "max_tokens":
            raise ClaudeUnavailable("Claude 응답이 길이 한도에서 잘렸습니다")
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise ClaudeUnavailable("Claude 응답에 텍스트가 없습니다")
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise ClaudeUnavailable("Claude 응답을 해석할 수 없습니다") from error

    def suggest(self, evidence_texts, hits):
        if not hits:
            return None
        candidates = "\n".join(
            f'<candidate index="{i}" document="{hit.page.document_name}" page="{hit.page.page}">\n{hit.page.text}\n</candidate>'
            for i, hit in enumerate(hits))
        transcript = "\n".join(evidence_texts)
        content = f"<transcript>\n{transcript}\n</transcript>\n<candidates>\n{candidates}\n</candidates>"
        try:
            data = self._ask(_SUGGEST_SYSTEM, content, _SUGGEST_SCHEMA, effort="low")
        except ClaudeUnavailable as error:
            self.last_error = str(error)
            return self._local.suggest(evidence_texts, hits) | {"fallback": True}
        self.last_error = None
        choice = data.get("choice")
        if not isinstance(choice, int) or not 0 <= choice < len(hits):
            return None  # Claude가 관련 없는 후보라고 판단
        title = (data.get("title") or "").strip() or f"{hits[choice].page.document_name} {hits[choice].page.page}쪽"
        return {"hit": hits[choice], "title": title[:80], "description": (data.get("description") or "").strip()}

    def feedback(self, sections, overall):
        local_sections, local_overall = self._local.feedback(sections, overall)
        blocks = []
        for i, section in enumerate(sections):
            pages = "\n".join(f'<page document="{hit.page.document_name}" page="{hit.page.page}">\n{hit.page.text}\n</page>'
                              for hit in section["pages"])
            transcript = "\n".join(f"[{_clock(event['start_ms'])}] {event['payload']['text']}" for event in section["transcripts"])
            blocks.append(
                f'<section index="{i}" start="{_clock(section["start_ms"])}" end="{_clock(section["end_ms"])}">\n'
                f"<transcript>\n{transcript}\n</transcript>\n"
                f"<notes>{' / '.join(section['notes'])}</notes>\n"
                f"<suggestions>{', '.join(section['suggestions'])}</suggestions>\n"
                f"<engagement>{section['engagement']}</engagement>\n"
                f"<material_pages>\n{pages}\n</material_pages>\n</section>")
        uncovered = ", ".join(overall["uncovered_pages"]) or "없음"
        content = (f"<lecture title=\"{overall['title']}\" duration=\"{_clock(overall['duration_ms'])}\">\n"
                   + "\n".join(blocks)
                   + f"\n<material_pages_not_matched>{uncovered}</material_pages_not_matched>\n</lecture>")
        try:
            data = self._ask(_FEEDBACK_SYSTEM, content, _FEEDBACK_SCHEMA, effort="medium")
        except ClaudeUnavailable as error:
            self.last_error = str(error)
            return local_sections, local_overall
        self.last_error = None
        texts = list(local_sections)
        for item in data.get("sections", []):
            index, text = item.get("section_index"), (item.get("feedback") or "").strip()
            if isinstance(index, int) and 0 <= index < len(texts) and text:
                texts[index] = text
        return texts, (data.get("overall") or "").strip() or local_overall


def make_writer(use_claude):
    return ClaudeWriter() if use_claude else LocalWriter()


def section_keywords(texts, limit=6):
    """구간에서 자주 나온 단어(설명용)."""
    from collections import Counter
    counter = Counter(word for text in texts for word in words(text) if len(word) >= 2)
    return counter.most_common(limit)
