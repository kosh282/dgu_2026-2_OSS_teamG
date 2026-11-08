# 강의 조교 (AI 에이전트)

교수자가 올린 강의 자료(PDF·PPTX)와 확정 전사를 비교해 **실시간 자료 제안**과 **수업 후 피드백**을 만드는 모듈입니다. 모든 입출력은 통합부(`IntegrationService`)의 공통 계약 이벤트로 주고받습니다. 제안과 피드백은 참고용 초안이며, 최종 판단은 교수자가 합니다.

## 구성

| 파일 | 역할 |
|---|---|
| `materials.py` | PDF(pypdf)·PPTX(python-pptx, 슬라이드 노트 포함)를 페이지 단위 텍스트로 읽음. 구형 `.ppt`, 암호 PDF, 스캔 이미지 PDF(텍스트 없음)는 읽지 못함 |
| `retrieval.py` | 로컬 BM25 검색. 한글은 글자 2-gram으로 색인해 조사 변화(`스케줄링을/에서는`)에 덜 민감함. 수업 말투(불용어, `~습니다` 서술어)는 제외 |
| `writer.py` | 문장 작성. `LocalWriter`(기본, 외부 전송 없음)와 `ClaudeWriter`(선택, Claude API) |
| `assistant.py` | `LiveAssistant`(수업 중 주기 실행)와 `generate_post_session`(수업 후 실행) |

```
STT ─ transcript_final ─▶ IntegrationService ─▶ LiveAssistant ──┐
                                 ▲                 (자료 색인 검색) │
대시보드 ◀─ 스냅샷 ─────────────┤◀─ resource_suggestion ───────────┘
                                 │◀─ post_session_feedback ── generate_post_session(사후 검토)
```

## 이벤트 계약

새 이벤트 유형이나 필드를 추가하지 않고 계약 v0.1.0의 이벤트를 그대로 사용합니다.

- `resource_suggestion`: `title`, `description`, `source`(`document_id`, `document_name`, `page`), `evidence_event_ids`(근거 확정 전사), `status: "suggested"`. 이벤트 구간은 근거 전사 구간입니다.
- `post_session_feedback`: `text`, `evidence_event_ids`(구간 전사와 그 구간의 자료 제안), `status: "draft"`. 구간별(기본 5분) 피드백과 수업 전체 피드백 1건(근거 없음, 구간 `0~duration_ms`)을 만듭니다.
- `analysis_status`: `component: "assistant"`로 실행·대체·종료 상태를 알립니다.
- `provenance.module`은 `assistant`입니다. `model_id/model_version`은 로컬이면 `local-bm25-ko`/`0.1.0`, Claude면 `claude-opus-5-5`/`prompt-v1`입니다. Claude 호출이 실패해 로컬 문장을 쓴 이벤트는 로컬 값으로 기록합니다.
- `document_id`는 파일 내용 해시로 만든 UUID입니다. 같은 자료를 수업 시작과 사후 검토에서 다시 올려도 같은 ID가 됩니다.

## 동작

**실시간 자료 제안 (`LiveAssistant`)**
- 10초마다 새 확정 전사가 있으면 최근 60초 전사로 자료를 검색합니다. 기준 점수 이상인 페이지가 있으면 제안합니다.
- 같은 페이지는 10분 동안 다시 제안하지 않습니다. 근거는 검색 키워드가 나온 전사 문장입니다.
- 세션이 종료되면 스스로 멈춥니다. 조교 오류는 `assistant: unavailable`로 알리며, 수업 세션과 다른 모듈은 계속 동작합니다.

**사후 피드백·자료 제안 (`generate_post_session`)**
- 수업 중 제안이 없던 구간은 그 구간 전사로 자료를 검색해 **수업 시각에 맞춘 자료 제안**을 추가합니다.
- 구간 피드백에는 다음 내용을 정리합니다.
  - 주요 표현, 관련 자료 페이지, 그 구간의 자료 제안
  - 교수자 메모
  - 실험 참여 점수 요약. 관측 불가를 0점으로 바꾸지 않고, 집중도·이해도로 해석하지 않습니다.
- 전체 피드백은 전사와 겹치는 설명을 찾지 못한 자료 페이지를 알려 줍니다.
- 이미 조교 피드백이 있는 세션은 중복 생성하지 않습니다. 결과는 세션 JSON에 저장되어 다시 열 때 복원됩니다.

## Claude 사용 (선택)

대시보드에서 **Claude로 제안·피드백 문장 작성**을 켠 경우에만 사용합니다. 기본값은 꺼짐입니다.

- **모델:** `claude-opus-5-5`를 구조화 출력(JSON schema)으로 호출합니다.
- **effort:** 실시간 제안은 `low`, 사후 피드백은 `medium`입니다.
- **거절 대비:** `server-side-fallback-2026-07-01` beta로 서버 측 대체 모델(`fallbacks: "default"`)을 켭니다. 안전 분류기가 요청을 거절하면 서버가 다른 모델로 다시 실행합니다.
- **실시간 제안:** 로컬 검색으로 후보 페이지를 최대 3개 고르고, Claude는 그중 하나를 고르거나 "관련 없음"으로 판단해 설명 문장을 씁니다.
- **전송 범위:** 확정 전사(실시간은 최근 60초), 교수자 메모, 후보 또는 관련 자료 페이지 텍스트, 실험 점수 요약 문장입니다. 원본 영상·음성, 얼굴 특징, 랜드마크는 보내지 않습니다. 전사에는 수업 중 언급된 이름 같은 개인정보가 들어 있을 수 있으므로, 동의 범위를 확인한 뒤 켜야 합니다.
- **자격 증명:** `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, 또는 `ant auth login` 프로필 순서로 찾습니다.
- **실패 처리:** 패키지 없음, 인증·네트워크 오류, 거절이 생기면 로컬 결과로 대체하고 상태 메시지와 화면 경고로 알립니다.

## 설치와 테스트

```powershell
video_recognition/.venv/Scripts/python.exe -m pip install -r ai_agent/requirements.txt
video_recognition/.venv/Scripts/python.exe -m unittest ai_agent.tests.test_assistant
```

테스트는 Claude를 가짜 클라이언트로 대신해 외부 API를 호출하지 않습니다.

## 검증 상태와 한계

- **확인한 것**
  - 단위 테스트 16개를 실행했습니다. PDF·PPTX 파싱, 검색, 근거·출처 연결, 쿨다운, 사후 피드백·저장·재열기, Claude 응답 처리와 실패 시 대체를 다룹니다.
  - sherpa 한국어 샘플 음성을 실시간 속도로 STT에 넣고 PPTX 자료로 실시간 제안을 실행했습니다. 관련 문장 2개에 맞는 페이지가 제안됐고, 근거 발화가 끝난 뒤 제안까지 약 1.5초 걸렸습니다. 무관한 문장 2개에는 제안이 없었습니다.
- **확인하지 않은 것**
  - **실제 Claude API 호출은 검증하지 않았습니다.** 이 PC에 자격 증명이 없어 가짜 클라이언트로만 시험했습니다.
  - 실제 수업 자료·수업 음성에서의 제안 정확도도 아직 평가하지 않았습니다.
- **한계**
  - 키워드 기반 검색이라 같은 뜻의 다른 표현(동의어, 영어↔한국어)은 놓칠 수 있습니다.
  - 검색 기준값(`min_score`, `min_coverage`)은 임시값입니다.
  - 자료는 브라우저 세션 메모리에만 있습니다. 앱을 다시 시작하면 사후 검토 화면에서 자료를 다시 올려야 합니다.
  - 제안의 `accepted`/`dismissed` 상태 변경은 아직 계약 이벤트로 기록하지 않습니다. 화면의 숨김·메모 저장만 지원합니다.
