# 교수자 라이브 세션 — Streamlit 프로토타입

Figma의 `1. 라이브 세션 (회의 옆 패널 · 화면 1/3)` 화면을 구현한 실행 가능한 프런트엔드야.

## 실행

Python 3.10 이상에서:

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에 표시되는 로컬 주소(일반적으로 `http://localhost:8501`)를 열면 돼.

## 동작하는 기능

- 큰 참여 상태 카드: 좋음(70~100), 보통(40~69), 나쁨(0~39), 분석 보류(값 없음)
- 점수 타임라인 및 관찰 신호 레인, 메모 마커
- 세션 타이머 시작/일시정지 및 새로고침
- 교수자 메모 작성·타임라인 반영
- 강의 자료 제안 표시, 메모로 저장 및 숨기기
- 이벤트 JSON 불러오기, 세션 JSON 내보내기

## 현재 제한

- 기본 표시 내용은 Figma의 예시 데이터를 바탕으로 한 데모야.
- 타이머를 시작해도 실제 영상 분석이나 STT가 자동 실행되지는 않아.
- JSON 불러오기는 `events` 배열의 `engagement_window`, `note_created`, `resource_suggestion`을 처리해. 참여 점수는 `payload.score` 또는 `payload.engagement_score`에서 읽어. 실제 팀의 세부 payload 키가 다르면 `ingest()`를 조정해야 해.
- 참여 점수 `null`은 **분석 보류**로 표시하며 0점으로 대체하지 않아.
- 예시 그래프의 관찰 신호 레인과 변화 알림 구간은 정적 목업이야. JSON에서 실시간으로 갱신하려면 추가 이벤트 필드 매핑이 필요해.
- 로컬 JSON 내보내기는 프로토타입용이며, 팀의 최종 통합 JSON 명세와 완전히 동일하다고 보장하지 않아.

## 파일

- `app.py`: Streamlit 화면 및 데모 이벤트 처리
- `requirements.txt`: 실행 패키지
- `sample_events.json`: 연결 테스트용 JSON
