"""강의 자료에서 음성 전사 힌트로 쓸 전문용어를 고른다. 외부 전송이 없다.

여러 쪽에 반복해서 나오는 영문 식별자(`onClick`, `AppCompatActivity`)와 한글 명사
(`인터페이스`, `추상 클래스`의 `추상`)를 뽑는다. 조사·서술어·일반 단어는 버린다.
품사 분석기 없이 규칙으로 고르므로 일반 단어가 섞이거나 용어가 빠질 수 있다.
"""
from collections import Counter
import re

from .retrieval import _is_noise

_KOREAN = re.compile(r"[가-힣]{2,}")
_ENGLISH = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*")
# 단어 끝에 붙는 조사(긴 것부터 확인)
_PARTICLES = tuple(sorted((
    "에서는", "으로는", "이라는", "에게는", "이란", "라는", "에서", "으로", "에게", "까지", "부터", "보다", "처럼",
    "이며", "이고", "에는", "와는", "과는", "들은", "들이", "들을", "들의",
    "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만", "들",
), key=len, reverse=True))
# 자료에 자주 나오지만 전사 힌트로 쓸모없는 일반 단어
_GENERIC = {
    "경우", "사용", "다음", "방법", "내용", "예제", "설명", "정리", "참고", "문제", "결과", "부분", "이유",
    "의미", "기능", "정의", "종류", "특징", "구조", "형태", "과정", "단계", "목표", "개요", "요약", "실습",
    "과제", "주차", "강의", "수업", "학습", "이번", "지난", "다른", "모든", "각각", "하나", "둘째", "첫째",
    "때문", "통해", "위해", "대한", "관련", "가능", "필요", "중요", "기본", "일반", "여러", "자신", "아래",
    "위의", "다음과", "그림", "출처", "페이지", "슬라이드", "교수", "학생",
    "에서", "에게", "간의", "따라", "또는", "및", "등의", "같이", "확인", "생성", "설치", "명령", "입력",
    "출력", "선택", "추가", "변경", "작성", "이용", "표시", "해당", "다양", "간단", "직접", "현재",
}
# 명사형 서술 끝('나타냄', '사용함', '저장됨')은 용어가 아니다.
_NOMINAL_ENDINGS = ("냄", "함", "됨", "임", "음", "짐", "킴")
_ENGLISH_STOP = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "you", "not", "can", "has", "have",
    "will", "use", "using", "used", "into", "our", "your", "its", "their", "than", "then", "also", "such",
    "page", "slide", "chapter", "example", "note", "see", "fig", "figure", "http", "https", "www", "com",
    "each", "other", "more", "all", "any", "only", "one", "two", "when", "which", "what", "how", "who",
    "there", "these", "those", "they", "them", "some", "same", "each", "both", "per", "via", "but", "may",
    "users", "user", "cont", "continued",
}


def _korean_stem(word):
    for particle in _PARTICLES:
        if word.endswith(particle) and len(word) - len(particle) >= 2:
            return word[: -len(particle)]
    return word


def _candidates(text):
    found = set()
    for word in _KOREAN.findall(text):
        stem = _korean_stem(word)
        if (2 <= len(stem) <= 8 and stem not in _GENERIC and not stem.endswith(_NOMINAL_ENDINGS)
                and not _is_noise(stem)):
            found.add(stem)
    for word in _ENGLISH.findall(text):
        if len(word) >= 3 and word.lower() not in _ENGLISH_STOP:
            found.add(word)
    return found


def lecture_terms(index, limit=20, min_pages=2, max_chars=150):
    """색인(`MaterialIndex`)의 자료에서 여러 쪽에 나오는 용어를 많이 나온 순서로 돌려준다.

    - 영문은 대소문자가 다른 표기를 하나로 묶고 가장 자주 쓰인 표기를 남긴다.
    - 쪽이 하나뿐인 자료는 `min_pages`를 1로 낮춘다.
    - 합친 길이가 `max_chars`를 넘지 않게 자른다(Whisper 힌트 길이 제한).
    """
    pages = list(getattr(index, "pages", []) or [])
    if not pages:
        return []
    needed = 1 if len(pages) == 1 else min_pages
    page_counts, spellings = Counter(), {}
    for page in pages:
        for term in _candidates(page.text or ""):
            key = term.lower() if term.isascii() else term
            page_counts[key] += 1
            spellings.setdefault(key, Counter())[term] += 1
    ranked = sorted((key for key, count in page_counts.items() if count >= needed),
                    key=lambda key: (-page_counts[key], -len(key), key))
    terms, length = [], 0
    for key in ranked:
        term = spellings[key].most_common(1)[0][0]
        # 이미 고른 용어에 포함되는 짧은 한글 조각('페이스' ⊂ '인터페이스')은 뺀다.
        if not term.isascii() and any(term in chosen and term != chosen for chosen in terms):
            continue
        extra = len(term) + (2 if terms else 0)
        if length + extra > max_chars:
            break
        terms.append(term)
        length += extra
        if len(terms) >= limit:
            break
    return terms
