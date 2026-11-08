"""강의 자료 페이지를 로컬에서 찾는 BM25 검색. 외부 전송이나 모델 다운로드가 없다.

한국어 조사·어미 변화에 덜 민감하도록 한글 단어는 글자 2-gram으로 나눠 색인한다.
"""
from collections import Counter
from dataclasses import dataclass
import math
import re

_WORD = re.compile(r"[가-힣]+|[a-zA-Z][a-zA-Z0-9+#.]*|\d+")
# 수업 말투에서 자주 나오지만 자료 내용과 무관한 말
STOPWORDS = {
    "그리고", "그래서", "그러면", "그런데", "이제", "여러분", "오늘", "이거", "저거", "그거",
    "이것", "저희", "우리", "지금", "이렇게", "그렇게", "어떻게", "있습니다", "합니다", "하는",
    "있는", "같은", "이런", "그런", "네", "자", "좀", "뭐", "음", "어", "아", "the", "and", "is",
}


# 서술어 어미로 끝나는 말('살펴보겠습니다', '해 보세요')은 자료 내용 단서가 아니다.
_VERB_ENDINGS = ("니다", "세요", "어요", "아요", "해요", "는데", "니까", "지요", "죠")


def _is_noise(word):
    if word in STOPWORDS or len(word) < 2:
        return True
    if re.fullmatch(r"[가-힣]+", word):
        if word.endswith(_VERB_ENDINGS):
            return True
        # 불용어에 조사만 붙은 말('오늘은', '여러분이')
        return any(word.startswith(stop) and len(word) - len(stop) <= 2 for stop in STOPWORDS if len(stop) >= 2)
    return False


def words(text):
    """불용어·서술어를 뺀 단어 목록(영문은 소문자)."""
    return [w for w in (w.lower() for w in _WORD.findall(text or "")) if not _is_noise(w)]


def tokens(text):
    """색인용 토큰. 한글 단어는 2-gram, 영문·숫자는 단어 그대로."""
    result = []
    for word in words(text):
        if re.fullmatch(r"[가-힣]+", word):
            result.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            result.append(word)
    return result


def shared_keywords(query_words, text, limit=5):
    """질의 단어 중 자료에 나온 부분. 조사가 붙은 말('스케줄링에서는')은 자료에 있는 가장 긴 앞부분('스케줄링')을 쓴다."""
    text = (text or "").lower()
    found = set()
    for word in query_words:
        for length in range(len(word), 1, -1):
            if word[:length] in text:
                found.add(word[:length])
                break
    return tuple(sorted(found, key=len, reverse=True)[:limit])


@dataclass(frozen=True)
class SearchHit:
    page: object  # MaterialPage
    score: float
    coverage: float  # 질의 토큰 중 페이지에 있는 비율(0~1)
    keywords: tuple  # 질의와 페이지에 함께 나온 단어(설명용)


class MaterialIndex:
    """여러 자료의 페이지를 하나의 BM25 색인으로 묶는다."""

    def __init__(self, materials=(), k1=1.5, b=0.75):
        self.materials = list(materials)
        self.pages = [page for material in self.materials for page in material.pages if page.text]
        self._k1, self._b = k1, b
        self._docs = [Counter(tokens(page.text)) for page in self.pages]
        self._lengths = [sum(doc.values()) for doc in self._docs]
        self._avg = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        frequency = Counter()
        for doc in self._docs:
            frequency.update(doc.keys())
        n = len(self._docs)
        self._idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    def __bool__(self):
        return bool(self.pages)

    @property
    def page_count(self):
        return sum(len(material.pages) for material in self.materials)

    def search(self, query, limit=3, min_score=2.0, min_coverage=0.15, min_shared=3):
        """질의와 관련 있는 페이지를 점수 순으로 돌려준다. 기준에 못 미치면 빈 목록."""
        query_terms = Counter(tokens(query))
        if not query_terms or not self._docs:
            return []
        query_words = set(words(query))
        hits = []
        for index, doc in enumerate(self._docs):
            shared = [term for term in query_terms if term in doc]
            if len(shared) < min_shared:
                continue
            length_norm = 1 - self._b + self._b * self._lengths[index] / (self._avg or 1)
            score = 0.0
            for term in shared:
                tf = doc[term]
                score += self._idf[term] * tf * (self._k1 + 1) / (tf + self._k1 * length_norm)
            coverage = len(shared) / len(query_terms)
            if score < min_score or coverage < min_coverage:
                continue
            page = self.pages[index]
            keywords = shared_keywords(query_words, page.text)
            hits.append(SearchHit(page, round(score, 3), round(coverage, 3), keywords))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]
