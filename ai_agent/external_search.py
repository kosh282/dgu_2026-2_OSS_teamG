"""선택한 AI 검색어로 한국어 위키백과의 공개 검색 결과를 조회한다."""
import json
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def search_reference(query, limit=3):
    """검색 실패는 빈 결과로 처리한다. 원문 페이지는 가져오지 않는다."""
    if not query or not query.strip():
        return []
    params = urlencode({'action': 'query', 'list': 'search', 'srsearch': query[:200],
                        'srlimit': min(limit, 3), 'format': 'json', 'utf8': '1'})
    request = Request('https://ko.wikipedia.org/w/api.php?' + params,
                      headers={'User-Agent': 'LectureAssistant/0.2 (educational search)'})
    try:
        with urlopen(request, timeout=4) as response:
            data = json.loads(response.read(100_000))
        results = []
        for item in data.get('query', {}).get('search', []):
            title = item.get('title')
            if isinstance(title, str) and title:
                results.append({'title': title[:200], 'url': 'https://ko.wikipedia.org/wiki/' + quote(title.replace(' ', '_'))})
        return results
    except (HTTPError, URLError, TimeoutError, ValueError, OSError, KeyError, TypeError):
        return []
