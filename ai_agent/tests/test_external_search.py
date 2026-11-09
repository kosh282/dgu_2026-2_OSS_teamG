import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from ai_agent.external_search import search_reference


class SearchTests(unittest.TestCase):
    def test_result_links_are_built_from_search_titles(self):
        response = io.BytesIO(json.dumps({'query': {'search': [{'title': '운영 체제'}]}}).encode())
        with patch('ai_agent.external_search.urlopen', return_value=response):
            found = search_reference('운영 체제')
        self.assertEqual(found[0]['title'], '운영 체제')
        self.assertTrue(found[0]['url'].startswith('https://ko.wikipedia.org/wiki/'))

    def test_network_failure_does_not_stop_suggestion(self):
        with patch('ai_agent.external_search.urlopen', side_effect=URLError('offline')):
            self.assertEqual(search_reference('운영 체제'), [])


if __name__ == '__main__':
    unittest.main()
