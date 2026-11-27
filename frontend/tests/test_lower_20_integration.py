"""합성 영상 특징에서 발행한 하위 점수를 프런트가 읽는 경로를 확인한다."""
import unittest

from engagement_model.tests import test_lower_20
from frontend.dashboard_data import live_score_display, lower_20_score, transcript_score_link


class Lower20FrontendIntegrationTests(unittest.TestCase):
    def test_published_lower_score_reaches_live_and_review_for_both_models(self):
        fixture = test_lower_20.Lower20Tests()
        for positional in (False, True):
            with self.subTest(positional=positional):
                service, sid, model, _ = fixture.prepare(4, positional=positional)
                windows = service.get_dashboard_snapshot(sid)['engagement_windows']
                event = windows[-1]
                expected = event['payload']['model_extensions'][model.model_id]['lower_20_mean_score']
                self.assertIsNotNone(expected)
                self.assertEqual(lower_20_score(event), expected)
                display = live_score_display(windows, event['end_ms'], running=True,
                                             model_enabled=True, model_status='running')
                self.assertEqual(display['state'], 'current')
                self.assertEqual(lower_20_score(display['window']), expected)
                self.assertEqual(transcript_score_link({'end_ms': event['end_ms']}, windows)['lower_20'], expected)


if __name__ == '__main__':
    unittest.main()
