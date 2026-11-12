import copy
import json
import unittest
from unittest.mock import patch
from engagement_model.position_tracks import PositionTrackGaussianRuntime
from engagement_model.tests.test_runtime import feature_window
from engagement_model.window_publisher import WindowPublisher
from frontend.live_session import LiveSession
from integration import IntegrationService


def sample(t, xs=(0.25, 0.75)):
    item=feature_window(t, faces=len(xs))
    for i,(face,x) in enumerate(zip(item['features']['faces'],xs),1):
        face['transient_track_key']=i
        face['head_position']={'x':x, 'y':0.5}
        face['head_angles_degrees']={'pitch':0.,'yaw':0.,'roll':0.}
        face['eye_aspect_ratio_left']=face['eye_aspect_ratio_right']=0.3
    return item

class PositionTrackTests(unittest.TestCase):
    def ready(self):
        model=PositionTrackGaussianRuntime()
        for t in range(0, 3301, 100): model(sample(t))
        return model

    def test_two_faces_calibrate_and_group_score_without_reid(self):
        model=self.ready()
        value=sample(3400)
        original=copy.deepcopy(value)
        result=model(value)
        self.assertEqual(value,original)
        self.assertEqual(len(model.participants()),2)
        self.assertTrue(all(r['calibration_status']=='ready' for r in model.participants()))
        self.assertEqual(result['current_direction_score'],100)

    def test_loss_and_reappearance_reset_previous_baselines(self):
        model=self.ready()
        model(sample(3400, (0.25,)))
        self.assertTrue(all(r['calibration_status']=='collecting' for r in model.participants()))
        model(sample(3500))
        self.assertTrue(all(r['calibration_status']=='collecting' for r in model.participants()))
        self.assertTrue(all(r['current_direction_score'] is None for r in model.participants()))

    def test_large_move_overlap_and_gap_reset(self):
        for data in (sample(3400,(0.25,0.55)), sample(3400,(0.49,0.51)),sample(4400)):
            model=self.ready(); result=model(data)
            self.assertIsNone(result['current_direction_score'])
            self.assertTrue(all(r['calibration_status']=='collecting' for r in model.participants()))

    def test_published_events_contain_only_anonymous_group_values(self):
        service=IntegrationService(); sid=service.start_session('test')
        model=PositionTrackGaussianRuntime(); pub=WindowPublisher(service,sid,model,start_ms=0)
        for t in range(0,15001,100):
            value=sample(t); value['session_id']=sid
            with patch.object(service,'elapsed_ms',return_value=t): pub(value)
        events=service.get_events_in_range(sid,0,15000)
        rendered=json.dumps(events)
        self.assertIn('position-track-time-weighted-gaussian',rendered)
        self.assertNotIn('transient_track_key',rendered)
        self.assertNotIn('head_angles',rendered)
        self.assertTrue(any(e['event_type']=='engagement_window' and e['payload']['model_output'] for e in events))

    def test_frontend_default_uses_position_group(self):
        live=LiveSession(1)
        self.assertIsInstance(live._make_model(), PositionTrackGaussianRuntime)

    def test_association_change_clears_previous_group_window(self):
        service = IntegrationService()
        sid = service.start_session('test')
        model = PositionTrackGaussianRuntime()
        publisher = WindowPublisher(service, sid, model, start_ms=0)
        for t in range(0, 11001, 100):
            value = sample(t)
            value['session_id'] = sid
            with patch.object(service, 'elapsed_ms', return_value=t):
                publisher(value)
        self.assertTrue(publisher._segments)
        value = sample(11100, (0.25,))
        value['session_id'] = sid
        with patch.object(service, 'elapsed_ms', return_value=11100):
            publisher(value)
        self.assertEqual(len(publisher._segments), 0)

    def test_frontend_receives_and_publishes_multiple_tracks_without_reid(self):
        live = LiveSession(1)
        live.sid = live.service.start_session('test')
        with patch.object(live.service, 'elapsed_ms', return_value=0):
            live._start_model()
        for t in range(0, 24001, 100):
            value = sample(t)
            value['session_id'] = live.sid
            with patch.object(live.service, 'elapsed_ms', return_value=t):
                live._consume_frame(value)
        self.assertFalse(live._model_failed)
        self.assertEqual(len(live.participant_snapshot()), 2)
        events = live.service.get_events_in_range(live.sid, 0, 24000)
        self.assertTrue(any(e['event_type'] == 'engagement_window' and e['payload']['model_output'] for e in events))

if __name__=='__main__': unittest.main()
