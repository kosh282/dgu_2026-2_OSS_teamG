import unittest
from unittest.mock import patch
from engagement_model.position_tracks import PositionTrackGaussianRuntime
from engagement_model.tests.test_position_tracks import sample
from engagement_model.window_publisher import WindowPublisher
from integration import IntegrationService
from engagement_model.eye_fusion_runtime import EyeFusionGaussianRuntime
from engagement_model.tests.test_eye_candidate import frame as single_frame


def eyes(t, value=0.3, xs=(.25,.75)):
    result = sample(t, xs)
    for face in result['features']['faces']:
        face['eye_aspect_ratio_left'] = face['eye_aspect_ratio_right'] = value
    return result

class EyeFusionTests(unittest.TestCase):
    def test_webcam_single_runtime_combines_and_defers_when_eyes_missing(self):
        model = EyeFusionGaussianRuntime()
        for t in range(0, 25001, 100):
            result = model(single_frame(t, 0.3 if t < 12000 else 0.1))
        self.assertLess(result['score'], result['eye_candidate']['head_score'])
        self.assertLess(result['current_direction_score'], result['head_direction_score'])
        missing = model(single_frame(25100, None))
        self.assertIsNone(missing['score'])
        self.assertIsNone(missing['current_direction_score'])
        self.assertIn('eye_observation_unavailable', missing['reason_codes'])

    def test_single_publisher_records_fused_version_and_weights(self):
        service = IntegrationService()
        sid = service.start_session('test')
        runtime = EyeFusionGaussianRuntime()
        publisher = WindowPublisher(service, sid, runtime, start_ms=0)
        for t in range(0, 25001, 100):
            feature = single_frame(t, 0.3 if t < 12000 else 0.1)
            feature['session_id'] = sid
            with patch.object(service, 'elapsed_ms', return_value=t):
                publisher(feature)
        events = [e for e in service.get_events_in_range(sid, 0, 25000)
                  if e['event_type'] == 'engagement_window' and e['payload']['model_output']]
        self.assertTrue(events)
        self.assertLess(events[-1]['payload']['model_output']['value'], 100)
        self.assertEqual(events[-1]['provenance']['model_version'], 'angle-time-0.5+eye')
        self.assertEqual(events[-1]['payload']['model_extensions']['gaussian-head-direction']['eye_weight'], 0.2)
    def test_eye_missing_defers_without_creating_zero(self):
        runtime = PositionTrackGaussianRuntime(eye_fusion=True)
        for t in range(0, 20001, 100):
            feature = sample(t)
            for face in feature['features']['faces']:
                face['eye_aspect_ratio_left'] = None
            result = runtime(feature)
        self.assertIsNone(result['score'])
        self.assertIsNone(result['current_direction_score'])
        self.assertIn('eye_observation_unavailable', result['reason_codes'])

    def test_closed_eye_reduces_final_published_window(self):
        service = IntegrationService(); sid=service.start_session('test')
        runtime = PositionTrackGaussianRuntime(eye_fusion=True)
        runtime.model_version = 'position-gaussian-0.2+face-transform-eye'
        pub=WindowPublisher(service,sid,runtime,start_ms=0)
        for t in range(0, 30001, 100):
            feature=eyes(t, 0.3 if t < 17000 else 0.1)
            feature['session_id']=sid
            with patch.object(service,'elapsed_ms',return_value=t): pub(feature)
        events=[e for e in service.get_events_in_range(sid,0,30000) if e['event_type']=='engagement_window']
        values=[e['payload']['model_output']['value'] for e in events if e['payload']['model_output']]
        self.assertGreaterEqual(max(values),99)
        self.assertLess(values[-1],max(values))
        extension=events[-1]['payload']['model_extensions']['position-track-gaussian']
        self.assertEqual((extension['head_weight'],extension['eye_weight']),(0.8,0.2))
        self.assertNotIn('eye_candidate',repr(events))
        self.assertNotIn('transient_track_key',repr(events))

    def test_one_valid_person_is_scoreable_without_fabricating_missing_eye(self):
        runtime=PositionTrackGaussianRuntime(eye_fusion=True)
        for t in range(0,15001,100):
            feature=eyes(t)
            feature['features']['faces'][1]['eye_aspect_ratio_left']=None
            result=runtime(feature)
        self.assertEqual(result['scoreable_count'],1)
        self.assertEqual(len(result['gaussian_observation']),1)
        self.assertIn('eye_observation_unavailable',result['reason_codes'])
        rows=runtime.participants()
        self.assertIsNone(rows[1]['current_direction_score'])
        self.assertEqual(rows[0]['current_direction_score'],100)

    def test_downward_grace_does_not_erase_valid_eye_observation(self):
        runtime = PositionTrackGaussianRuntime(eye_fusion=True)
        for t in range(0, 15001, 100):
            result = runtime(eyes(t))
        for t in range(15100, 16101, 100):
            feature = eyes(t)
            for face in feature['features']['faces']:
                face['head_angles_degrees']['pitch'] = 20.0
            result = runtime(feature)
        self.assertEqual(result['current_direction_score'], 100)
        self.assertEqual(result['scoreable_count'], 2)

if __name__=='__main__': unittest.main()
