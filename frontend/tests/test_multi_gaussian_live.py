"""실제 카메라 없이 프런트의 Gaussian·재식별 결선과 저장 경계를 검증한다."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from frontend.live_session import LiveSession
from engagement_model.position_tracks import PositionTrackGaussianRuntime
from engagement_model.tests.test_runtime import feature_window
from engagement_model.tests.test_multi_face import frame as multi_frame
from video_recognition.window_capture import FrameNotAvailable


class MultiGaussianLiveTests(unittest.TestCase):
    def live(self, reidentify=False):
        live = LiveSession(123, reidentification_enabled=reidentify)
        live.sid = live.service.start_session('synthetic',
            {'config_version':'test', 'participation_model':live._model_identity(), 'stt_model':None})
        live.service.elapsed_ms = Mock(return_value=0)
        self.addCleanup(lambda: live.service.stop_session(live.sid))
        return live

    def test_single_default_and_group_factory_are_gaussian(self):
        single = self.live()
        single._start_model()
        self.assertIsInstance(single.model_runtime, PositionTrackGaussianRuntime)
        group = self.live(True)
        group._start_model()
        self.assertEqual(group.model_runtime.model_id, 'multi-face-gaussian')
        self.assertEqual(group.model_runtime.config.scoring_method, 'gaussian')

    def test_started_session_uses_matching_gaussian_metadata(self):
        for enabled, expected in ((False,'position-track-gaussian'),(True,'multi-face-gaussian')):
            live = LiveSession(123,reidentification_enabled=enabled)
            with patch('frontend.live_session.threading.Thread'):
                live.start('synthetic')
            with tempfile.TemporaryDirectory() as directory:
                path = live.service.save_session(live.sid, Path(directory)/'session.json')
                saved = json.loads(Path(path).read_text(encoding='utf-8'))
            self.assertEqual(saved['config']['participation_model']['model_id'], expected)
            live.service.stop_session(live.sid)

    def test_consumer_identifies_same_frame_without_preview_and_keeps_features_unchanged(self):
        live = self.live(True)
        live._start_model()
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        live._active_capture = Mock(last_frame=image)
        live._active_face = Mock(last_face_boxes=[[.1,.1,.9,.9]])
        live.reidentifier = Mock()
        live.reidentifier.identify.return_value = {0: 'PRIVATE-ID'}
        feature = feature_window(0, track_key=0)
        feature['session_id'] = live.sid
        original = deepcopy(feature)
        live._consume_frame(feature)
        self.assertIs(live.reidentifier.identify.call_args.args[0], image)
        self.assertFalse(live.developer_preview)
        self.assertEqual(feature, original)
        self.assertEqual(live.participant_snapshot()[0]['id'], 'PRIVATE-ID')

    def test_reidentification_failure_does_not_stop_video_or_produce_person_scores(self):
        live = self.live(True)
        live._start_model()
        live._active_capture = Mock(last_frame=np.zeros((240,320,3),dtype=np.uint8))
        live._active_face = Mock(last_face_boxes=[])
        live.reidentifier = Mock()
        live.reidentifier.identify.side_effect = RuntimeError('synthetic failure')
        feature = feature_window(0)
        feature['session_id'] = live.sid
        live._consume_frame(feature)
        self.assertIsNone(live.error)
        self.assertIsNotNone(live.reidentification_error)
        self.assertFalse(live.stop_event.is_set())
        self.assertFalse(live.participant_snapshot())

    def test_saved_group_event_has_multi_provenance_and_no_identity(self):
        live = self.live(True)
        live._start_model()
        live._active_capture = Mock(last_frame=np.zeros((240,320,3),dtype=np.uint8))
        live._active_face = Mock(last_face_boxes=[[.1,.1,.4,.9],[.6,.1,.9,.9]])
        live.reidentifier = Mock()
        live.reidentifier.identify.return_value = {0:'PRIVATE-A', 1:'PRIVATE-B'}
        for ms in range(0,16001,100):
            live.service.elapsed_ms.return_value = ms
            feature = feature_window(ms, track_key=0)
            feature['session_id'] = live.sid
            face = feature['features']['faces'][0]
            face['head_angles_degrees'] = {'pitch':0.,'yaw':0.,'roll':0.}
            face['head_angles_smoothed_degrees'] = dict(face['head_angles_degrees'])
            other = deepcopy(face)
            other['transient_track_key'] = 1
            feature['features']['faces'].append(other)
            live._consume_frame(feature)
        rows = live.participant_snapshot()
        self.assertEqual(len(rows), 2)
        events = live.service.get_dashboard_snapshot(live.sid)['engagement_windows']
        self.assertTrue(any(e['payload']['model_output'] is not None for e in events))
        self.assertTrue(all(e['provenance']['model_id']=='multi-face-gaussian' for e in events))
        with tempfile.TemporaryDirectory() as directory:
            path = live.service.save_session(live.sid, Path(directory)/'synthetic.json')
            saved = Path(path).read_text(encoding='utf-8')
            for forbidden in ('PRIVATE-', 'transient_track_key', 'embedding', 'head_angles', 'participants'):
                self.assertNotIn(forbidden, saved)
            from integration import IntegrationService
            restored = IntegrationService()
            restored.load_session(path)
            self.assertEqual(events, restored.get_dashboard_snapshot(live.sid)['engagement_windows'])

    def test_snapshot_marks_capture_gap_unobservable_and_returns_copies(self):
        live = self.live(True)
        live._participant_rows = [{'id':'P001','status':'observed','score':95.,'current_direction_score':96.}]
        live.latest = {'start_ms':0}
        with patch.object(live.service,'elapsed_ms',return_value=2000):
            rows = live.participant_snapshot()
        self.assertIsNone(rows[0]['score'])
        rows[0]['id']='changed'
        self.assertEqual(live._participant_rows[0]['id'],'P001')

    def test_new_person_calibration_does_not_clear_existing_group_window(self):
        live = self.live(True)
        live._start_model()
        live.model_runtime.update_identities({1:'A',2:'B'})
        for ms in range(0,13101,100):
            live.service.elapsed_ms.return_value = ms
            feature = multi_frame(ms, (0,) if ms <= 12000 else (0,20))
            feature['session_id'] = live.sid
            live._consume_model(feature)
        self.assertFalse(live._model_failed)
        event = live.service.get_dashboard_snapshot(live.sid)['engagement_windows'][-1]
        self.assertEqual(event['payload']['model_output']['value'],100.)
        self.assertGreater(len(live.window_publisher._segments),80)

    def test_group_recalibration_resumes_status_after_new_window(self):
        live = self.live(True)
        live._start_model()
        live.model_runtime.update_identities({1:'A'})
        for ms in range(0,12001,100):
            live.service.elapsed_ms.return_value = ms
            feature=multi_frame(ms,(0,)); feature['session_id']=live.sid
            live._consume_model(feature)
        with patch.object(live,'is_running',return_value=True):
            live.request_head_calibration()
        live._recalibrate_pipeline(Mock())
        for ms in range(12100,23101,100):
            live.service.elapsed_ms.return_value = ms
            feature=multi_frame(ms,(0,)); feature['session_id']=live.sid
            live._consume_model(feature)
        self.assertEqual(live.service.get_dashboard_snapshot(live.sid)['status']['model'],'running')

    def test_actual_pipeline_identifies_without_preview_and_clears_people_on_exit(self):
        live = self.live(True)
        clock = {'ms':0}
        live.service.elapsed_ms.side_effect = lambda sid: clock['ms']
        image=np.zeros((240,320,3),dtype=np.uint8)

        class Window:
            process_id=1
            def read(self):
                clock['ms']+=100
                if clock['ms']==15000:
                    raise FrameNotAvailable('synthetic capture gap')
                if clock['ms']==15100:
                    self_test.assertTrue(live._participant_rows)
                    self_test.assertTrue(all(row['current_direction_score'] is None
                                             for row in live._participant_rows))
                if clock['ms']>=16000:
                    live.stop_event.set()
                return image
            close=Mock()

        class Face:
            last_face_centers=[(.25,.5),(.75,.5)]
            last_face_boxes=[[.1,.1,.4,.9],[.6,.1,.9,.9]]
            last_landmarks=[]
            def analyze(self, frame):
                return [dict(quality={'status':'valid','reason_codes':[]},
                             head_angles_degrees={'pitch':0.,'yaw':yaw,'roll':0.},
                             eye_openness=.3, head_position={'x':x,'y':.5})
                        for x,yaw in ((.25,0.),(.75,20.))]
            close=Mock()

        self_test=self
        pose=Mock()
        pose.analyze.return_value=[]
        reid=Mock()
        reid.identify.side_effect=lambda image,ms,faces,**kwargs: {
            face['transient_track_key']:f'P{index}' for index,face in enumerate(faces)}
        with patch('frontend.live_session.WindowCapture',return_value=Window()), \
             patch('frontend.live_session.FaceAnalyzer',return_value=Face()), \
             patch('frontend.live_session.PoseAnalyzer',return_value=pose), \
             patch('frontend.live_session.FaceReidentifier',return_value=reid), \
             patch.object(live,'_close_developer_windows'):
            live._run_vision()
        self.assertIsNone(live.error)
        self.assertGreater(reid.identify.call_count,100)
        reid.render.assert_not_called()
        reid.close.assert_called_once()
        events=live.service.get_dashboard_snapshot(live.sid)['engagement_windows']
        self.assertTrue(any(event['payload']['model_output'] is not None for event in events))
        self.assertEqual(live.participant_snapshot(),[])
        self.assertEqual(live.model_runtime.participants(),[])
        self.assertIsNone(live.reidentifier)
