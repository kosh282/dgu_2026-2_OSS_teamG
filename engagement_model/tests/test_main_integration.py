"""최신 main의 공개 호출과 독립 자세 비교를 함께 검증한다. 카메라·식별 모델은 모의 입력이다."""
import contextlib
from copy import deepcopy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from engagement_model import webcam_demo as demo
from engagement_model.gaussian import GaussianRuntime
from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.tests.test_runtime import feature_window
from engagement_model.tests.test_yaw_calibration import face
from frontend.live_session import LiveSession


class MainIntegrationTests(unittest.TestCase):
    def test_live_renderer_keeps_page_api_with_transform_model(self):
        model=GaussianRuntime()
        frame=feature_window(0)
        model(frame)
        model.latest['angle_estimator']='mediapipe_face_transform'
        image=np.zeros((480,640,3),np.uint8)
        rendered=demo.render_panel(image,frame,{'gaussian':model},overlay_smoother=None,
            face=SimpleNamespace(last_landmarks=[]),pose=SimpleNamespace(last_landmarks=[]),
            show_overlay=False,debug=True,debug_page=1,live=True)
        self.assertEqual(rendered.shape,(860,1400,3))

    def test_transform_primary_runs_both_calibrations_and_keeps_main_callback(self):
        image=np.zeros((480,640,3),np.uint8)
        analyzer=SimpleNamespace(last_landmarks=[],last_face_centers=[(.5,.5)],close=Mock(),
                                 transform_faces=[face(dict(pitch=180,yaw=0,roll=0))])
        counter=[0]
        class Pipeline:
            def __init__(self,*args,**kwargs):
                self.consumer=kwargs['model_consumer']
                self.face_postprocessor=SimpleNamespace(reset=Mock())
            def process_one(self):
                counter[0]+=1
                timestamp=counter[0]*100
                frame=feature_window(timestamp)
                frame['features']['faces']=[face(dict(pitch=170,yaw=0,roll=0))]
                self.head_calibration.update(frame['features']['faces'],timestamp)
                frame['features']['head_calibration_status']=self.head_calibration.status
                frame['features']['head_calibration_diagnostics']=self.head_calibration.diagnostics(timestamp)
                self.consumer(frame)
                return frame
        camera=Mock()
        camera.isOpened.return_value=True
        with patch.object(demo.cv2,'VideoCapture',return_value=camera), \
             patch.object(demo,'FaceAnalyzer'), \
             patch.object(demo,'FaceTransformAnalyzer',return_value=analyzer), \
             patch.object(demo,'PoseAnalyzer'), \
             patch.object(demo,'PreviewCapture',return_value=SimpleNamespace(last_frame=image)), \
             patch.object(demo,'VisionPipeline',side_effect=Pipeline) as factory, \
             patch.object(demo.cv2,'namedWindow'),patch.object(demo.cv2,'resizeWindow'), \
             patch.object(demo.cv2,'setMouseCallback'),patch.object(demo.cv2,'imshow'), \
             patch.object(demo.cv2,'getWindowProperty',return_value=1), \
             patch.object(demo.cv2,'waitKey',side_effect=lambda _: ord('q') if counter[0]>=40 else -1), \
             patch.object(demo,'close_window'), \
             patch.object(demo,'render_panel',return_value=image) as render, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            demo.run(compare=False,pose_source='face-transform',calibrate=True)
        self.assertIsInstance(factory.call_args.kwargs['model_consumer'],GaussianRuntime)
        displayed=render.call_args.args[2]
        self.assertEqual(displayed['gaussian'].latest['angle_estimator'],'mediapipe_face_transform')
        self.assertEqual(displayed['gaussian'].latest['current_direction_score'],100)
        self.assertEqual(displayed['gaussian_pnp'].latest['current_direction_score'],100)
        self.assertIn('FACE_TRANSFORM_COMPARE',output.getvalue())
        self.assertIn('display primary=face-transform',output.getvalue())
        analyzer.close.assert_called_once()
        camera.release.assert_called_once()

    def test_frontend_reidentification_to_person_scores_and_anonymous_events(self):
        live=LiveSession(123,reidentification_enabled=True)
        live.sid=live.service.start_session('integration test')
        live._active_capture=SimpleNamespace(last_frame=np.zeros((100,100,3),np.uint8))
        live._active_face=SimpleNamespace(last_face_boxes=[])
        live.reidentifier=Mock()
        live.reidentifier.identify.return_value={1:'private-person-A',2:'private-person-B'}
        with patch.object(live.service,'elapsed_ms',return_value=0):
            live._start_model()
        self.assertIsInstance(live.model_runtime,MultiFaceGaussianRuntime)
        for timestamp in range(0,14001,100):
            frame=feature_window(timestamp,faces=2)
            for index,item in enumerate(frame['features']['faces'],1):
                item['transient_track_key']=index
                item['head_angles_degrees']=dict(pitch=0,yaw=index*5,roll=0)
            before=deepcopy(frame)
            with patch.object(live.service,'elapsed_ms',return_value=timestamp):
                live._consume_frame(frame)
            self.assertEqual(frame,before)
        self.assertFalse(live._model_failed)
        self.assertEqual(len(live.model_runtime.participants()),2)
        self.assertEqual(live.model_latest['score'],100)
        events=live.service.get_events_in_range(live.sid,0,15000)
        self.assertTrue(any(event['event_type']=='engagement_window' for event in events))
        self.assertNotIn('private-person',json.dumps(events))

    def test_new_cli_preserves_yaw_log_and_old_gaze_flags(self):
        for flags in (['--pose-source','face-transform','--yaw-diagnostic','--yaw-log'],['--gaze']):
            with patch('sys.argv',['webcam_demo',*flags]),patch.object(demo,'run') as run:
                demo.main()
            if '--gaze' in flags:
                self.assertTrue(run.call_args.kwargs['gaze'])
            else:
                self.assertTrue(run.call_args.kwargs['yaw_log'])
                self.assertEqual(run.call_args.kwargs['pose_source'],'face-transform')


if __name__=='__main__':
    unittest.main()
