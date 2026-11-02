import json
import tempfile
import unittest
from pathlib import Path

from integration import IntegrationService
from experimental_vision.vision import VisionPipeline


class Image:
    shape = (480, 640, 3)


class Capture:
    def read(self):
        return Image()


class Analyzer:
    def analyze(self, frame):
        return [{'eye_openness': 0.2}, {'eye_openness': 0.3}]

    def close(self):
        pass


class VisionTests(unittest.TestCase):
    def test_model_delivery_status_and_separate_intermediate_file(self):
        service = IntegrationService()
        sid = service.start_session('test')
        received = []
        with tempfile.TemporaryDirectory(dir='.') as directory:
            intermediate = Path(directory) / 'intermediate.jsonl'
            output = Path(directory) / 'session.json'
            pipeline = VisionPipeline(service, sid, Capture(), Analyzer(), received.append, intermediate)
            result = pipeline.process_one()
            self.assertEqual(len(result['features']['faces']), 2)
            self.assertEqual(received, [result])
            self.assertEqual(json.loads(intermediate.read_text(encoding='utf-8')), result)
            self.assertEqual(service.get_dashboard_snapshot(sid)['status']['vision'], 'running')
            pipeline.close()
            service.stop_session(sid)
            service.save_session(sid, output)
            saved = json.loads(output.read_text(encoding='utf-8'))
            self.assertNotIn('eye_openness', json.dumps(saved))
            self.assertEqual(service.get_dashboard_snapshot(sid)['status']['vision'], 'stopped')

    def test_capture_failure_reports_unavailable(self):
        class BrokenCapture:
            def read(self):
                raise RuntimeError('capture failed')
        service = IntegrationService()
        sid = service.start_session('test')
        pipeline = VisionPipeline(service, sid, BrokenCapture(), Analyzer())
        with self.assertRaises(RuntimeError):
            pipeline.process_one()
        self.assertEqual(service.get_dashboard_snapshot(sid)['status']['vision'], 'unavailable')
        pipeline.close()


if __name__ == '__main__':
    unittest.main()
