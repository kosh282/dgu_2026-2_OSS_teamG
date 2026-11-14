"""겹침 없이 음성 구간을 한 번씩만 전사하는 비교용 콘솔 데모."""
import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np

from integration import IntegrationService
from sherpa_stt.audio import to_mono_float32
from faster_whisper_stt.capture import make_capture


class DirectTranscriber:
    def __init__(self, model, sample_rate=16000, chunk_seconds=2.0, queue_size=10):
        self.model = model
        self.rate = sample_rate
        self.size = round(chunk_seconds * sample_rate)
        if self.size < 1:
            raise ValueError('음성 구간 길이는 양수여야 합니다')
        self.buffer = np.empty(0, dtype=np.float32)
        self.start_ms = None
        self.last_id = None
        self.tasks = queue.Queue(maxsize=queue_size)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def accept(self, chunk):
        if self.error:
            raise RuntimeError(f'전사 오류: {self.error}') from self.error
        if chunk.sample_rate_hz != self.rate:
            raise ValueError('16kHz 입력이 필요합니다')
        if self.last_id is not None and chunk.chunk_id != self.last_id + 1:
            raise RuntimeError('오디오 입력이 누락되어 테스트를 중단합니다')
        self.last_id = chunk.chunk_id
        if self.start_ms is None:
            self.start_ms = float(chunk.start_ms)
        self.buffer = np.concatenate((self.buffer, to_mono_float32(chunk)))
        while len(self.buffer) >= self.size:
            end = self.start_ms + self.size * 1000 / self.rate
            try:
                self.tasks.put_nowait((self.start_ms, end, self.buffer[:self.size].copy()))
            except queue.Full as exc:
                raise RuntimeError('전사 대기열이 가득 차 음성이 지연됩니다') from exc
            self.buffer = self.buffer[self.size:].copy()
            self.start_ms = end

    def _run(self):
        try:
            while True:
                task = self.tasks.get()
                if task is None:
                    break
                start, end, audio = task
                began = time.perf_counter()
                segments, _ = self.model.transcribe(audio, language='ko', beam_size=1,
                                                     vad_filter=False, word_timestamps=False,
                                                     condition_on_previous_text=False)
                text = ''.join(segment.text for segment in segments).strip()
                elapsed = time.perf_counter() - began
                print(f'[{start/1000:.1f}~{end/1000:.1f}초] {text or "(인식 없음)"} '
                      f'· 추론 {elapsed:.2f}초 · 대기 {self.tasks.qsize()}구간', flush=True)
        except Exception as exc:
            self.error = exc
            print(f'[전사 오류] {exc}', flush=True)

    def _finish_task(self, task):
        while self.thread.is_alive():
            try:
                self.tasks.put(task, timeout=0.2)
                return
            except queue.Full:
                pass

    def stop(self):
        if len(self.buffer):
            end = self.start_ms + len(self.buffer) * 1000 / self.rate
            self._finish_task((self.start_ms, end, self.buffer.copy()))
        self._finish_task(None)
        self.thread.join()
        if self.error:
            raise RuntimeError(f'전사 오류: {self.error}') from self.error


def main():
    parser = argparse.ArgumentParser(description='중첩 없이 faster-whisper 전사 테스트')
    parser.add_argument('--source', choices=['loopback', 'microphone'], default='loopback')
    parser.add_argument('--device', default=None, help='출력 장치 이름 또는 마이크 번호')
    parser.add_argument('--model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large-v3'])
    parser.add_argument('--chunk-seconds', type=float, default=2.0)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if not 0.5 <= args.chunk_seconds <= 30:
        parser.error('--chunk-seconds는 0.5~30 사이여야 합니다')
    from faster_whisper import WhisperModel
    model_path = Path(__file__).resolve().parent / 'models'
    print(f'[준비] {args.model} 모델 로딩 중', flush=True)
    model = WhisperModel(args.model, device='cpu', compute_type='int8',
                         cpu_threads=args.cpu_threads, download_root=str(model_path))
    service = IntegrationService()
    sid = service.start_session('직접 전사 테스트', {'config_version': 'direct-stt-demo',
                                'participation_model': None, 'stt_model': None})
    capture = make_capture(service, sid, args.source, args.device)
    worker = DirectTranscriber(model, chunk_seconds=args.chunk_seconds)
    capture.start()
    worker.start()
    print(f'[시작] {args.chunk_seconds:g}초마다 한 번 전사합니다. 종료: Ctrl+C', flush=True)
    try:
        while True:
            try:
                chunk = capture.chunks.get(timeout=0.2)
            except queue.Empty:
                if getattr(capture, 'error', None):
                    raise RuntimeError(f'입력 오류: {capture.error}')
                if worker.error:
                    raise RuntimeError(f'전사 오류: {worker.error}')
                continue
            worker.accept(chunk)
    except KeyboardInterrupt:
        print('\n[종료] 남은 구간을 처리합니다.', flush=True)
    finally:
        capture.stop()
        pending = capture.chunks.qsize()
        if pending:
            print(f'[입력 안내] 종료 시 캡처 큐의 미처리 청크 {pending}개', flush=True)
        try:
            worker.stop()
        finally:
            service.stop_session(sid)
        print('[안내] 오디오와 전사는 파일에 저장하지 않았습니다.', flush=True)


if __name__ == '__main__':
    main()
