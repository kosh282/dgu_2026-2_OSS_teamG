"""자동 음량 조절과 통합 이벤트를 거치지 않는 sherpa 입력 비교 데모."""
import argparse
import json
import queue
import threading
import time
import wave
import warnings

import numpy as np

from .config import DEFAULT_MODEL_DIR, SttConfig
from .engine import SherpaEngine


class LiveCapture:
    """캡처를 별도 스레드에서 유지하고 누락과 장치 오류를 드러낸다."""
    def __init__(self, source, device=None, queue_size=100):
        self.source, self.device = source, device
        self.blocks = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.dropped = 0
        self.discontinuities = 0
        self.captured = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _put(self, samples):
        self.captured += 1
        try:
            self.blocks.put_nowait((self.captured, time.monotonic(), samples.copy()))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        try:
            if self.source == 'microphone':
                import sounddevice as sd
                device = int(self.device) if self.device is not None else None
                with sd.InputStream(samplerate=16000, channels=1, dtype='float32', device=device) as source:
                    while not self.stop_event.is_set():
                        samples, overflow = source.read(1600)
                        self.discontinuities += int(bool(overflow))
                        self._put(samples)
            else:
                import soundcard as sc
                from .loopback import _co_initialize
                _co_initialize()
                speaker = sc.get_speaker(self.device) if self.device else sc.default_speaker()
                microphone = sc.get_microphone(id=str(speaker.id), include_loopback=True)
                print(f'[입력 장치] {speaker.name}', flush=True)
                with microphone.recorder(samplerate=16000) as source:
                    while not self.stop_event.is_set():
                        # SoundCard의 불연속 경고를 횟수로 기록한다.
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            samples = source.record(numframes=1600)
                        for warning in caught:
                            if 'discontinuity' in str(warning.message).lower():
                                self.discontinuities += 1
                            else:
                                print(f'[입력 경고] {warning.message}', flush=True)
                        self._put(samples)
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description='기본 입력을 sherpa 모델에 직접 전달')
    parser.add_argument('--source', choices=['microphone', 'loopback'], default='microphone')
    parser.add_argument('--wav', help='16비트 PCM WAV 파일을 실시간 입력 대신 사용')
    parser.add_argument('--device', help='마이크 번호 또는 출력 장치 이름')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--duration', type=float, help='실시간 입력 시간(초)')
    args = parser.parse_args()

    config = SttConfig(model_dir=args.model_dir, auto_gain=False)
    engine = SherpaEngine(config)
    stream = engine.create_stream()
    recognizer = engine._recognizer  # 토큰 재조합 전 모델의 text 필드를 확인한다.
    previous = ''
    finals = []
    level_time = time.monotonic()
    processed_seconds = 0.0
    processing_seconds = 0.0

    def result():
        return json.loads(recognizer.get_result_as_json_string(stream)).get('text', '').strip()

    def feed(samples, rate):
        nonlocal previous, level_time, processed_seconds, processing_seconds
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if not samples.size:
            return
        if time.monotonic() - level_time >= 2:
            rms = float(np.sqrt(np.mean(samples * samples)))
            print(f'[입력] RMS {rms:.6f} · 샘플레이트 {rate}Hz', flush=True)
            level_time = time.monotonic()
        started = time.perf_counter()
        engine.accept_waveform(stream, rate, samples)
        engine.decode(stream)
        text = result()
        processing_seconds += time.perf_counter() - started
        processed_seconds += len(samples) / rate
        if text and text != previous:
            print(f'[인식 중] {text}', flush=True)
            previous = text
        if engine.is_endpoint(stream):
            if text:
                finals.append(text)
                print(f'[확정 전사] {text}', flush=True)
                print(f"[확정 전체 전사] {' '.join(finals)}", flush=True)
            engine.reset(stream)
            previous = ''

    print('모델 준비 완료. 종료: Ctrl+C', flush=True)
    try:
        if args.wav:
            with wave.open(args.wav, 'rb') as source:
                if source.getsampwidth() != 2:
                    raise ValueError('16비트 PCM WAV가 필요합니다')
                rate, channels = source.getframerate(), source.getnchannels()
                while data := source.readframes(max(1, rate // 10)):
                    samples = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0
                    feed(samples.reshape(-1, channels), rate)
        else:
            capture = LiveCapture(args.source, args.device)
            capture.start()
            deadline = time.monotonic() + args.duration if args.duration else None
            previous_id = 0
            report_at = time.monotonic() + 2
            try:
                while deadline is None or time.monotonic() < deadline:
                    try:
                        block_id, captured_at, samples = capture.blocks.get(timeout=0.2)
                    except queue.Empty:
                        if capture.done.is_set():
                            if capture.error:
                                raise RuntimeError(f'오디오 입력 실패: {capture.error}') from capture.error
                            break
                        continue
                    if block_id != previous_id + 1:
                        print('[입력 경고] 큐에서 음성이 누락되어 인식 구간을 다시 시작합니다.', flush=True)
                        engine.reset(stream)
                        previous = ''
                    previous_id = block_id
                    lag = max(0, (time.monotonic() - captured_at) * 1000)
                    feed(samples, 16000)
                    if time.monotonic() >= report_at:
                        rtf = processing_seconds / processed_seconds if processed_seconds else 0
                        print(f'[상태] 큐 {capture.blocks.qsize()}개 · 대기 {lag:.0f}ms · '
                              f'처리 RTF {rtf:.3f} · 버린 블록 {capture.dropped}개 · '
                              f'장치 불연속 {capture.discontinuities}회', flush=True)
                        report_at = time.monotonic() + 2
            finally:
                capture.stop()
                remaining = capture.blocks.qsize()
                print(f'[입력 종료] 처리하지 않은 블록 {remaining}개 · 버린 블록 {capture.dropped}개 · '
                      f'장치 불연속 {capture.discontinuities}회', flush=True)
    except KeyboardInterrupt:
        print('\n입력 종료', flush=True)
    finally:
        # 인식기에 남아 있는 마지막 소리를 처리한다. 저장용 오디오는 생성하지 않는다.
        engine.input_finished(stream)
        engine.decode(stream)
        text = result()
        if text:
            finals.append(text)
            print(f'[마지막 전사] {text}', flush=True)
        print(f"[확정 전체 전사] {' '.join(finals) or '(없음)'}", flush=True)


if __name__ == '__main__':
    main()
