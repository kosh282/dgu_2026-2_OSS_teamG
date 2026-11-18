"""한 번 캡처한 PCM을 sherpa와 faster-whisper에 전달하는 실험 데모."""
import argparse
import json
import time
import wave

import numpy as np
from faster_whisper import WhisperModel

from integration import IntegrationService
from sherpa_stt.audio import AudioChunk
from sherpa_stt.basic_demo import LiveCapture
from sherpa_stt.config import SttConfig
from sherpa_stt.engine import SherpaEngine
from faster_whisper_stt.live_demo import RollingWhisper


def main():
    parser = argparse.ArgumentParser(description='동일 음성으로 sherpa와 faster-whisper 비교')
    parser.add_argument('--source', choices=['loopback', 'microphone'], default='loopback')
    parser.add_argument('--device', help='출력 장치 이름 또는 마이크 번호')
    parser.add_argument('--model', choices=['tiny', 'base', 'small', 'medium', 'large-v3'], default='base')
    parser.add_argument('--duration', type=float, default=None, help='입력 시간(초)')
    parser.add_argument('--wav', help='비교용 16kHz 모노 16비트 PCM WAV')
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error('--duration은 양수여야 합니다')

    print('[준비] 두 모델 로딩 중입니다.', flush=True)
    engine = SherpaEngine(SttConfig(auto_gain=False))
    stream = engine.create_stream()
    whisper = WhisperModel(args.model, device='cpu', compute_type='int8', cpu_threads=4,
                           download_root='faster_whisper_stt/models')
    service = IntegrationService()
    sid = service.start_session('음성 모델 비교', {'config_version': 'stt-compare',
                                'participation_model': None, 'stt_model': None})
    worker = RollingWhisper(service, sid, whisper, label='whisper')
    worker.start()
    previous = ''
    samples_total = 0
    processing = 0.0
    latest_stats = time.monotonic()
    capture = None

    def feed(samples):
        nonlocal samples_total, previous, processing, latest_stats
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1, dtype=np.float32)
        if not len(samples):
            return
        start = samples_total * 1000 // 16000
        samples_total += len(samples)
        end = samples_total * 1000 // 16000
        # 동일한 모노 float32 샘플을 두 경로에 전달한다. 자동 음량 조절은 없다.
        worker.accept(AudioChunk(sid, start, start, end, 16000, 1, 'float32', samples))
        began = time.perf_counter()
        engine.accept_waveform(stream, 16000, samples)
        engine.decode(stream)
        text = json.loads(engine._recognizer.get_result_as_json_string(stream)).get('text', '').strip()
        processing += time.perf_counter() - began
        if text and text != previous:
            print(f'[sherpa 인식 중] {text}', flush=True)
            previous = text
        if engine.is_endpoint(stream):
            if text:
                print(f'[sherpa 확정] {text}', flush=True)
            engine.reset(stream)
            previous = ''
        if time.monotonic() - latest_stats >= 2:
            rms = float(np.sqrt(np.mean(samples * samples)))
            q = capture.blocks.qsize() if capture else 0
            dropped = capture.dropped if capture else 0
            discontinuities = capture.discontinuities if capture else 0
            print(f'[공통 입력] RMS {rms:.5f} · 큐 {q} · 손실 {dropped} · 불연속 {discontinuities} · '
                  f'sherpa RTF {processing / (samples_total / 16000):.3f}', flush=True)
            latest_stats = time.monotonic()

    print('[시작] 동일 오디오 비교 중. 종료: Ctrl+C', flush=True)
    try:
        if args.wav:
            with wave.open(args.wav, 'rb') as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                    raise ValueError('16kHz 모노 16비트 PCM WAV가 필요합니다')
                start_clock = time.monotonic()
                while data := wav.readframes(1600):
                    feed(np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768)
                    time.sleep(max(0, start_clock + samples_total / 16000 - time.monotonic()))
        else:
            import queue
            capture = LiveCapture(args.source, args.device)
            capture.start()
            deadline = time.monotonic() + args.duration if args.duration else None
            previous_id = 0
            while deadline is None or time.monotonic() < deadline:
                try:
                    block_id, _, samples = capture.blocks.get(timeout=0.2)
                except queue.Empty:
                    if capture.done.is_set():
                        if capture.error:
                            raise RuntimeError(f'입력 실패: {capture.error}') from capture.error
                        break
                    continue
                if block_id != previous_id + 1:
                    print('[공통 입력 오류] 청크가 누락되어 비교를 종료합니다.', flush=True)
                    break
                previous_id = block_id
                feed(samples)
    except KeyboardInterrupt:
        print('\n[종료] 입력을 마칩니다.', flush=True)
    finally:
        if capture:
            capture.stop()
            print(f'[공통 입력 종료] 버린 블록 {capture.dropped} · 불연속 {capture.discontinuities} · '
                  f'미처리 블록 {capture.blocks.qsize()}', flush=True)
        engine.input_finished(stream)
        engine.decode(stream)
        text = json.loads(engine._recognizer.get_result_as_json_string(stream)).get('text', '').strip()
        if text:
            print(f'[sherpa 마지막] {text}', flush=True)
        worker.stop()
        # 종료 시 진행 중인 추론의 창이 오래됐을 수 있으므로 마지막 창을 별도로 확인한다.
        with worker.lock:
            last_audio = np.asarray(worker.samples, dtype=np.float32)
        if len(last_audio) >= 16000:
            started = time.perf_counter()
            segments, _ = whisper.transcribe(last_audio, language='ko', beam_size=1,
                                              vad_filter=True, condition_on_previous_text=False)
            text = ''.join(segment.text for segment in segments).strip()
            print(f'[whisper 마지막 창] {text or "(없음)"} '
                  f'· 처리 {(time.perf_counter()-started)*1000:.0f}ms (앞 전사와 겹칠 수 있음)', flush=True)
        service.stop_session(sid)
        print('[안내] 오디오와 전사 결과는 파일에 저장하지 않았습니다.', flush=True)


if __name__ == '__main__':
    main()
