"""sherpa 스트리밍 전사만 확인하는 최소 콘솔 데모."""
import argparse
import json
import queue
import wave

import numpy as np

from .basic_demo import LiveCapture
from .config import DEFAULT_MODEL_DIR, SttConfig
from .engine import SherpaEngine


def main():
    parser = argparse.ArgumentParser(description='sherpa 실시간 음성 전사')
    parser.add_argument('--source', choices=['loopback', 'microphone'], default='loopback')
    parser.add_argument('--device', help='출력 장치 이름 또는 마이크 번호')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--wav', help='입력 확인용 16비트 PCM WAV')
    args = parser.parse_args()

    engine = SherpaEngine(SttConfig(model_dir=args.model_dir, auto_gain=False))
    stream = engine.create_stream()
    previous = ''

    def model_text():
        raw = engine._recognizer.get_result_as_json_string(stream)
        return json.loads(raw).get('text', '').strip()

    def feed(audio, rate):
        nonlocal previous
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1, dtype=np.float32)
        if not audio.size:
            return
        engine.accept_waveform(stream, rate, audio)
        engine.decode(stream)
        text = model_text()
        if text and text != previous:
            print(f'[인식 중] {text}', flush=True)
            previous = text
        if engine.is_endpoint(stream):
            if text:
                print(f'[확정] {text}', flush=True)
            engine.reset(stream)
            previous = ''

    print('sherpa 준비 완료 · 종료: Ctrl+C', flush=True)
    capture = None
    try:
        if args.wav:
            with wave.open(args.wav, 'rb') as wav:
                if wav.getsampwidth() != 2:
                    raise ValueError('16비트 PCM WAV가 필요합니다')
                rate, channels = wav.getframerate(), wav.getnchannels()
                while data := wav.readframes(max(1, rate // 10)):
                    samples = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768
                    feed(samples.reshape(-1, channels), rate)
        else:
            capture = LiveCapture(args.source, args.device)
            capture.start()
            last_id = 0
            while True:
                try:
                    chunk_id, _, samples = capture.blocks.get(timeout=0.2)
                except queue.Empty:
                    if capture.done.is_set():
                        if capture.error:
                            raise RuntimeError(f'오디오 입력 실패: {capture.error}') from capture.error
                        break
                    continue
                if chunk_id != last_id + 1:
                    print('[입력 경고] 오디오가 누락되어 인식을 다시 시작합니다.', flush=True)
                    engine.reset(stream)
                    previous = ''
                last_id = chunk_id
                feed(samples, 16000)
    except KeyboardInterrupt:
        print('\n입력 종료', flush=True)
    finally:
        if capture:
            capture.stop()
            if capture.blocks.qsize():
                print(f'[입력 안내] 종료 시 미처리 블록 {capture.blocks.qsize()}개', flush=True)
            if capture.discontinuities:
                print(f'[입력 안내] 장치 불연속 {capture.discontinuities}회', flush=True)
        engine.input_finished(stream)
        engine.decode(stream)
        text = model_text()
        if text:
            print(f'[마지막 전사] {text}', flush=True)


if __name__ == '__main__':
    main()
