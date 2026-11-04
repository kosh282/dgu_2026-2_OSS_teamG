"""CLI: real-time microphone, Windows speaker loopback, or audio/video files."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from stt_core import Session, STTProcessor, create_recognizer, audio_chunks_from_f32


def main():
    ap = argparse.ArgumentParser(description='Korean streaming STT + team events v0.1.0')
    ap.add_argument('--mode', choices=['file', 'mic', 'loopback'], help='file: video/audio; mic: microphone; loopback: Windows speaker')
    ap.add_argument('--list-devices', action='store_true', help='List microphone and speaker devices')
    ap.add_argument('--input', help='Audio or video path, used with --mode file')
    ap.add_argument('--model-dir', default='models/sherpa-onnx-streaming-zipformer-korean-2024-06-16')
    ap.add_argument('--output', default='output/session.json')
    ap.add_argument('--events-jsonl', default=None, help='All emitted partial/final events, for UI integration. Not session JSON.')
    ap.add_argument('--title', default='STT Test Lecture')
    ap.add_argument('--session-id', help='Shared UUID from team coordinator')
    ap.add_argument('--clock-origin', type=float, help='Shared time.monotonic origin from team coordinator')
    ap.add_argument('--device', type=int, help='Microphone input device index')
    ap.add_argument('--chunk-ms', type=int, default=200)
    ap.add_argument('--timeline-offset-ms', type=int, default=0, help='File timeline offset in milliseconds; use for media A/V alignment')
    args = ap.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print('Microphone devices:')
        print(sd.query_devices())
        try:
            import soundcard as sc
            print('\nSpeaker devices for loopback:')
            for speaker in sc.all_speakers():
                print(f'  {speaker.name} ({speaker.id})')
        except Exception as exc:
            print(f'Speaker enumeration unavailable: {exc}')
        return
    if not args.mode:
        ap.error('--mode is required (or use --list-devices)')
    if args.timeline_offset_ms < 0:
        ap.error('--timeline-offset-ms must be >= 0')
    if args.chunk_ms < 50 or args.chunk_ms > 1000:
        ap.error('--chunk-ms must be between 50 and 1000')
    if args.mode == 'file' and not args.input:
        ap.error('--input required for file mode')
    if args.mode == 'file' and not shutil.which('ffmpeg'):
        ap.error('ffmpeg executable is required for audio/video file mode')

    # Offline file processing uses media timeline, while created_ms records processing time.
    session = Session(args.title, session_id=args.session_id, clock_origin=args.clock_origin)
    rec = create_recognizer(args.model_dir)
    stream = None
    if args.events_jsonl:
        event_path = Path(args.events_jsonl)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        stream = event_path.open('w', encoding='utf-8')

    def handle_event(event):
        tag = 'FINAL' if event['event_type'] == 'transcript_final' else 'PARTIAL'
        print(f"{tag:7} [{event['start_ms']/1000:7.2f}-{event['end_ms']/1000:7.2f}] {event['payload']['text']}", flush=True)
        if stream is not None:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()

    p = STTProcessor(session, rec, handle_event)
    nsamples = 0
    seq = 0
    capture_offset_ms = None
    chunk_size = args.chunk_ms * 16

    def feed(arr):
        nonlocal nsamples, seq, capture_offset_ms
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        if not len(a):
            return
        if capture_offset_ms is None:
            # For live input, anchor the first chunk to the session clock using the
            # approximate instant the full chunk became available.
            capture_offset_ms = (args.timeline_offset_ms if args.mode == 'file' else
                                 max(0, session.elapsed_ms() - round(len(a) / 16)))
        p.accept(audio_chunks_from_f32(a, session.session_id, seq, nsamples, capture_offset_ms))
        nsamples += len(a)
        seq += 1

    try:
        if args.mode == 'file':
            source = Path(args.input)
            if not source.is_file():
                raise FileNotFoundError(source)
            # File audio is decoded in memory as PCM stream; video frames are untouched.
            cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-i', str(source), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
                   '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1']
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert proc.stdout is not None
            bytes_per_chunk = chunk_size * 4
            try:
                while True:
                    raw = proc.stdout.read(bytes_per_chunk)
                    if not raw:
                        break
                    feed(np.frombuffer(raw[:len(raw)//4*4], dtype='<f4'))
                code = proc.wait(timeout=20)
                if code:
                    err = proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr else ''
                    raise RuntimeError(f'ffmpeg failed: {err}')
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            p.finish()

        elif args.mode == 'mic':
            import sounddevice as sd
            from scipy.signal import resample_poly
            sample_rate = 48000
            frames = int(sample_rate * args.chunk_ms / 1000)
            print('Microphone active. Ctrl+C to stop.')
            with sd.InputStream(device=args.device, samplerate=sample_rate,
                                channels=1, dtype='float32') as mic:
                while True:
                    data, overflow = mic.read(frames)
                    if overflow:
                        raise RuntimeError('Audio overflow: chunk dropped; restarting is required')
                    feed(resample_poly(data[:, 0], 1, 3).astype(np.float32))

        elif args.mode == 'loopback':
            import soundcard as sc
            from scipy.signal import resample_poly
            speaker = sc.default_speaker()
            if speaker is None:
                raise RuntimeError('No default speaker; select Windows audio output first')
            mic = sc.get_microphone(id=str(speaker.id), include_loopback=True)
            if mic is None:
                raise RuntimeError('No speaker loopback available')
            print(f'Capturing system sound from {speaker.name}. Ctrl+C to stop.')
            with mic.recorder(samplerate=48000, blocksize=int(48000*args.chunk_ms/1000)) as recorder:
                while True:
                    data = recorder.record(numframes=int(48000*args.chunk_ms/1000))
                    # Mixed-to-mono is intentional for stereo speaker audio.
                    mono = np.asarray(data, dtype=np.float32).mean(axis=1)
                    feed(resample_poly(mono, 1, 3).astype(np.float32))
    except KeyboardInterrupt:
        print('\nStopping STT...')
        p.finish()
    finally:
        if stream is not None:
            stream.close()
        # Duration includes the user-configured media offset in file mode.
        duration_ms = (capture_offset_ms or 0) + round(nsamples / 16)
        if args.mode != 'file':
            duration_ms = max(duration_ms, session.elapsed_ms())
        session.save(args.output, duration_ms=duration_ms)
        print(f'Saved {len(session.events)} persisted events to {args.output}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
