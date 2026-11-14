"""faster-whisper 롤링 윈도우 실시간 STT 비교 데모.

프로젝트 루트에서 실행합니다. 오디오는 메모리에만 두며, 전사 결과는 기본적으로 콘솔에만 출력합니다.
"""
import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration import IntegrationService
from sherpa_stt.config import SttConfig
from sherpa_stt.audio import to_mono_float32
from faster_whisper_stt.overlap import OverlapMerger


class RollingWhisper:
    """최근 오디오 창을 주기적으로 재전사해 콘솔에 표시한다."""
    def __init__(self, service, sid, model, sample_rate=16000, window_s=8.0,
                 update_s=1.5, stable_passes=2, label=''):
        self.service, self.sid, self.model = service, sid, model
        self.sample_rate = sample_rate
        self.window_samples = int(window_s * sample_rate)
        self.update_s = update_s
        self.stable_passes = stable_passes
        self.label = f'[{label}] ' if label else ''
        self.lock = threading.Lock()
        self.samples = deque(maxlen=self.window_samples)
        self.window_start_ms = 0
        self.sample_generation = 0
        self.previous = ""
        self.merger = OverlapMerger(stable_passes=stable_passes)
        self.last_lag_ms = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="faster-whisper-live", daemon=True)

    def start(self):
        self._thread.start()

    def accept(self, chunk):
        audio = to_mono_float32(chunk)
        if chunk.sample_rate_hz != self.sample_rate:
            raise ValueError("faster-whisper 데모는 16kHz 입력을 요구합니다")
        with self.lock:
            self.samples.extend(audio.tolist())
            self.window_start_ms = max(0, chunk.end_ms - int(len(self.samples) * 1000 / self.sample_rate))
            if len(audio):
                self.sample_generation += 1

    def _run(self):
        last_generation = -1
        while not self._stop.wait(self.update_s):
            started = time.perf_counter()
            with self.lock:
                audio = np.asarray(self.samples, dtype=np.float32)
                start_ms = self.window_start_ms
                generation = self.sample_generation
            # 새 음성이 없는데 같은 창을 재인식한 결과를 독립 확인으로 세지 않는다.
            if len(audio) < self.sample_rate or generation == last_generation:
                continue
            try:
                segments, _ = self.model.transcribe(audio, language="ko", beam_size=1,
                    vad_filter=True, condition_on_previous_text=False, word_timestamps=True)
                segments = list(segments)
                text = "".join(segment.text for segment in segments).strip()
                words = [(start_ms / 1000 + word.start, start_ms / 1000 + word.end, word.word)
                         for segment in segments for word in (segment.words or [])]
                last_generation = generation
            except Exception as exc:
                print(f"\nfaster-whisper 오류: {exc}", file=sys.stderr, flush=True)
                continue
            if not text:
                self.merger.update([])
                continue
            end_ms = start_ms + int(len(audio) * 1000 / self.sample_rate)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.last_lag_ms = max(0, self.service.elapsed_ms(self.sid) - end_ms)
            print(f"{self.label}[처리] 오디오 {len(audio)/self.sample_rate:.1f}초 · 추론 {elapsed_ms}ms · "
                  f"창 종료 후 경과 약 {self.last_lag_ms}ms", flush=True)
            if text != self.previous:
                # 갱신된 부분 전사를 덮어쓰지 않고 시간순으로 남긴다.
                print(f"{self.label}[{start_ms/1000:7.2f}s ~ {end_ms/1000:7.2f}s] 인식 중: {text}", flush=True)
                self.previous = text
            added = self.merger.update(words)
            if added:
                print(f"{self.label}[정리된 전사 추가] {added}", flush=True)
                print(f"{self.label}[정리된 전체 전사] {self.merger.final_text()}", flush=True)

    def stop(self):
        self._stop.set()
        self._thread.join()
        print(f"\n{self.label}[정리된 전체 전사] {self.merger.final_text() or '(없음)'}", flush=True)
        pending = self.merger.pending_text()
        if pending:
            print(f"{self.label}[미확정 끝부분] {pending}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="faster-whisper 한국어 실시간 비교 데모")
    parser.add_argument("--source", choices=["loopback", "microphone"], default="loopback")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parent / "models"),
                        help="모델 다운로드 위치 (기본: faster_whisper_stt/models)")
    parser.add_argument("--compute-type", default="int8", help="예: int8 (CPU), float16 (CUDA)")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--update-seconds", type=float, default=2.0)
    parser.add_argument("--stable-passes", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--target-process-id", type=int, default=None,
                        help="Windows에서 선택한 프로그램의 오디오만 캡처(미지정 시 출력 루프백 전체)")
    args = parser.parse_args(argv)
    if args.window_seconds < 2 or args.update_seconds <= 0 or args.stable_passes < 1:
        parser.error("window-seconds는 2 이상, update-seconds와 stable-passes는 양수여야 합니다")
    from faster_whisper import WhisperModel
    print(f"모델 로드: {args.model} ({args.compute_type}). 첫 실행은 모델 다운로드가 필요합니다.")
    model = WhisperModel(args.model, device="cuda" if args.compute_type == "float16" else "cpu",
                         compute_type=args.compute_type, cpu_threads=args.cpu_threads,
                         download_root=args.model_dir)
    from faster_whisper_stt.capture import make_capture
    config = SttConfig()
    service = IntegrationService()
    sid = service.start_session("faster-whisper 비교", {
        "config_version": "faster-whisper-demo", "participation_model": None,
        "stt_model": {"model_id": f"faster-whisper-{args.model}", "model_version": "1.2.1",
                      "sample_rate_hz": 16000, "window_seconds": args.window_seconds,
                      "update_seconds": args.update_seconds}})
    capture = make_capture(service, sid, args.source, args.device, args.target_process_id)
    worker = RollingWhisper(service, sid, model, window_s=args.window_seconds,
                            update_s=args.update_seconds, stable_passes=args.stable_passes)
    capture.start()
    worker.start()
    print(f"실시간 전사 중 · 분석 대기 {args.update_seconds:g}초 · 안정화 {args.stable_passes}회. 종료: Ctrl+C", flush=True)
    count = 0
    started = time.monotonic()
    try:
        while True:
            try:
                chunk = capture.chunks.get(timeout=0.2)
            except Exception:
                continue
            worker.accept(chunk)
            count += 1
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        worker.stop()
        service.stop_session(sid)
        print(f"\n청크 {count}개, 실행 {time.monotonic()-started:.1f}s, 마지막 추정 지연 {worker.last_lag_ms}ms")
        print("비교 데모는 기존 세션에 전사 이벤트를 저장하지 않았습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
