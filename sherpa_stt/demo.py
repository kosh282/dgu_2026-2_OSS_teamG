"""sherpa-onnx STT 개발자 데모.

예)
  python -m sherpa_stt.demo --list-devices
  python -m sherpa_stt.demo --model-dir models/sherpa-onnx-streaming-zipformer-korean-2024-06-16
  python -m sherpa_stt.demo --model-dir <모델 폴더> --wav sample.wav --save out/session.json

전사 이벤트를 콘솔에 출력한다. 원본 오디오는 저장하지 않는다.
`--save`로 저장하는 세션 JSON에는 전사 텍스트가 들어가므로 실제 수업 결과를 커밋하지 않는다.
"""
import argparse
import sys
import time
import wave

import numpy as np

from integration import IntegrationService

from .audio import AudioChunk
from .config import DEFAULT_MODEL_DIR, SttConfig
from .engine import SherpaEngine
from .transcriber import SherpaTranscriber


def _print_event(event):
    kind, payload = event["event_type"], event["payload"]
    span = f"[{event['start_ms'] / 1000:7.2f}s ~ {event['end_ms'] / 1000:7.2f}s]"
    if kind == "transcript_partial":
        print(f"\r{span} … {payload['text']}", end="", flush=True)
    elif kind == "transcript_final":
        print(f"\r{span} ✔ {payload['text']}", flush=True)
    elif kind == "analysis_status":
        reasons = ",".join(event["quality"]["reason_codes"])
        print(f"\n{span} [{payload['component']}:{payload['state']}] {payload['message']} {reasons}".rstrip(), flush=True)


class _PrintingService(IntegrationService):
    """수신한 이벤트를 콘솔에 출력하는 데모용 통합 서비스."""

    def receive_event(self, event):
        accepted = super().receive_event(event)
        if accepted:
            _print_event(event)
        return accepted


def _read_wav(path):
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("16-bit PCM WAV만 지원합니다")
        rate, channels = handle.getframerate(), handle.getnchannels()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    return rate, channels, data.reshape(-1, channels) if channels > 1 else data


def _run_wav(service, sid, transcriber, path, chunk_ms):
    rate, channels, samples = _read_wav(path)
    step = max(1, rate * chunk_ms // 1000)
    for chunk_id, offset in enumerate(range(0, len(samples), step)):
        part = samples[offset:offset + step]
        start_ms = offset * 1000 // rate
        end_ms = (offset + len(part)) * 1000 // rate
        # 세션 공통 시계보다 앞선 이벤트를 만들지 않도록 실시간 속도로 공급한다.
        while service.elapsed_ms(sid) < end_ms:
            time.sleep(0.01)
        transcriber.accept_chunk(AudioChunk(sid, chunk_id, start_ms, end_ms, rate, channels, "int16", part))
    transcriber.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description="sherpa-onnx 한국어 스트리밍 STT 데모")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="tokens.txt와 encoder/decoder/joiner .onnx가 있는 폴더 (기본: sherpa_stt/models/한국어 모델)")
    parser.add_argument("--source", choices=["loopback", "microphone"], default="loopback",
                        help="loopback: 스피커로 재생되는 소리(수업 프로그램·영상), microphone: 마이크")
    parser.add_argument("--wav", help="실시간 입력 대신 사용할 16-bit PCM WAV 파일")
    parser.add_argument("--device", default=None,
                        help="마이크는 장치 번호, 루프백은 출력 장치 이름 (--list-devices 참고). 생략하면 기본 장치")
    parser.add_argument("--list-devices", action="store_true", help="--source에 맞는 장치 목록을 출력하고 종료")
    parser.add_argument("--duration", type=float, default=None, help="실시간 입력 시간(초). 생략하면 Ctrl+C까지")
    parser.add_argument("--save", help="종료 후 세션 JSON 저장 경로")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-auto-gain", action="store_true", help="자동 음량 조절을 끄고 원본 크기로 인식(비교용)")
    args = parser.parse_args(argv)

    if args.list_devices:
        if args.source == "loopback":
            from .loopback import list_output_devices
            for name in list_output_devices():
                print(name)
        else:
            from .microphone import list_input_devices
            for index, name in list_input_devices():
                print(f"{index}: {name}")
        return 0
    device = args.device
    if args.source == "microphone" and device is not None:
        device = int(device)
    config = SttConfig(model_dir=args.model_dir, num_threads=args.threads, auto_gain=not args.no_auto_gain)
    # 모델 로드 시간이 세션 시간에 섞이지 않도록 세션 시작 전에 불러온다.
    engine = SherpaEngine(config)
    service = _PrintingService()
    sid = service.start_session("sherpa-stt demo", {
        "config_version": "sherpa-stt-demo", "participation_model": None, "stt_model": config.public_info()})
    try:
        if args.wav:
            transcriber = SherpaTranscriber(engine, service, sid, config.model_id, config.model_version, config.max_gap_ms,
                                            auto_gain=config.make_auto_gain())
            transcriber.report_status("running", f"WAV 파일 전사: {args.wav}")
            _run_wav(service, sid, transcriber, args.wav, config.chunk_ms)
            transcriber.report_status("stopped", "WAV 파일 전사를 마쳤습니다.")
        else:
            from .runner import LiveTranscription
            live = LiveTranscription(service, sid, config, source=args.source, device=device, engine=engine)
            if not live.start():
                print(f"\n시작 실패: {live.last_error}", file=sys.stderr)
                return 1
            hint = "수업 프로그램이나 영상을 재생하세요." if args.source == "loopback" else "말해 보세요."
            print(f"{hint} 종료: Ctrl+C", flush=True)
            try:
                deadline = None if args.duration is None else time.monotonic() + args.duration
                while deadline is None or time.monotonic() < deadline:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
            finally:
                live.stop()
                print(f"\n실시간 계측: {live.stats()}")
    finally:
        service.stop_session(sid)
    snapshot = service.get_dashboard_snapshot(sid)
    print(f"\n확정 발화 {len(snapshot['transcripts'])}개, 수업 길이 {snapshot['session']['duration_ms'] / 1000:.1f}s")
    if args.save:
        print(f"저장: {service.save_session(sid, args.save)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
