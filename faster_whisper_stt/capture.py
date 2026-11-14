"""기존 프로젝트의 메모리 전용 오디오 입력 재사용."""

def make_capture(service, sid, source, device=None, target_process_id=None):
    from sherpa_stt.config import SttConfig
    config = SttConfig()
    if source == "microphone":
        from sherpa_stt.microphone import MicrophoneCapture
        return MicrophoneCapture(service, sid, config.sample_rate_hz, config.chunk_ms,
                                 None if device is None else int(device))
    from sherpa_stt.loopback import LoopbackCapture
    return LoopbackCapture(service, sid, config.sample_rate_hz, config.chunk_ms,
                           process_id=target_process_id)
