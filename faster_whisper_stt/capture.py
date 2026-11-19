"""콘솔 데모용 메모리 전용 오디오 입력 생성."""

SAMPLE_RATE_HZ = 16000
CHUNK_MS = 100


def make_capture(service, sid, source, device=None, target_process_id=None):
    if source == "microphone":
        from .microphone import MicrophoneCapture
        return MicrophoneCapture(service, sid, SAMPLE_RATE_HZ, CHUNK_MS, None if device is None else int(device))
    from .loopback import LoopbackCapture
    return LoopbackCapture(service, sid, SAMPLE_RATE_HZ, CHUNK_MS, speaker=device, process_id=target_process_id)
