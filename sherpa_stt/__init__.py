"""sherpa-onnx 기반 한국어 스트리밍 STT 모듈.

sherpa-onnx·sounddevice는 실제 모델·마이크를 사용할 때만 불러온다.
"""
from .audio import AudioChunk, to_mono_float32
from .config import SttConfig
from .transcriber import SherpaTranscriber

__all__ = ["AudioChunk", "SttConfig", "SherpaTranscriber", "to_mono_float32"]
