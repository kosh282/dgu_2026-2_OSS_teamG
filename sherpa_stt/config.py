"""sherpa-onnx 스트리밍 STT 설정. 값은 검증 전 실험 후보다."""
from dataclasses import dataclass
import math
from pathlib import Path

DEFAULT_MODEL_ID = "sherpa-onnx-streaming-zipformer-korean-2024-06-16"
DEFAULT_MODELS_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_DIR = str(DEFAULT_MODELS_ROOT / DEFAULT_MODEL_ID)


@dataclass(frozen=True)
class SttConfig:
    """스트리밍 transducer 모델과 엔드포인트·오디오 입력 설정.

    시간 단위: 엔드포인트 규칙은 초, 그 밖의 `_ms` 항목은 밀리초.
    `model_dir` 기본값은 `python -m sherpa_stt.download_model`이 받는 위치다.
    """
    model_dir: str = DEFAULT_MODEL_DIR
    model_id: str = DEFAULT_MODEL_ID
    model_version: str = "2024-06-16"
    sample_rate_hz: int = 16000
    feature_dim: int = 80
    num_threads: int = 2
    provider: str = "cpu"
    decoding_method: str = "greedy_search"
    # sherpa-onnx 엔드포인트 규칙
    # rule1: 아무 글자도 인식되지 않은 상태로 이 시간만큼 무음이면 구간 리셋
    # rule2: 글자가 인식된 뒤 이 시간만큼 무음이면 발화 확정
    # rule3: 발화 길이가 이 시간을 넘으면 강제 확정
    rule1_min_trailing_silence: float = 2.4
    rule2_min_trailing_silence: float = 1.2
    rule3_min_utterance_length: float = 20.0
    # 연속 청크 사이 허용 공백. 이를 넘으면 오디오 누락으로 보고 현재 발화를 확정한다.
    max_gap_ms: int = 500
    # 마이크 입력 청크 길이
    chunk_ms: int = 100

    def __post_init__(self):
        if not isinstance(self.model_dir, str) or not self.model_dir.strip():
            raise ValueError("model_dir는 비어 있지 않은 문자열이어야 합니다")
        for name in ("model_id", "model_version", "provider", "decoding_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}은 비어 있지 않은 문자열이어야 합니다")
        for name in ("sample_rate_hz", "feature_dim", "num_threads", "chunk_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}은 양의 정수여야 합니다")
        if isinstance(self.max_gap_ms, bool) or not isinstance(self.max_gap_ms, int) or self.max_gap_ms < 0:
            raise ValueError("max_gap_ms는 0 이상의 정수여야 합니다")
        for name in ("rule1_min_trailing_silence", "rule2_min_trailing_silence", "rule3_min_utterance_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}은 양의 유한한 실수여야 합니다")

    def public_info(self):
        """세션 JSON `config`에 넣을 수 있는 공개 설정. 로컬 경로는 제외한다."""
        return {
            "model_id": self.model_id, "model_version": self.model_version,
            "sample_rate_hz": self.sample_rate_hz, "decoding_method": self.decoding_method,
            "endpoint": {
                "rule1_min_trailing_silence": self.rule1_min_trailing_silence,
                "rule2_min_trailing_silence": self.rule2_min_trailing_silence,
                "rule3_min_utterance_length": self.rule3_min_utterance_length,
            },
            "max_gap_ms": self.max_gap_ms, "chunk_ms": self.chunk_ms,
        }
