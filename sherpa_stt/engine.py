"""sherpa-onnx OnlineRecognizer 래퍼.

전사기(`transcriber.py`)는 이 클래스의 메서드만 사용하므로 테스트에서는
같은 메서드를 가진 가짜 엔진으로 대체할 수 있다.
"""
import json
from pathlib import Path

import numpy as np

_LEADING_PUNCT = ".,?!…·"


def _pick(model_dir, prefix, prefer_int8):
    candidates = sorted(model_dir.glob(f"{prefix}*.onnx"))
    if not candidates:
        raise FileNotFoundError(f"{model_dir}에서 {prefix}*.onnx 파일을 찾을 수 없습니다")
    int8 = [c for c in candidates if ".int8." in c.name]
    full = [c for c in candidates if ".int8." not in c.name]
    ordered = (int8 + full) if prefer_int8 else (full + int8)
    return str(ordered[0])


def find_transducer_files(model_dir):
    """모델 폴더에서 tokens/encoder/decoder/joiner 파일을 찾는다.

    encoder·joiner는 int8 양자화본을, decoder는 원본을 우선한다.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"모델 폴더가 없습니다: {root} (python -m sherpa_stt.download_model 로 내려받으세요)")
    tokens = root / "tokens.txt"
    if not tokens.is_file():
        raise FileNotFoundError(f"tokens.txt가 없습니다: {tokens}")
    return {
        "tokens": str(tokens),
        "encoder": _pick(root, "encoder", True),
        "decoder": _pick(root, "decoder", False),
        "joiner": _pick(root, "joiner", True),
    }


class SherpaEngine:
    """스트리밍 zipformer transducer 인식기. 스트림 하나당 한 스레드에서 사용한다."""

    def __init__(self, config):
        try:
            import sherpa_onnx
        except ImportError as error:
            raise ImportError("sherpa-onnx가 설치되지 않았습니다. sherpa_stt/requirements.txt를 설치하세요") from error
        files = find_transducer_files(config.model_dir)
        self._sample_rate_hz = config.sample_rate_hz
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            **files,
            num_threads=config.num_threads,
            sample_rate=config.sample_rate_hz,
            feature_dim=config.feature_dim,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=config.rule1_min_trailing_silence,
            rule2_min_trailing_silence=config.rule2_min_trailing_silence,
            rule3_min_utterance_length=config.rule3_min_utterance_length,
            decoding_method=config.decoding_method,
            provider=config.provider,
        )

    def create_stream(self):
        return self._recognizer.create_stream()

    def accept_waveform(self, stream, sample_rate_hz, samples):
        # 모델 샘플레이트와 다르면 sherpa-onnx가 내부에서 리샘플링한다.
        stream.accept_waveform(sample_rate_hz, samples)

    def decode(self, stream):
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

    def get_result(self, stream):
        """(텍스트, 토큰 구간)을 반환한다.

        - 텍스트: sherpa-onnx의 `text`는 CJK 문자 사이 공백을 지워 한국어 띄어쓰기가 사라진다.
          BPE 토큰에는 단어 경계 공백(" 보여")이 남아 있으므로 토큰을 직접 이어 붙인다.
          직전 발화에서 넘어온 앞쪽 구두점 토큰은 버린다.
        - 토큰 구간: 첫·마지막 토큰의 스트림 기준 오디오 시각(초). 토큰이 없으면 None.
          타임스탬프는 리셋 시점(`start_time`) 기준 상대값이므로 더해서 절대값으로 만든다.
        """
        data = json.loads(self._recognizer.get_result_as_json_string(stream))
        tokens = list(data.get("tokens") or [])
        stamps = list(data.get("timestamps") or [])
        if len(stamps) != len(tokens):
            stamps = []
        while tokens and not tokens[0].replace("▁", " ").strip(_LEADING_PUNCT + " "):
            tokens.pop(0)
            if stamps:
                stamps.pop(0)
        if not tokens:
            return "", None
        text = " ".join("".join(tokens).replace("▁", " ").split())
        if not stamps:
            return text, None
        base = float(data.get("start_time") or 0.0)
        return text, (base + stamps[0], base + stamps[-1])

    def get_text(self, stream):
        return self.get_result(stream)[0]

    def is_endpoint(self, stream):
        return bool(self._recognizer.is_endpoint(stream))

    def reset(self, stream):
        self._recognizer.reset(stream)

    def input_finished(self, stream):
        # 마지막 프레임까지 디코딩되도록 꼬리 무음을 넣는다(sherpa-onnx 예제와 같은 0.66초).
        # 인식기 내부 처리용이며 이벤트 시각에는 반영하지 않는다.
        stream.accept_waveform(self._sample_rate_hz, np.zeros(int(0.66 * self._sample_rate_hz), dtype=np.float32))
        stream.input_finished()
