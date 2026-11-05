"""한국어 스트리밍 zipformer 모델을 sherpa_stt/models/ 아래에 내려받는다.

  python -m sherpa_stt.download_model

모델 파일은 .gitignore로 커밋에서 제외한다.
"""
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .config import DEFAULT_MODEL_ID, DEFAULT_MODELS_ROOT

URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2"


def download(model_id=DEFAULT_MODEL_ID, root=DEFAULT_MODELS_ROOT):
    root = Path(root)
    target = root / model_id
    if (target / "tokens.txt").is_file():
        print(f"이미 있습니다: {target}")
        return target
    root.mkdir(parents=True, exist_ok=True)
    url = URL.format(name=model_id)
    with tempfile.TemporaryDirectory(dir=root) as folder:
        archive = Path(folder) / f"{model_id}.tar.bz2"
        print(f"내려받는 중: {url}")
        urllib.request.urlretrieve(url, archive)
        print("압축 해제 중")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(root, filter="data")
    if not (target / "tokens.txt").is_file():
        raise RuntimeError(f"압축 해제 후 {target}/tokens.txt를 찾을 수 없습니다")
    print(f"모델 준비 완료: {target}")
    return target


if __name__ == "__main__":
    download(*sys.argv[1:2])
