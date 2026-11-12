"""공식 MobileGaze 배포 가중치만 다운로드한다. 미디어는 다루지 않는다."""
from pathlib import Path
import hashlib
import os
import tempfile
from urllib.request import urlopen

from .gaze_estimation import DEFAULT_MODEL

MODEL_URL = 'https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx'
MODEL_SHA256 = '8b4fdc4e3da44733c9a82e7776b411e4a39f94e8e285aee0fc85a548a55f7d9f'


def main():
    target = Path(DEFAULT_MODEL)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('기존 모델의 체크섬이 다릅니다. 파일을 확인하세요. 자동으로 덮어쓰지 않습니다.')
        print(f'Already exists: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
            with urlopen(MODEL_URL, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError('공식 가중치의 체크섬이 예상값과 다릅니다.')
        import onnxruntime as ort
        # 다운로드 오류 문서나 손상된 파일을 정상 모델 이름으로 옮기지 않는다.
        ort.InferenceSession(str(temporary), providers=['CPUExecutionProvider'])
        os.replace(temporary, target)
        print(f'Downloaded: {target}')
        print('SHA256:', hashlib.sha256(target.read_bytes()).hexdigest())
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()
