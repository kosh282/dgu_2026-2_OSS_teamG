# Eye Analysis

EyeGestures V3와 MediaPipe Face Landmarker를 이용한
실시간 시선 및 얼굴 분석 모듈입니다.

## 1. Environment
- Python 3.10

## 2. Install dependencies

```bash
pip install -r requirements.txt


3. MediaPipe model
MediaPipe Face Landmarker 모델 파일이 필요합니다.

모델 파일명: face_landmarker.task

모델 다운로드:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

다운로드한 파일을 다음 위치에 저장합니다.

models/face_landmarker.task
4. Run
python integrated_analysis.py
5. Output
분석 결과는 JSON 파일로 저장됩니다.
웹캠 영상 및 개인 분석 결과 파일은 저장소에 업로드하지 않습니다.
