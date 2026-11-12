# 사전 학습 시선 방향 후보

## 선택한 모델과 범위

[MobileGaze](https://github.com/yakhyo/gaze-estimation)의 MobileOne S0 ONNX 가중치를 사용한다.
공식 설명은 L2CS-Net을 기반으로 여러 backbone을 제공하며 Gaze360으로 학습했다고 밝힌다.
원본 L2CS-Net ResNet50/PyTorch 경로보다 작은 ONNX 배포 모델과 기존 ONNX Runtime을
활용하기 위해 선택했다. 더 정확하거나 발열이 적다는 실제 비교 결과는 아니다.

개인별 화면 지점 보정 없이 얼굴 crop에서 카메라 기준 시선 Yaw/Pitch를 추정한다.
영상 분석의 기존 MediaPipe 얼굴 경계를 재사용한다. 실제 화면 영역과 카메라 배치를 모르는
상태에서 화면 주시 여부를 확정할 수 없으므로 정면/이탈/불명확 후보로 표현한다.
정면 후보에서 벗어난 것만으로 비참여를 단정하지 않는다.

연결 범위는 단일 얼굴 독립 `engagement_model.webcam_demo --gaze`이다.
기존 webcam_test 및 Mac 재식별 데모, 프런트, 다인별 점수화에는 연결하지 않는다.
사용자가 여러 명인 수업 창에서도 작동하는 화면 주시 판별기로 설명하지 않는다.

## 사용

기본 영상 환경에 선택 `video_recognition/requirements-gaze.txt`를 설치한다.
`python -m video_recognition.download_gaze_model`은 공식 배포처에서 약 5MB의 모델을 받아
체크섬과 ONNX 로딩을 검사한다. 손상된 다운로드는 정상 파일명으로 남기지 않는다.
모델 파일은 git에서 제외한다. 실행 시 자동 다운로드하지 않는다.

Mac 실행:

```bash
video_recognition/.venv/bin/python -m engagement_model.webcam_demo --gaze --single-model --debug
```

Windows 실행:

```powershell
video_recognition/.venv/Scripts/python.exe -m engagement_model.webcam_demo --gaze --single-model --debug
```

시선 후보만 보기 위해 화면 점 보정을 할 필요는 없다. 기존 Gaussian 머리 각도 점수를
함께 보려면 C/Space를 눌러 정면을 약 3초 유지한다. 눈 기준 보정은 없다.
상태·점수 보류 조건은 [모델 README](../../engagement_model/README.md#선택-옵션-보정-없는-시선-방향-모델)에 있다.
눈 개방도 0.12 미만·80픽셀 미만 얼굴·다중 얼굴 등은 시선 불명확으로 둔다.
홍채 높이 규칙으로 방향을 분류하지 않는다. 눈 개방도는 감김/관측 품질 제외에만 쓴다.

### 아래 보기로 잘 연결되지 않을 때

현재 실제 사용에서 고개와 시선을 내려도 아래 보기로 잘 연결되지 않는 현상이 보고됐다.
원인은 아직 확정하지 않았으며, 아래 진단 표시는 원인 확인을 위한 보완이다.
진단 추가가 인식 정확도 문제를 해결했다는 의미는 아니다.

`Gaze reason`은 각도 경계 구간도 표시하며 눈 품질 필터에 걸리면 실제 EAR와
최솟값을 함께 표시한다. 눈을 아래로 향할 때 눈꺼풀도 좁아져 감김 필터에 걸리는지 확인한다.
`Down gate` 행은 머리 Pitch≥15°, 모델 시선 Pitch≤−30°, abs(Yaw)<35°,
측정 최신성 및 확인된 추론 구간 길이를 표시한다. `no`인 조건으로 어디서 막히는지 구분한다.
모델이 실제 하향에서 어떤 시선 각도를 내는지 확인하기 전 임계값을 임의로 완화하지 않는다.
이 진단 정보도 메모리/화면 전용이며 저장하지 않는다.

## 검증과 성능 해석

현재 Mac 환경은 Python 3.13.9와 ONNX Runtime 1.31.1이다. 공식 ONNX의
입력 `[1,3,448,448]`, 출력 `yaw`/`pitch` 각각 `[1,90]` 로딩을 확인했다.
검은 합성 입력의 워밍업 후 10회 측정에서 전처리 포함 평균 33.32ms, 최대 36.6ms였다.
두 CPU 스레드 설정이며 실제 얼굴 정확도·카메라 처리 지연·발열 개선이나 Windows 검증 결과가 아니다.
합성 영상에 얼굴 품질을 강제로 유효로 제공해 추론만 실행했으므로 해당 예측의 의미는 평가하지 않는다.

실제 시험은 정면/눈만 좌우·아래/고개만 숙임/고개+눈 아래/깜빡임/안경/조명/자리 비움으로 나누고,
후보 오판·unknown 비율과 영상 지연을 함께 확인해야 한다. 현재 임계값은 검증 전 실험값이다.

## 출처와 데이터 경계

- 모델: https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx
- SHA256: `8b4fdc4e3da44733c9a82e7776b411e4a39f94e8e285aee0fc85a548a55f7d9f`
- 전처리·90-bin 복원 근거: https://github.com/yakhyo/gaze-estimation/blob/main/onnx_inference.py
- 프로젝트 라이선스: https://github.com/yakhyo/gaze-estimation/blob/main/LICENSE
- L2CS-Net: https://github.com/Ahmednull/L2CS-Net

추론은 로컬에서 수행한다. 얼굴 crop은 추론 중에만 사용하고 저장·업로드하지 않는다.
시선 각도·임시 캐시는 메모리와 실험 패널에만 두며 FeatureWindow·세션 JSON에 추가하지 않는다.
프런트 계약과 영구 개인 식별자는 추가하지 않는다. 공식 사전 학습 가중치 외 원본 데이터를 받지 않는다.
