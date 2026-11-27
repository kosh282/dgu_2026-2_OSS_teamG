# 영상 분석 통합 실험

영상 실험의 공통 구현은 [영상 분석](../video_recognition/README.md)에 통합했습니다. 이 폴더에는 통합 구현을 확인하는 테스트가 있습니다. 독립 웹캠과 시선·보정 진단은 [모델 안내](../engagement_model/README.md)를 참고하세요.

## 장치 없는 회귀 검사

[루트 설치 안내](../README.md#설치)에 따라 환경을 준비한 뒤 저장소 루트에서 실행합니다.

```powershell
video_recognition/.venv/Scripts/python.exe -B -m unittest discover -s experimental_vision -v
```
