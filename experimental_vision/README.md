# 영상 분석 통합 실험

실험 코드는 영상 담당 폴더의 공통 구현으로 통합했습니다. 사용법은 [영상 분석 안내](../video_recognition/README.md)를 참고하세요. 이 폴더에는 공통 구현을 직접 확인하는 테스트를 유지하며, 중복 호환 경로인 `vision.py`는 제거했습니다.

독립 웹캠 데모와 MobileGaze·EyeGestures, Gaussian 보정 비교·자동 진단은 유지합니다. Legacy/Huang 모델 비교와 상체 pose 측정은 제거했습니다.
