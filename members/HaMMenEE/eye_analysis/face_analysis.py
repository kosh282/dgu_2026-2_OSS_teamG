
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

timestamp = 0

# 눈의 위아래 간격 / 좌우 간격을 이용한 눈 개방 비율
def eye_ratio(landmarks, indices, w, h):
    points = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in indices
    ])

    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    return (vertical1 + vertical2) / (2.0 * horizontal)


# Face Landmarker의 눈 주변 랜드마크 인덱스
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 33
    result = detector.detect_for_video(mp_image, timestamp)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        left_ratio = eye_ratio(landmarks, LEFT_EYE, w, h)
        right_ratio = eye_ratio(landmarks, RIGHT_EYE, w, h)

        avg_ratio = (left_ratio + right_ratio) / 2

        # 임계값은 개인별 보정이 필요함
        eye_status = "OPEN" if avg_ratio > 0.20 else "CLOSED"

        # 머리 방향 추정용 얼굴 기준점
        image_points = np.array([
            [landmarks[1].x * w, landmarks[1].y * h],       # 코
            [landmarks[152].x * w, landmarks[152].y * h],   # 턱
            [landmarks[33].x * w, landmarks[33].y * h],     # 왼쪽 눈 바깥
            [landmarks[263].x * w, landmarks[263].y * h],   # 오른쪽 눈 바깥
            [landmarks[61].x * w, landmarks[61].y * h],     # 입 왼쪽
            [landmarks[291].x * w, landmarks[291].y * h]    # 입 오른쪽
        ], dtype=np.float64)

        model_points = np.array([
            [0, 0, 0],
            [0, -330, -65],
            [-225, 170, -135],
            [225, 170, -135],
            [-150, -150, -125],
            [150, -150, -125]
        ], dtype=np.float64)

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            pitch, yaw, roll = angles

            if yaw < -15:
                head_direction = "LEFT"
            elif yaw > 15:
                head_direction = "RIGHT"
            elif pitch < -15:
                head_direction = "UP"
            elif pitch > 15:
                head_direction = "DOWN"
            else:
                head_direction = "FRONT"

            cv2.putText(
                frame, f"Head: {head_direction}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2
            )

        cv2.putText(
            frame, f"Eyes: {eye_status} ({avg_ratio:.2f})",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2
        )

    cv2.imshow("Eye and Head Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
