import os
import json
import sys
import cv2
import pygame
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pygame.init()
pygame.font.init()

# Get the display dimensions
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

# Set up the screen
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("EyeGestures v3 example")
font_size = 48
bold_font = pygame.font.Font(None, font_size)
bold_font.set_bold(True)  # Set the font to bold

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')

from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

gestures = EyeGestures_v3()
cap = VideoCapture(0)
# MediaPipe Face Landmarker 초기화
base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

face_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)

timestamp = 0

x = np.arange(0, 1.1, 0.2)
y = np.arange(0, 1.1, 0.2)

xx, yy = np.meshgrid(x, y)

targets = [
    (0.6,0.3,0.1,0.05,"target_1"),
    (0.8,0.6,0.15,0.1,"target_2"),
    (0.1,0.9,0.1,0.1,"target_3")
]

calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
n_points = min(len(calibration_map),25)
np.random.shuffle(calibration_map)
gestures.uploadCalibrationMap(calibration_map,context="my_context")
gestures.setFixation(1.0)
# Initialize Pygame
# Set up colors
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
BLANK = (0,0,0)
WHITE = (255, 255, 255)

clock = pygame.time.Clock()

# Main game loop
running = True
iterator = 0
prev_x = 0
prev_y = 0
records = []
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                running = False


    # Generate new random position for the cursor
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # frame = np.rot90(frame)
    frame = np.flip(frame, axis=1)
    # MediaPipe 얼굴 분석
    timestamp += 33

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(frame)
    )

    face_result = face_detector.detect_for_video(
    mp_image,
    timestamp
)

    face_detected = bool(face_result.face_landmarks)

    calibrate = (iterator <= n_points)  # calibrate 25 points

    try:
        event, calibration = gestures.step(
            frame, calibrate,
            screen_width, screen_height,
            context="my_context"
        )

        # 한 프레임의 분석 결과 저장
        record = {
            "timestamp_ms": timestamp,
            "face_detected": face_detected,
            "gaze_point": event.point if event is not None else None,
            "fixation": event.fixation if event is not None else None,
            "saccades": event.saccades if event is not None else None
        }

        records.append(record)
    except TypeError as e:
        print("EyeGestures face detection failed:", e)
        continue

    if event is None:
        continue

    screen.fill((0, 0, 0))
    frame = pygame.surfarray.make_surface(frame)
    frame = pygame.transform.scale(frame, (400, 400))
    if event is not None or calibration is not None:
        # Display frame on Pygame screen
        screen.blit(
            pygame.surfarray.make_surface(
                np.rot90(event.sub_frame)
            ),
            (0, 0)
        )
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(f'{event.fixation}', False, (0, 0, 0))
        screen.blit(text_surface, (0,0))
        if calibrate:
            if calibration.point[0] != prev_x or calibration.point[1] != prev_y:
                iterator += 1
                prev_x = calibration.point[0]
                prev_y = calibration.point[1]
            # pygame.draw.circle(screen, GREEN, fit_point, calibration_radius)
            pygame.draw.circle(screen, BLUE, calibration.point, calibration.acceptance_radius)
            text_surface = bold_font.render(f"{iterator}/{n_points}", True, WHITE)
            text_square = text_surface.get_rect(center=calibration.point)
            screen.blit(text_surface, text_square)
        else:
            pass
        if gestures.whichAlgorithm(context="my_context") == "Ridge":
            pygame.draw.circle(screen, RED, event.point, 50)
        if gestures.whichAlgorithm(context="my_context") == "LassoCV":
            pygame.draw.circle(screen, BLUE, event.point, 50)
        if event.saccades:
            pygame.draw.circle(screen, GREEN, event.point, 50)

        for target in targets:
            pygame.draw.rect(
                screen,
                RED,
                pygame.Rect(
                int(target[0] * screen_width),
                int(target[1] * screen_height),
                int(target[2] * screen_width),
                int(target[3] * screen_height),
            ))
            text_surface = my_font.render(f'{target[4]}', False, (0, 0, 0))
            screen.blit(text_surface, (int(target[0] * screen_width),int(target[1] * screen_height)))


        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(f'{gestures.whichAlgorithm(context="my_context")}', False, (0, 0, 0))
        screen.blit(text_surface, event.point)
    
    # MediaPipe 얼굴 검출 상태 표시
    status = "Face Detected" if face_detected else "No Face"

    text_surface = bold_font.render(status, True, WHITE)
    screen.blit(text_surface, (20, 500))
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# JSON 파일 저장
with open("analysis_result.json", "w", encoding="utf-8") as f:
    json.dump(
        records,
        f,
        ensure_ascii=False,
        indent=4,
        default=str
    )

print("JSON file saved!")

# 리소스 정리
cap.close()
face_detector.close()
pygame.quit()