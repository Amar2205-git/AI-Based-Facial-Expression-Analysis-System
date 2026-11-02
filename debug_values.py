import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)
timestamp = 0

def pt(lms, i, w, h):
    return np.array([lms[i].x * w, lms[i].y * h])

def dist(a, b):
    return np.linalg.norm(a - b)

print("Make each expression and watch the printed values.")
print("Press ESC to quit.\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1
    frame_count += 1

    if result.face_landmarks and frame_count % 15 == 0:
        lms = result.face_landmarks[0]
        face_h = dist(pt(lms, 10, w, h), pt(lms, 152, w, h))

        top_lip    = pt(lms, 13, w, h)
        bot_lip    = pt(lms, 14, w, h)
        left_mouth = pt(lms, 61, w, h)
        right_mouth= pt(lms, 291, w, h)

        mouth_open  = dist(top_lip, bot_lip) / face_h
        mouth_width = dist(left_mouth, right_mouth) / face_h
        mouth_center_y = (top_lip[1] + bot_lip[1]) / 2
        corners_y      = (left_mouth[1] + right_mouth[1]) / 2
        smile_score    = mouth_center_y - corners_y

        def ear(top, bot, l, r):
            return dist(pt(lms, top, w, h), pt(lms, bot, w, h)) / (dist(pt(lms, l, w, h), pt(lms, r, w, h)) + 1e-6)

        avg_ear = (ear(159,145,33,133) + ear(386,374,362,263)) / 2

        l_brow_raise = dist(pt(lms,70,w,h), pt(lms,159,w,h)) / face_h
        r_brow_raise = dist(pt(lms,300,w,h), pt(lms,386,w,h)) / face_h
        avg_brow     = (l_brow_raise + r_brow_raise) / 2
        brow_furrow  = dist(pt(lms,107,w,h), pt(lms,336,w,h)) / face_h
        brow_asym    = abs(l_brow_raise - r_brow_raise)

        print(f"mouth_open={mouth_open:.4f}  mouth_width={mouth_width:.4f}  smile_score={smile_score:.2f}")
        print(f"avg_ear={avg_ear:.4f}  avg_brow={avg_brow:.4f}  brow_furrow={brow_furrow:.4f}  brow_asym={brow_asym:.4f}")
        print("-" * 60)

    cv2.putText(frame, "Make expressions - watch terminal", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Debug", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
