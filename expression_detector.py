import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import subprocess
import threading

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera not opened")
    exit()

base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)
timestamp = 0

def speak(text):
    cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
    threading.Thread(
        target=lambda: subprocess.run(
            ["powershell", "-Command", cmd],
            creationflags=subprocess.CREATE_NO_WINDOW
        ), daemon=True
    ).start()

def pt(lms, i, w, h):
    return np.array([lms[i].x * w, lms[i].y * h])

def dist(a, b):
    return np.linalg.norm(a - b)

def detect_expression(lms, w, h):
    face_h = dist(pt(lms, 10, w, h), pt(lms, 152, w, h))
    if face_h == 0:
        return "Neutral", (200, 200, 200)

    # Mouth
    top_lip    = pt(lms, 13, w, h)
    bot_lip    = pt(lms, 14, w, h)
    left_mouth = pt(lms, 61, w, h)
    right_mouth= pt(lms, 291, w, h)

    mouth_open  = dist(top_lip, bot_lip) / face_h
    mouth_width = dist(left_mouth, right_mouth) / face_h

    # Smile: corners of mouth vs center
    mouth_center_y = (top_lip[1] + bot_lip[1]) / 2
    corners_y      = (left_mouth[1] + right_mouth[1]) / 2
    smile_score    = mouth_center_y - corners_y  # positive = smile

    # Eyes
    def ear(top, bot, l, r):
        return dist(pt(lms, top, w, h), pt(lms, bot, w, h)) / (dist(pt(lms, l, w, h), pt(lms, r, w, h)) + 1e-6)

    left_ear  = ear(159, 145, 33, 133)
    right_ear = ear(386, 374, 362, 263)
    avg_ear   = (left_ear + right_ear) / 2

    # Eyebrows
    l_brow       = pt(lms, 70, w, h)
    r_brow       = pt(lms, 300, w, h)
    l_brow_inner = pt(lms, 107, w, h)
    r_brow_inner = pt(lms, 336, w, h)
    l_eye_top    = pt(lms, 159, w, h)
    r_eye_top    = pt(lms, 386, w, h)

    l_brow_raise = dist(l_brow, l_eye_top) / face_h
    r_brow_raise = dist(r_brow, r_eye_top) / face_h
    avg_brow     = (l_brow_raise + r_brow_raise) / 2
    brow_furrow  = dist(l_brow_inner, r_brow_inner) / face_h
    brow_asym    = abs(l_brow_raise - r_brow_raise)

    # --- Detection rules ---
    # Surprised: mouth open + wide eyes + raised brows
    if mouth_open > 0.055 and avg_ear > 0.32 and avg_brow > 0.082:
        return "Surprised! 😮", (0, 255, 255)

    # Happy: wide mouth + smile (corners up)
    if mouth_width > 0.27 and smile_score > 1.5:
        return "Happy! 😊", (0, 255, 0)

    # Angry: strong frown + tight mouth (tuned to your face)
    if smile_score < -3.0 and mouth_open < 0.03 and brow_furrow < 0.20:
        return "Angry! 😠", (0, 0, 255)

    # Sad: mouth corners down, not as extreme as angry
    if smile_score < -2.0 and mouth_open < 0.03 and brow_furrow > 0.19:
        return "Sad! 😔", (100, 100, 255)

    # Confused: asymmetric brows
    if brow_asym > 0.013:
        return "Confused! 😕", (0, 165, 255)

    # Disgust: wide mouth open + frown or neutral smile
    if mouth_open > 0.06 and smile_score < 2.0:
        return "Disgust! 🤮", (0, 180, 80)

    return "Neutral 😐", (200, 200, 200)


last_spoken = ""
speak_counter = 0
SPEAK_INTERVAL = 60  # frames (~2 sec at 30fps)

print("Auto expression detection running. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    expression, color = "No Face Detected", (100, 100, 100)

    if result.face_landmarks:
        lms = result.face_landmarks[0]

        for lm in lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (0, 200, 0), -1)

        expression, color = detect_expression(lms, w, h)

        # Speak when expression changes, every SPEAK_INTERVAL frames
        speak_counter += 1
        if speak_counter >= SPEAK_INTERVAL:
            speak_counter = 0
            word = expression.split()[0].replace("!", "")
            if word != last_spoken:
                last_spoken = word
                speak(f"You look {word}")

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 55), (30, 30, 30), -1)
    cv2.putText(frame, expression, (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    cv2.imshow("Expression Detector", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
