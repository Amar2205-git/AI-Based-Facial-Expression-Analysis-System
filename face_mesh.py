import cv2
import mediapipe as mp
import os

print("Testing Minimal FaceMesh...")

# Disable MSMF
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

mp_face_mesh = mp.solutions.face_mesh

# VERY BASIC SETTINGS
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1
)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        print("FACE MESH WORKING ✅")

    cv2.imshow("Test Window", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()