import cv2
import mediapipe as mp
import os

print("Running FaceMesh with new Mediapipe API...")

# Disable MSMF (your system needs this)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

# NEW IMPORT STYLE (important)
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False
)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        print("FaceMesh detected ✅")

        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

    cv2.imshow("FaceMesh", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()