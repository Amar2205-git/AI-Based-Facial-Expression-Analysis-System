import cv2
import mediapipe as mp

print("Starting face mesh with green dots...")

# Open camera (Windows: CAP_MSMF works well)
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera failed")
    exit()

# Use MediaPipe Face Mesh to get facial landmarks
use_face_mesh = False
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                      refine_landmarks=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    use_face_mesh = True
except Exception:
    print("Mediapipe Face Mesh unavailable — falling back to Haar cascade dots")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = None
    if use_face_mesh:
        results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if use_face_mesh:
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw small green dots for each landmark
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    else:
        # Haar cascade fallback: draw a grid of green dots inside detected face boxes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (fx, fy, fw, fh) in faces:
            cols = 6
            rows = 6
            for i in range(rows):
                for j in range(cols):
                    cx = fx + int((j + 1) * fw / (cols + 1))
                    cy = fy + int((i + 1) * fh / (rows + 1))
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    cv2.imshow("Face Dots", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()