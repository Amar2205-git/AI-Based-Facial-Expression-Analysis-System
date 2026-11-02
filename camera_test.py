import cv2
import os

print("Force DirectShow camera opening...")

# Disable MSMF completely
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not detected at index 0 or 1")
    exit()

print("✅ Camera opened successfully")

cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        continue

    cv2.imshow("Camera Output", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()