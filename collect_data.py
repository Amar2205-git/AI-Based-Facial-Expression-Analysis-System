import cv2
import os

label = "confused"   # change later to not_confused

save_path = f"dataset/{label}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

count = 0

print("Press 's' to save image")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Collecting Data", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        file_name = f"{save_path}/{count}.jpg"
        cv2.imwrite(file_name, frame)
        print("Saved:", file_name)
        count += 1

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()