import cv2
import os
os.makedirs("frames", exist_ok=True)
cap = cv2.VideoCapture("input.mp4")
fps_video = cap.get(cv2.CAP_PROP_FPS)
step = max(1, int(fps_video / 3))
count = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % step == 0:
        cv2.imwrite("frames/frame_%04d.png" % saved, frame)
        saved += 1
    count += 1
cap.release()
print("Извлечено", saved, "кадров из", count)
