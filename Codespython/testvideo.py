# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
import os

model = YOLO("D:/ET5/projet/Cyclope_Vision/runs/train/custom_yolov11_training_v3/weights/best.pt")

video_path = "video_de_test.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # 1 image / seconde

frame_count = 0
saved_count = 0

os.makedirs("frames_detected", exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            iou=0.5,
            device=0,
            verbose=False
        )

        annotated = results[0].plot()
        cv2.imwrite(f"frames_detected/frame_{saved_count:05d}.jpg", annotated)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"{saved_count} images traitées (1 par seconde)")
