
import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov5s.pt")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def extract_keyframes(video_path, interval_sec=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)
    frame_count = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def detect_objects(frame):
    results = yolo_model(frame)
    objects = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            objects.add(yolo_model.names[class_id])
    return list(objects)

def summarize_frame(frame):
    detected_objects = detect_objects(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
    summary = model.generate(**inputs)
    caption = processor.decode(summary[0], skip_special_tokens=True)
    if detected_objects:
        caption += " | Objects detected: " + ", ".join(detected_objects)
    return caption

def summarize_video(video_path):
    frames = extract_keyframes(video_path)
    return [summarize_frame(f) for f in frames]
