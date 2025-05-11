import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog
from transformers import BlipProcessor, BlipForConditionalGeneration
from moviepy.video.io.VideoFileClip import VideoFileClip
import os

# Load YOLOv4 model files
yolo_cfg = "yolov4.cfg"  # Path to YOLOv4 configuration file
yolo_weights = "yolov4.weights"  # Path to YOLOv4 weights
yolo_names = "coco.names"  # Path to class names

# Load YOLOv4 network
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open(yolo_names, "r") as f:
    classes = f.read().strip().split("\n")

# Load BLIP Model for summarization
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("blip_model")
model = BlipForConditionalGeneration.from_pretrained("blip_model").to(device)

def extract_keyframes(video_path):
    """Extract keyframes from a video."""
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % int(frame_rate * 2) == 0:  # Capture a frame every 2 seconds
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def detect_objects(frame):
    """Run YOLOv4 object detection on a frame."""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    detected_objects = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                detected_objects.append(classes[class_id])

    return list(set(detected_objects))  # Return unique objects detected

def summarize_frame(frame):
    """Generate a caption for a frame using BLIP and YOLOv4 detections."""
    detected_objects = detect_objects(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
    summary = model.generate(**inputs)
    caption = processor.decode(summary[0], skip_special_tokens=True)

    # Combine BLIP caption with YOLO detections
    if detected_objects:
        caption += " | Objects detected: " + ", ".join(detected_objects)

    return caption

def summarize_video(video_path):
    """Summarize a video by extracting keyframes and captions."""
    frames = extract_keyframes(video_path)
    summaries = []

    for frame in frames:
        caption = summarize_frame(frame)
        summaries.append(caption)
    
    return summaries

def select_video():
    """Open file dialog to select a video."""
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        summary_text.set("Processing...")
        summaries = summarize_video(file_path)
        summary_text.set("\n".join(summaries))

# GUI Setup
root = tk.Tk()
root.title("Video Summarization with YOLOv4")
root.geometry("600x400")

summary_text = tk.StringVar()
summary_text.set("Select a video to summarize")

btn_select = tk.Button(root, text="Select Video", command=select_video, padx=10, pady=5)
btn_select.pack(pady=20)

lbl_summary = tk.Label(root, textvariable=summary_text, wraplength=500, justify="left")
lbl_summary.pack(pady=10)

root.mainloop()
