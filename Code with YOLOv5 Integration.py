import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from transformers import BlipProcessor, BlipForConditionalGeneration
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
from ultralytics import YOLO  # Import YOLOv5

# Load YOLOv5 model for object detection
yolo_model = YOLO("yolov5s.pt")  # Load the small version of YOLOv5

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
    """Run YOLOv5 object detection on a frame."""
    results = yolo_model(frame)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class ID
            class_name = yolo_model.names[class_id]  # Get class name
            detected_objects.append(class_name)

    return detected_objects

def summarize_frame(frame):
    """Generate a caption for a frame using BLIP and YOLOv5 detections."""
    detected_objects = detect_objects(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
    summary = model.generate(**inputs)
    caption = processor.decode(summary[0], skip_special_tokens=True)

    # Combine BLIP caption with YOLO detections
    if detected_objects:
        caption += " | Objects detected: " + ", ".join(set(detected_objects))

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
root.title("Video Summarization with YOLOv5")
root.geometry("600x400")

summary_text = tk.StringVar()
summary_text.set("Select a video to summarize")

btn_select = tk.Button(root, text="Select Video", command=select_video, padx=10, pady=5)
btn_select.pack(pady=20)

lbl_summary = tk.Label(root, textvariable=summary_text, wraplength=500, justify="left")
lbl_summary.pack(pady=10)

root.mainloop()
