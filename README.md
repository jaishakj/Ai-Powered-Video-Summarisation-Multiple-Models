# Ai-Powered-Video-Summarisation-Using-Yolov5 + BLIP

This project performs automatic video summarization by extracting keyframes using OpenCV, detecting objects using **YOLOv5**, and generating natural language captions for those frames using **BLIP** (Bootstrapped Language Image Pretraining). It combines **Computer Vision** and **Natural Language Processing** to produce concise and informative video summaries.

---

## 📌 Features

- 🎥 Extract keyframes from videos
- 🧩 Detect objects in frames using YOLOv5
- 🧠 Generate frame-wise captions using BLIP (transformer-based image captioning)
- 📋 Display the summary in a clean GUI using Tkinter
- ⚡ Runs on CPU or GPU (CUDA-enabled)

---

## 🛠 Requirements

### Python (≥ 3.8)

### Python Packages
Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Setup Instructions
1. Clone the Repository
```bash
git clone https://github.com/your-username/video-summarizer-blip-yolov5.git
cd video-summarizer-blip-yolov5
```
2. Download Pre-trained Models
BLIP Model (from HuggingFace):
The BLIP model will be auto-downloaded:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base"
```

YOLOv5 Model:
```bash

pip install ultralytics
```
This will install YOLOv5 and automatically download the default yolov5s.pt model on first run.

### 📦 Usage
## GUI Mode
```bash

python main.py
```
~Click Select Video
~The application will extract frames, detect objects, and generate captions.
~Summarized captions will appear on the GUI.

## Headless Mode (optional - CLI)
```python

from summarize import summarize_video
summarize_video("sample.mp4")
```

### 📁 Project Structure
```bash

.
├── main.py                  # GUI + Core logic
├── summarize.py             # Summarization logic
├── yolov5s.pt               # YOLOv5 model (auto-downloaded)
├── README.md
├── requirements.txt
```

### 🧠 Models Used
## 🟦 YOLOv5
-Version: YOLOv5s (Ultralytics)
-Use: Object detection on each keyframe
-Source: https://github.com/ultralytics/yolov5

## 🟩 BLIP
-Model: Salesforce/blip-image-captioning-base
-Use: Generate captions for frames
-HuggingFace: https://huggingface.co/Salesforce/blip-image-captioning-base

### 📸 Example Output
```less

Frame 1: A man sitting on a bench in a park. | Objects detected: person, bench, dog  
Frame 2: A car is parked on a busy street. | Objects detected: car, truck, person  
...
```
### 🧪 Tests

-Tested on .mp4 and .avi formats.
-Runs on both Windows and Linux (Tkinter GUI supported).

### ⚠️ Known Issues

-YOLOv5 detection may be inaccurate on low-resolution frames.
-BLIP processing is slower on CPU; use CUDA-enabled GPU for best performance.
-Tkinter GUI may freeze on very large videos (>300MB)

### 📬 Contribution Guidelines

-Fork the repo
-Create a new branch (git checkout -b feature-name)
-Commit your changes (git commit -m "Add feature")
-Push to the branch (git push origin feature-name)
-Create a pull request
