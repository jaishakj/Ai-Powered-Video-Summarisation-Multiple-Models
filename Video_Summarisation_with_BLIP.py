import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import threading

# Load BLIP Model for summarization
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model, device

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

def summarize_frame(frame, processor, model, device):
    """Generate a caption for a frame using BLIP."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        summary = model.generate(**inputs, max_length=30)
    return processor.decode(summary[0], skip_special_tokens=True)

def summarize_video(video_path, processor, model, device, progress_callback=None):
    """Summarize a video by extracting keyframes and captions."""
    frames = extract_keyframes(video_path)
    summaries = []
    
    for i, frame in enumerate(frames):
        caption = summarize_frame(frame, processor, model, device)
        summaries.append(caption)
        
        # Update progress if callback is provided
        if progress_callback:
            progress = (i + 1) / len(frames) * 100
            progress_callback(progress, f"Processing frame {i+1}/{len(frames)}")
    
    return summaries

class VideoSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Summarization")
        self.root.geometry("700x500")
        
        # Initialize model
        self.processor, self.model, self.device = load_model()
        
        # Create GUI elements
        self.setup_gui()
    
    def setup_gui(self):
        # Frame for buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # Select video button
        self.btn_select = tk.Button(btn_frame, text="Select Video", command=self.select_video, padx=10, pady=5)
        self.btn_select.pack(side=tk.LEFT, padx=5)
        
        # Save summary button
        self.btn_save = tk.Button(btn_frame, text="Save Summary", command=self.save_summary, padx=10, pady=5)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        self.btn_save.config(state=tk.DISABLED)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(self.root, variable=self.progress_var, 
                                    from_=0, to=100, orient=tk.HORIZONTAL, 
                                    length=600, state=tk.DISABLED)
        self.progress_bar.pack(pady=5, padx=20, fill=tk.X)
        
        # Status label
        self.status_text = tk.StringVar()
        self.status_text.set("Select a video to summarize")
        self.lbl_status = tk.Label(self.root, textvariable=self.status_text)
        self.lbl_status.pack(pady=5)
        
        # Summary text area
        self.summary_frame = tk.Frame(self.root)
        self.summary_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
        
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD, height=15)
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.summary_frame, command=self.summary_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_text.config(yscrollcommand=scrollbar.set)
        
        # Video path
        self.video_path = None
        self.summaries = []
    
    def select_video(self):
        """Open file dialog to select a video."""
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_path = file_path
            self.status_text.set(f"Selected video: {os.path.basename(file_path)}")
            self.btn_select.config(state=tk.DISABLED)
            
            # Clear previous summaries
            self.summary_text.delete(1.0, tk.END)
            self.summaries = []
            
            # Start processing in a separate thread
            thread = threading.Thread(target=self.process_video)
            thread.daemon = True
            thread.start()
    
    def process_video(self):
        """Process the video in a separate thread."""
        try:
            # Enable progress bar
            self.root.after(0, lambda: self.progress_bar.config(state=tk.NORMAL))
            
            # Process video
            self.summaries = summarize_video(
                self.video_path, 
                self.processor, 
                self.model, 
                self.device,
                self.update_progress
            )
            
            # Update UI with results
            self.root.after(0, self.update_summary)
            
        except Exception as e:
            # Handle errors
            self.root.after(0, lambda: self.status_text.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.btn_select.config(state=tk.NORMAL))
    
    def update_progress(self, progress_value, status_message):
        """Update progress bar and status message from the processing thread."""
        self.root.after(0, lambda: self.progress_var.set(progress_value))
        self.root.after(0, lambda: self.status_text.set(status_message))
    
    def update_summary(self):
        """Update the summary text area with results."""
        # Join summaries with timestamps
        formatted_summaries = []
        for i, summary in enumerate(self.summaries):
            time_seconds = i * 2  # Since we capture every 2 seconds
            minutes = time_seconds // 60
            seconds = time_seconds % 60
            timestamp = f"{minutes:02d}:{seconds:02d}"
            formatted_summaries.append(f"[{timestamp}] {summary}")
        
        # Update text area
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "\n\n".join(formatted_summaries))
        
        # Enable save button and select video button
        self.btn_save.config(state=tk.NORMAL)
        self.btn_select.config(state=tk.NORMAL)
        
        # Update status
        self.status_text.set(f"Summarization complete: {len(self.summaries)} keyframes processed")
        
        # Disable progress bar
        self.progress_bar.config(state=tk.DISABLED)
    
    def save_summary(self):
        """Save the generated summary to a text file."""
        if not self.summaries:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"{os.path.splitext(os.path.basename(self.video_path))[0]}_summary.txt"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                for i, summary in enumerate(self.summaries):
                    time_seconds = i * 2
                    minutes = time_seconds // 60
                    seconds = time_seconds % 60
                    timestamp = f"{minutes:02d}:{seconds:02d}"
                    f.write(f"[{timestamp}] {summary}\n\n")
            
            self.status_text.set(f"Summary saved to {os.path.basename(file_path)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoSummarizerApp(root)
    root.mainloop()
