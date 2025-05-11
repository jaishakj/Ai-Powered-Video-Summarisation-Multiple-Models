
import tkinter as tk
from tkinter import filedialog
from summarize import summarize_video

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if file_path:
        summary_text.set("Processing...")
        root.update()
        summaries = summarize_video(file_path)
        summary_text.set("\n".join(summaries))

# GUI Setup
root = tk.Tk()
root.title("YOLOv5 + BLIP Video Summarizer")
root.geometry("700x500")

summary_text = tk.StringVar()
summary_text.set("Select a video to summarize")

btn_select = tk.Button(root, text="Select Video", command=select_video, padx=10, pady=5)
btn_select.pack(pady=20)

lbl_summary = tk.Label(root, textvariable=summary_text, wraplength=650, justify="left")
lbl_summary.pack(pady=10)

root.mainloop()
