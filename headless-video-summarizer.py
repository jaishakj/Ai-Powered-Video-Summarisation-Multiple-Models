import cv2
import torch
import argparse
import os
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

def summarize_video(video_path, interval=2.0, output_file=None, use_gpu=True):
    """
    Summarize a video by extracting keyframes and generating captions.
    
    Args:
        video_path (str): Path to the video file
        interval (float): Seconds between frames to process
        output_file (str): Path to save the summary (if None, prints to console)
        use_gpu (bool): Whether to use GPU if available
    
    Returns:
        list: List of captions with timestamps
    """
    # Set up device
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    print(f"Using device: {device}")
    
    # Load the BLIP model
    print("Loading BLIP model...")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return []
    
    # Open the video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Frames per second: {fps:.2f}")
    
    # Calculate frame interval
    frame_interval = int(fps * interval)
    estimated_frames = total_frames // frame_interval + 1
    print(f"Processing approximately {estimated_frames} frames (every {interval} seconds)")
    
    # Process video
    frame_count = 0
    keyframe_count = 0
    captions = []
    
    start_time = time.time()
    
    # Set up progress bar
    progress_bar = tqdm(total=total_frames, desc="Processing frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress bar
        progress_bar.update(1)
        
        # Process keyframes at specified interval
        if frame_count % frame_interval == 0:
            try:
                # Convert frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Generate caption
                inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
                outputs = model.generate(**inputs, max_length=50)
                caption = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Get timestamp
                timestamp = frame_count / fps
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d}"
                
                # Add to captions list
                caption_text = f"[{timestamp_str}] {caption}"
                captions.append(caption_text)
                keyframe_count += 1
                
            except Exception as e:
                print(f"Error processing frame at {frame_count}: {e}")
        
        frame_count += 1
    
    # Close progress bar
    progress_bar.close()
    
    # Release video capture
    cap.release()
    
    # Calculate processing stats
    elapsed = time.time() - start_time
    fps_processing = keyframe_count / elapsed if elapsed > 0 else 0
    
    print(f"\nProcessing complete!")
    print(f"Processed {keyframe_count} keyframes in {elapsed:.2f} seconds")
    print(f"Average processing speed: {fps_processing:.2f} frames/sec")
    
    # Write output if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Video Summary: {os.path.basename(video_path)}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Processed {keyframe_count} keyframes\n\n")
            for caption in captions:
                f.write(f"{caption}\n")
        print(f"Summary saved to {output_file}")
    else:
        print("\nSummary:")
        for caption in captions:
            print(caption)
    
    return captions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize a video using BLIP AI captioning")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between frames to process (default: 2.0)")
    parser.add_argument("--output", type=str, help="Path to save the summary (default: print to console)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    summarize_video(
        args.video_path,
        interval=args.interval,
        output_file=args.output,
        use_gpu=not args.cpu
    )
