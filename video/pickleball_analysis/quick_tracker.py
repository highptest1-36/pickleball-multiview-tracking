import cv2
from ultralytics import YOLO
import json
import numpy as np

# Load model
model = YOLO('yolov8x.pt')

# Open video
video_path = r'C:\Users\highp\pickerball\video\data_video\san1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {total_frames} frames at {fps} FPS")
print("Processing (will take a while)...")

# Track through video
tracking_data = []
frame_count = 0
save_interval = 1000  # Save every 1000 frames

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Track objects
        results = model.track(frame, persist=True, verbose=False)
        
        # Extract tracking data
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
                    x, y, w, h = box
                    
                    # Get class name
                    class_name = model.names[int(cls)]
                    
                    tracking_data.append({
                        'frame': frame_count,
                        'track_id': int(track_id),
                        'class': class_name,
                        'x': float(x),
                        'y': float(y),
                        'width': float(w),
                        'height': float(h),
                        'confidence': float(conf)
                    })
        
        # Progress and intermediate save
        if frame_count % save_interval == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
            # Save intermediate data
            with open('tracking_data_san1_temp.json', 'w') as f:
                json.dump(tracking_data, f)
        
        # Quick test - stop after 5000 frames for now
        if frame_count >= 5000:
            print(f"Stopping at frame {frame_count} for testing...")
            break

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    
    # Save final data
    print(f"Saving {len(tracking_data)} tracking records...")
    with open('tracking_data_san1.json', 'w') as f:
        json.dump(tracking_data, f)
    
    print("Done!")