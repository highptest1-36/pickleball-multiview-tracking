"""
Full Video Tracking Script
Generate tracking data cho toàn bộ video với better detection
"""

import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

class FullVideoTracker:
    """Full video tracking với improved detection."""
    
    def __init__(self):
        """Initialize tracker."""
        self.model = None
        self.tracking_data = []
        
    def load_model(self):
        """Load YOLO model."""
        try:
            model_path = "yolov8x.pt"
            if not os.path.exists(model_path):
                print("⬇️ Downloading YOLOv8x model...")
                
            self.model = YOLO(model_path)
            print("✅ YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def track_video(self, video_path: str, max_frames: int = 1000, output_dir: str = "full_tracking_output"):
        """
        Track toàn bộ video.
        
        Args:
            video_path: Đường dẫn video
            max_frames: Số frames tối đa
            output_dir: Thư mục output
        """
        if not self.load_model():
            return False
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/tracking_data", exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Video Info:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f}s")
        print(f"  Will process: {min(max_frames, total_frames)} frames")
        
        frame_count = 0
        processed_frames = 0
        
        # Track with YOLO
        print(f"\n🔍 Starting tracking...")
        
        # Use YOLO tracking
        results = self.model.track(
            source=video_path,
            save=False,
            tracker="bytetrack.yaml",  # Use ByteTrack
            classes=[0, 32],  # person=0, sports ball=32
            conf=0.3,  # Lower confidence for better detection
            iou=0.5,
            max_det=20,  # Allow more detections
            vid_stride=1,  # Process every frame
        )
        
        for result in results:
            if frame_count >= max_frames:
                break
                
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get tracking info
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        track_id = int(boxes.id[i])
                    else:
                        track_id = i  # Fallback if no tracking ID
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Get class
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    
                    # Map class names
                    if class_id == 0:
                        class_name = "player"
                    elif class_id == 32:
                        class_name = "ball"
                    else:
                        continue
                    
                    # Add to tracking data
                    self.tracking_data.append({
                        'frame_id': frame_count,
                        'object_id': track_id,
                        'class': class_name,
                        'confidence': confidence,
                        'bbox_x1': x1,
                        'bbox_y1': y1,
                        'bbox_x2': x2,
                        'bbox_y2': y2,
                        'center_x': center_x,
                        'center_y': center_y,
                        'timestamp': frame_count / fps
                    })
                
                processed_frames += 1
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames, {len(self.tracking_data)} detections")
        
        cap.release()
        
        # Save tracking data
        if self.tracking_data:
            df = pd.DataFrame(self.tracking_data)
            output_path = f"{output_dir}/tracking_data/full_tracking.csv"
            df.to_csv(output_path, index=False)
            
            print(f"\n📊 TRACKING COMPLETED:")
            print(f"  Total frames processed: {frame_count}")
            print(f"  Frames with detections: {processed_frames}")
            print(f"  Total detections: {len(self.tracking_data)}")
            print(f"  Unique players: {len(df[df['class']=='player']['object_id'].unique())}")
            print(f"  Unique balls: {len(df[df['class']=='ball']['object_id'].unique())}")
            print(f"  Output: {output_path}")
            
            return True
        else:
            print("❌ No tracking data generated")
            return False

def main():
    """Main function."""
    video_path = r"C:\Users\highp\pickerball\video\data_video\san1.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print("🏓 FULL VIDEO TRACKING")
    print("=" * 50)
    
    tracker = FullVideoTracker()
    success = tracker.track_video(video_path, max_frames=1000, output_dir="full_tracking_output")
    
    if success:
        print("✅ Full video tracking completed!")
    else:
        print("❌ Tracking failed")

if __name__ == "__main__":
    main()