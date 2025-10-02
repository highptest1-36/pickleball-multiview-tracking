import cv2
import numpy as np
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt

def test_ball_detection():
    """Test ball detection on San4 video"""
    print("Testing ball detection...")
    
    # Load model
    model = YOLO('yolov8x.pt')
    
    # Load video
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Load calibration
    with open('court_calibration_san4.json', 'r') as f:
        calibration = json.load(f)
    homography = np.array(calibration['homography'])
    
    frame_count = 0
    ball_detections = []
    person_detections = []
    
    # Test first 500 frames
    while frame_count < 500:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run detection (lower confidence for balls)
        results = model(frame, verbose=False, conf=0.1)
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    class_name = model.names[int(cls)]
                    
                    if class_name in ['sports ball', 'ball'] and conf > 0.15:
                        x, y, w, h = box
                        ball_detections.append({
                            'frame': frame_count,
                            'conf': conf,
                            'pos': [x, y],
                            'size': max(w, h)
                        })
                        
                    elif class_name == 'person' and conf > 0.3:
                        person_detections.append({
                            'frame': frame_count,
                            'conf': conf
                        })
        
        if frame_count % 100 == 0:
            print(f"Frame {frame_count}: {len(ball_detections)} balls, {len(person_detections)} persons")
    
    cap.release()
    
    print(f"\nResults after {frame_count} frames:")
    print(f"Total ball detections: {len(ball_detections)}")
    print(f"Total person detections: {len(person_detections)}")
    
    if ball_detections:
        # Analyze ball detections
        ball_confs = [b['conf'] for b in ball_detections]
        ball_sizes = [b['size'] for b in ball_detections]
        
        print(f"Ball confidence range: {min(ball_confs):.3f} - {max(ball_confs):.3f}")
        print(f"Ball size range: {min(ball_sizes):.1f} - {max(ball_sizes):.1f}")
        
        # Show ball detection distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist([b['frame'] for b in ball_detections], bins=20)
        plt.title('Ball Detections by Frame')
        plt.xlabel('Frame')
        plt.ylabel('Count')
        
        plt.subplot(1, 3, 2)
        plt.hist(ball_confs, bins=20)
        plt.title('Ball Detection Confidence')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        
        plt.subplot(1, 3, 3)
        plt.hist(ball_sizes, bins=20)
        plt.title('Ball Size Distribution')
        plt.xlabel('Size (pixels)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('ball_detection_analysis.png')
        plt.show()
    
    else:
        print("❌ No balls detected! Try:")
        print("1. Lower confidence threshold")
        print("2. Different YOLO model")
        print("3. Check if 'sports ball' class exists")
        
        # Print available classes
        print(f"\nAvailable classes: {list(model.names.values())}")

if __name__ == "__main__":
    test_ball_detection()