import cv2
import numpy as np
import json
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import threading
import queue
import time

class PickleballTracker2D:
    def __init__(self, video_path, calibration_file):
        # Load calibration data
        with open(calibration_file, 'r') as f:
            self.calibration = json.load(f)
        
        self.homography = np.array(self.calibration['homography'])
        self.court_width = self.calibration['court_width']
        self.court_length = self.calibration['court_length']
        
        # Video setup
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # YOLO setup
        from ultralytics import YOLO
        self.model = YOLO('yolov8n.pt')  # Using nano for speed
        
        # Tracking data
        self.tracks = defaultdict(lambda: deque(maxlen=100))  # Store last 100 positions
        self.current_positions = {}
        self.frame_count = 0
        
        # Setup matplotlib for real-time display
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.setup_court_view()
        
        # Threading for video processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.running = True
        
    def setup_court_view(self):
        """Setup the 2D court visualization"""
        self.ax2.clear()
        self.ax2.set_xlim(0, self.court_width)
        self.ax2.set_ylim(0, self.court_length)
        self.ax2.set_aspect('equal')
        self.ax2.set_title('2D Court View - San4', fontsize=14, fontweight='bold')
        
        # Draw court boundaries
        court_rect = patches.Rectangle((0, 0), self.court_width, self.court_length, 
                                     linewidth=3, edgecolor='black', facecolor='lightgreen', alpha=0.3)
        self.ax2.add_patch(court_rect)
        
        # Draw net (vertical line in middle)
        net_y = self.court_length / 2
        self.ax2.plot([0, self.court_width], [net_y, net_y], 'k-', linewidth=4, label='Net')
        
        # Draw service lines
        service_bottom = self.court_length / 4
        service_top = 3 * self.court_length / 4
        self.ax2.plot([0, self.court_width], [service_bottom, service_bottom], 'b--', linewidth=2, alpha=0.7)
        self.ax2.plot([0, self.court_width], [service_top, service_top], 'b--', linewidth=2, alpha=0.7)
        
        # Draw center line
        center_x = self.court_width / 2
        self.ax2.plot([center_x, center_x], [0, self.court_length], 'b--', linewidth=2, alpha=0.7)
        
        # Side labels
        self.ax2.text(self.court_width/4, -0.3, 'LEFT SIDE', ha='center', fontsize=12, fontweight='bold')
        self.ax2.text(3*self.court_width/4, -0.3, 'RIGHT SIDE', ha='center', fontsize=12, fontweight='bold')
        
        # Grid
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlabel('Court Width (m)')
        self.ax2.set_ylabel('Court Length (m)')
        
    def transform_to_court(self, image_points):
        """Transform image coordinates to court coordinates"""
        if len(image_points) == 0:
            return []
        
        # Ensure points are in the right format
        points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform using homography
        court_points = cv2.perspectiveTransform(points, self.homography)
        
        return court_points.reshape(-1, 2)
    
    def process_frame(self, frame):
        """Process a single frame for tracking"""
        # Run YOLO detection and tracking
        results = self.model.track(frame, persist=True, verbose=False)
        
        detections = []
        
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
                    class_name = self.model.names[int(cls)]
                    
                    if class_name in ['person', 'sports ball'] and conf > 0.5:
                        x, y, w, h = box
                        
                        detections.append({
                            'track_id': int(track_id),
                            'class': class_name,
                            'image_pos': [x, y],
                            'confidence': conf
                        })
        
        return detections
    
    def update_tracking(self, detections):
        """Update tracking data with new detections"""
        # Transform image positions to court coordinates
        if detections:
            image_points = [d['image_pos'] for d in detections]
            court_points = self.transform_to_court(image_points)
            
            for detection, court_pos in zip(detections, court_points):
                track_id = detection['track_id']
                class_name = detection['class']
                
                # Filter points within court bounds (with some tolerance)
                margin = 1.0  # 1 meter margin
                if (-margin <= court_pos[0] <= self.court_width + margin and 
                    -margin <= court_pos[1] <= self.court_length + margin):
                    
                    self.tracks[track_id].append({
                        'frame': self.frame_count,
                        'court_pos': court_pos,
                        'class': class_name,
                        'confidence': detection['confidence']
                    })
                    
                    self.current_positions[track_id] = {
                        'court_pos': court_pos,
                        'class': class_name
                    }
    
    def update_visualization(self, frame):
        """Update both original video and 2D court views"""
        # Update original video view
        self.ax1.clear()
        self.ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.ax1.set_title(f'Original Video - Frame {self.frame_count}')
        self.ax1.axis('off')
        
        # Update 2D court view
        self.setup_court_view()
        
        # Draw current positions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (track_id, data) in enumerate(self.current_positions.items()):
            court_pos = data['court_pos']
            class_name = data['class']
            color = colors[i % len(colors)]
            
            if class_name == 'person':
                # Draw player
                self.ax2.scatter(court_pos[0], court_pos[1], c=color, s=100, 
                               marker='o', edgecolors='black', linewidth=2, 
                               label=f'Player {track_id}')
                
                # Draw trail
                if track_id in self.tracks:
                    trail = [t['court_pos'] for t in list(self.tracks[track_id])[-20:]]  # Last 20 positions
                    if len(trail) > 1:
                        trail_x = [p[0] for p in trail]
                        trail_y = [p[1] for p in trail]
                        self.ax2.plot(trail_x, trail_y, color=color, alpha=0.6, linewidth=2)
                
            elif class_name == 'sports ball':
                # Draw ball
                self.ax2.scatter(court_pos[0], court_pos[1], c='yellow', s=80, 
                               marker='*', edgecolors='red', linewidth=2, 
                               label='Ball')
        
        # Add legend
        if self.current_positions:
            self.ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add frame info
        self.ax2.text(0.02, 0.98, f'Frame: {self.frame_count}\nPlayers: {len([p for p in self.current_positions.values() if p["class"] == "person"])}', 
                     transform=self.ax2.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def run(self):
        """Main execution loop"""
        print("Starting 2D Pickleball Tracker for San4...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video or cannot read frame")
                    break
                
                self.frame_count += 1
                
                # Process every 2nd frame for speed
                if self.frame_count % 2 == 0:
                    detections = self.process_frame(frame)
                    self.update_tracking(detections)
                
                # Update visualization every 3rd frame
                if self.frame_count % 3 == 0:
                    self.update_visualization(frame)
                
                # Skip frames for real-time-ish playback
                if self.frame_count % 30 == 0:
                    print(f"Processed {self.frame_count} frames, Active players: {len(self.current_positions)}")
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cap.release()
            plt.close()

if __name__ == "__main__":
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    calibration_file = "court_calibration_san4.json"
    
    tracker = PickleballTracker2D(video_path, calibration_file)
    tracker.run()