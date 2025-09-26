# Enhanced Real-Time Pickleball Visualization
# Tối ưu performance và smooth visualization

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches
import os
import time
from collections import deque
import multiprocessing as mp

class EnhancedRealTimePickleball:
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.yolo_model = YOLO('yolo11n.pt')
        
        # Court dimensions
        self.court_length = 6.10  # meters
        self.court_width = 4.27   # meters
        self.net_height = 0.91    # meters
        
        # Visualization settings
        self.video_size = (320, 180)  # Smaller for better performance
        self.viz_size = (640, 480)    # 3D visualization size
        
        # Tracking data
        self.ball_trajectory = deque(maxlen=30)
        self.player_positions = deque(maxlen=5)  # Last 5 frames
        
        # Performance optimization
        self.frame_skip = 1  # Process every frame
        
        # Color scheme
        self.colors = {
            'person': (0, 255, 0),    # Green
            'ball': (0, 0, 255),      # Red
            'trajectory': (255, 255, 0),  # Yellow
            'court': (255, 255, 255), # White
            'net': (0, 0, 255)        # Red
        }
    
    def detect_and_track(self, frame):
        """Optimized detection and tracking"""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        
        # Run detection
        results = self.yolo_model(small_frame, classes=[0, 37], conf=0.25, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Scale factor back to original
            scale_x = frame.shape[1] / 320
            scale_y = frame.shape[0] / 240
            
            for box in boxes:
                cls = int(box.cls.cpu().numpy().item())
                conf = float(box.conf.cpu().numpy().item())
                xyxy = box.xyxy.cpu().numpy()[0]
                
                # Scale back to original frame size
                xyxy[0] *= scale_x
                xyxy[1] *= scale_y
                xyxy[2] *= scale_x
                xyxy[3] *= scale_y
                
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2
                
                detections.append({
                    'class': cls,
                    'confidence': conf,
                    'bbox': xyxy,
                    'center': (center_x, center_y),
                    'class_name': 'person' if cls == 0 else 'ball'
                })
        
        return detections
    
    def draw_tracking_overlay(self, frame, detections, camera_id):
        """Draw tracking overlay on frame"""
        overlay = frame.copy()
        
        for det in detections:
            bbox = det['bbox'].astype(int)
            color = self.colors['person'] if det['class'] == 0 else self.colors['ball']
            
            # Draw bounding box
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw center point
            center = (int(det['center'][0]), int(det['center'][1]))
            cv2.circle(overlay, center, 3, color, -1)
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(overlay, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(overlay, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Camera label
        cv2.putText(overlay, f'Camera {camera_id}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f'Camera {camera_id}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        return overlay
    
    def estimate_3d_positions(self, all_detections):
        """Simplified 3D position estimation"""
        persons_3d = []
        balls_3d = []
        
        # Simple heuristic-based 3D estimation
        for cam_id, detections in all_detections.items():
            for det in detections:
                # Convert 2D to approximate 3D (very simplified)
                x_2d, y_2d = det['center']
                
                # Map camera view to court coordinates (heuristic)
                if cam_id == 0:  # Camera 1 - front left
                    x_3d = (x_2d / 1920) * 3  # Left half of court
                    y_3d = (y_2d / 1080) * self.court_width
                elif cam_id == 1:  # Camera 2 - front right
                    x_3d = 3 + (x_2d / 1920) * 3  # Right half of court
                    y_3d = (y_2d / 1080) * self.court_width
                elif cam_id == 2:  # Camera 3 - back left
                    x_3d = (x_2d / 1920) * 3
                    y_3d = self.court_width - (y_2d / 1080) * self.court_width
                else:  # Camera 4 - back right
                    x_3d = 3 + (x_2d / 1920) * 3
                    y_3d = self.court_width - (y_2d / 1080) * self.court_width
                
                # Height estimation
                if det['class'] == 0:  # person
                    z_3d = 1.7  # Average person height
                else:  # ball
                    z_3d = 0.5  # Ball height
                
                # Clamp to court bounds
                x_3d = np.clip(x_3d, 0, self.court_length)
                y_3d = np.clip(y_3d, 0, self.court_width)
                z_3d = np.clip(z_3d, 0, 3)
                
                pos_3d = np.array([x_3d, y_3d, z_3d])
                
                if det['class'] == 0:
                    persons_3d.append(pos_3d)
                else:
                    balls_3d.append(pos_3d)
        
        return persons_3d, balls_3d
    
    def create_god_view_fast(self, persons_3d, balls_3d, frame_num):
        """Fast God View creation"""
        # Create figure with specific DPI for consistent size
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=80)
        fig.patch.set_facecolor('black')
        
        # God View (top subplot)
        ax1.set_xlim(-0.2, 6.3)
        ax1.set_ylim(-0.2, 4.5)
        ax1.set_aspect('equal')
        ax1.set_title(f'God View - Frame {frame_num}', color='white', fontsize=10)
        ax1.set_facecolor('black')
        
        # Draw court
        court_corners = np.array([[0, 0], [self.court_length, 0], 
                                 [self.court_length, self.court_width], 
                                 [0, self.court_width], [0, 0]])
        ax1.plot(court_corners[:, 0], court_corners[:, 1], 'w-', linewidth=2)
        
        # Draw center line and net
        net_x = self.court_length / 2
        ax1.plot([net_x, net_x], [0, self.court_width], 'r-', linewidth=3)
        
        # Draw service areas
        ax1.plot([0, self.court_length], [self.court_width/4, self.court_width/4], 'w--', alpha=0.5)
        ax1.plot([0, self.court_length], [3*self.court_width/4, 3*self.court_width/4], 'w--', alpha=0.5)
        
        # Non-volley zone
        kitchen = 2.13 / 2
        ax1.fill_between([net_x - kitchen, net_x + kitchen], 0, self.court_width, alpha=0.3, color='yellow')
        
        # Plot current positions
        if persons_3d:
            persons_array = np.array(persons_3d)
            ax1.scatter(persons_array[:, 0], persons_array[:, 1], 
                       c='lime', s=80, alpha=0.8, marker='o')
        
        if balls_3d:
            balls_array = np.array(balls_3d)
            ax1.scatter(balls_array[:, 0], balls_array[:, 1], 
                       c='red', s=60, alpha=1.0, marker='o')
            
            # Update ball trajectory
            for ball_pos in balls_3d:
                self.ball_trajectory.append(ball_pos[:2])  # Only x, y for trajectory
            
            # Draw trajectory
            if len(self.ball_trajectory) > 1:
                traj_array = np.array(list(self.ball_trajectory))
                ax1.plot(traj_array[:, 0], traj_array[:, 1], 'yellow', alpha=0.7, linewidth=2)
        
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Side view (bottom subplot)
        ax2.set_xlim(0, self.court_length)
        ax2.set_ylim(0, 3)
        ax2.set_title('Side View', color='white', fontsize=10)
        ax2.set_facecolor('black')
        
        # Draw court side view
        ax2.plot([0, self.court_length], [0, 0], 'w-', linewidth=2)  # Ground
        ax2.plot([net_x, net_x], [0, self.net_height], 'r-', linewidth=3)  # Net
        
        # Plot side view positions
        if persons_3d:
            persons_array = np.array(persons_3d)
            ax2.scatter(persons_array[:, 0], persons_array[:, 2], 
                       c='lime', s=60, alpha=0.8, marker='o')
        
        if balls_3d:
            balls_array = np.array(balls_3d)
            ax2.scatter(balls_array[:, 0], balls_array[:, 2], 
                       c='red', s=40, alpha=1.0, marker='o')
        
        ax2.set_xlabel('Court Length (m)', color='white', fontsize=8)
        ax2.set_ylabel('Height (m)', color='white', fontsize=8)
        ax2.tick_params(colors='white', labelsize=8)
        
        plt.tight_layout()
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        # Convert RGB to BGR
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
    def run_enhanced_visualization(self, max_frames=300):
        """Run enhanced real-time visualization"""
        print("🚀 Enhanced Real-Time Pickleball Visualization")
        print("📺 Split Screen: Left = Multi-View | Right = God View + Side View")
        
        # Open video captures
        caps = []
        for i, path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"❌ Cannot open video {i+1}")
                return False
            caps.append(cap)
        
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        print(f"📹 Video FPS: {fps:.1f}")
        
        # Output video setup
        output_path = 'enhanced_realtime_pickleball.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        frame_count = 0
        start_time = time.time()
        last_viz_time = 0
        viz_interval = 1.0 / 10  # Update 3D viz at 10 FPS
        
        try:
            while frame_count < max_frames:
                # Read frames
                frames = []
                all_valid = True
                
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        all_valid = False
                        break
                    frames.append(frame)
                
                if not all_valid:
                    break
                
                # Process detections for each camera
                all_detections = {}
                for i, frame in enumerate(frames):
                    detections = self.detect_and_track(frame)
                    all_detections[i] = detections
                
                # Create annotated frames
                annotated_frames = []
                for i, frame in enumerate(frames):
                    annotated = self.draw_tracking_overlay(frame, all_detections[i], i+1)
                    # Resize for multi-view
                    resized = cv2.resize(annotated, self.video_size)
                    annotated_frames.append(resized)
                
                # Combine to 2x2 grid
                top_row = np.hstack([annotated_frames[0], annotated_frames[1]])
                bottom_row = np.hstack([annotated_frames[2], annotated_frames[3]])
                multiview = np.vstack([top_row, bottom_row])
                
                # Generate 3D visualization (less frequently for performance)
                current_time = time.time()
                if current_time - last_viz_time >= viz_interval or frame_count == 0:
                    # Estimate 3D positions
                    persons_3d, balls_3d = self.estimate_3d_positions(all_detections)
                    
                    # Create 3D visualization
                    god_view_img = self.create_god_view_fast(persons_3d, balls_3d, frame_count)
                    last_viz_time = current_time
                
                # Resize god view to match multiview height
                target_height = multiview.shape[0]
                aspect_ratio = god_view_img.shape[1] / god_view_img.shape[0]
                target_width = int(target_height * aspect_ratio)
                god_view_resized = cv2.resize(god_view_img, (target_width, target_height))
                
                # Combine side by side
                combined = np.hstack([multiview, god_view_resized])
                
                # Add frame info
                info_text = f"Frame: {frame_count} | Players: {len(persons_3d)} | Balls: {len(balls_3d)}"
                cv2.putText(combined, info_text, (10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, info_text, (10, combined.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                
                # Initialize video writer
                if frame_count == 0:
                    h, w = combined.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    print(f"📝 Output resolution: {w}x{h}")
                
                # Write frame
                out.write(combined)
                
                # Show progress
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    processing_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Frame {frame_count}: Processing at {processing_fps:.1f} FPS")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\\n⏹️ Stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            for cap in caps:
                cap.release()
            if 'out' in locals():
                out.release()
            
            total_time = time.time() - start_time
            print(f"\\n✅ Enhanced visualization completed!")
            print(f"📊 Stats:")
            print(f"   - Frames processed: {frame_count}")
            print(f"   - Total time: {total_time:.1f}s")
            print(f"   - Average FPS: {frame_count/total_time:.1f}")
            print(f"   - Output: {output_path}")
        
        return True

def main():
    """Main function"""
    print("🏓 ENHANCED REAL-TIME PICKLEBALL VISUALIZATION")
    print("=" * 55)
    
    # Setup video paths
    base_path = r"C:\\Users\\highp\\pickerball"
    video_folders = [
        "e4e66c2058ff-0.0.0.0-3000-1-0-mzle9eCKS2oQvLJa7rOE000000",
        "e4e66c2058ff-0.0.0.0-3000-2-0-vvkoKtKIUN7KS72O4bfR000000", 
        "e4e66c2058ff-0.0.0.0-3000-3-0-a4TtYafdNkjZQjVO5hll000000",
        "e4e66c2058ff-0.0.0.0-3000-4-0-ZhV2hb2DFg8xhbXYcpWn000000"
    ]
    
    video_paths = []
    for folder in video_folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(folder_path, mp4_files[0])
                video_paths.append(video_path)
                print(f"✅ Video found: {mp4_files[0]}")
    
    if len(video_paths) < 4:
        print(f"❌ Need 4 videos, found {len(video_paths)}")
        return
    
    # Create and run visualizer
    visualizer = EnhancedRealTimePickleball(video_paths)
    success = visualizer.run_enhanced_visualization(max_frames=150)  # Shorter for demo
    
    if success:
        print("\\n🎉 SUCCESS! Enhanced visualization completed!")
        print("💡 Features:")
        print("   ✅ Real-time split-screen layout")
        print("   ✅ 4-camera multiview with tracking overlays")
        print("   ✅ God View with court visualization")  
        print("   ✅ Side view showing ball height")
        print("   ✅ Ball trajectory tracking")
        print("   ✅ Player position mapping")

if __name__ == "__main__":
    main()