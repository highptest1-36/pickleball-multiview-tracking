# Real-Time Pickleball 3D Visualization System
# Hiển thị đồng bộ: Bên trái 4 video gốc + Bên phải 3D visualization

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
import time
import threading
from collections import deque
import queue

class RealTimePickleballVisualization:
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.yolo_model = YOLO('yolo11n.pt')
        
        # Camera parameters (simplified)
        self.setup_cameras()
        
        # Court dimensions
        self.court_length = 6.10  # meters
        self.court_width = 4.27   # meters
        self.net_height = 0.91    # meters
        
        # Tracking history
        self.tracking_history = deque(maxlen=30)  # Last 30 frames
        self.ball_trajectory = deque(maxlen=50)   # Ball trajectory
        
        # Visualization settings
        self.frame_size = (480, 270)  # Reduced size for each video
        self.god_view_size = (600, 400)
        
        # Real-time data queue
        self.data_queue = queue.Queue(maxsize=10)
        
    def setup_cameras(self):
        """Setup camera parameters"""
        # Simplified camera setup for demo
        self.cameras = {}
        focal_length = 600
        
        for i in range(4):
            K = np.array([
                [focal_length, 0, 240],
                [0, focal_length, 135],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Camera positions (estimated)
            positions = [
                {'rvec': np.array([0.3, 0.0, 0.0]), 'tvec': np.array([-2.0, -1.0, 2.0])},
                {'rvec': np.array([0.3, 0.0, 0.5]), 'tvec': np.array([8.0, -1.0, 2.0])},
                {'rvec': np.array([-0.3, 0.0, 3.14]), 'tvec': np.array([-2.0, 5.5, 2.0])},
                {'rvec': np.array([-0.3, 0.0, 2.64]), 'tvec': np.array([8.0, 5.5, 2.0])}
            ]
            
            self.cameras[i] = {
                'K': K,
                'rvec': positions[i]['rvec'],
                'tvec': positions[i]['tvec']
            }
    
    def detect_objects_in_frame(self, frame):
        """Detect objects in a single frame"""
        results = self.yolo_model(frame, classes=[0, 37], conf=0.3, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls = int(box.cls.cpu().numpy().item())
                conf = float(box.conf.cpu().numpy().item())
                xyxy = box.xyxy.cpu().numpy()[0]
                
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
    
    def triangulate_3d_point(self, points_2d, camera_ids):
        """Simple triangulation for 3D position"""
        if len(points_2d) < 2:
            return None
        
        # Simplified triangulation - using average of projections
        # In real implementation, use proper DLT triangulation
        
        # Convert 2D points to normalized coordinates and estimate 3D
        # This is a simplified version for demo purposes
        
        avg_x = np.mean([p[0] for p in points_2d]) / 100  # Scale factor
        avg_y = np.mean([p[1] for p in points_2d]) / 100
        
        # Estimate 3D position (simplified)
        x_3d = (avg_x - 240) * 0.01  # Rough conversion
        y_3d = (avg_y - 135) * 0.01
        z_3d = 1.7 if points_2d[0] else 0.0  # Assume person height or ball on ground
        
        # Clamp to court bounds
        x_3d = np.clip(x_3d, 0, self.court_length)
        y_3d = np.clip(y_3d, 0, self.court_width)
        z_3d = np.clip(z_3d, 0, 3)
        
        return np.array([x_3d, y_3d, z_3d])
    
    def process_frame_data(self, frames):
        """Process frames and extract 3D positions"""
        all_detections = {}
        
        # Detect objects in each frame
        for i, frame in enumerate(frames):
            detections = self.detect_objects_in_frame(frame)
            all_detections[i] = detections
        
        # Match detections across views (simplified)
        persons_3d = []
        balls_3d = []
        
        # Simple matching: if detected in multiple views
        for cam_id, detections in all_detections.items():
            for det in detections:
                if det['class'] == 0:  # person
                    # Try to find corresponding detections in other cameras
                    matched_points = [det['center']]
                    matched_cameras = [cam_id]
                    
                    for other_cam, other_dets in all_detections.items():
                        if other_cam != cam_id:
                            for other_det in other_dets:
                                if other_det['class'] == 0:
                                    matched_points.append(other_det['center'])
                                    matched_cameras.append(other_cam)
                                    break
                    
                    if len(matched_points) >= 2:
                        pos_3d = self.triangulate_3d_point(matched_points, matched_cameras)
                        if pos_3d is not None:
                            persons_3d.append(pos_3d)
                
                elif det['class'] == 37:  # ball
                    matched_points = [det['center']]
                    matched_cameras = [cam_id]
                    
                    for other_cam, other_dets in all_detections.items():
                        if other_cam != cam_id:
                            for other_det in other_dets:
                                if other_det['class'] == 37:
                                    matched_points.append(other_det['center'])
                                    matched_cameras.append(other_cam)
                                    break
                    
                    if len(matched_points) >= 2:
                        pos_3d = self.triangulate_3d_point(matched_points, matched_cameras)
                        if pos_3d is not None:
                            balls_3d.append(pos_3d)
        
        return {
            'detections': all_detections,
            'persons_3d': persons_3d,
            'balls_3d': balls_3d
        }
    
    def create_multiview_frame(self, frames, detections):
        """Create 2x2 multiview frame with tracking overlay"""
        # Resize frames
        resized_frames = []
        for i, frame in enumerate(frames):
            resized = cv2.resize(frame, self.frame_size)
            
            # Draw detections
            if i in detections:
                for det in detections[i]:
                    bbox = det['bbox']
                    # Scale bbox to resized frame
                    scale_x = self.frame_size[0] / frame.shape[1]
                    scale_y = self.frame_size[1] / frame.shape[0]
                    
                    x1 = int(bbox[0] * scale_x)
                    y1 = int(bbox[1] * scale_y)
                    x2 = int(bbox[2] * scale_x)
                    y2 = int(bbox[3] * scale_y)
                    
                    # Draw bbox
                    color = (0, 255, 0) if det['class'] == 0 else (0, 0, 255)
                    cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    cv2.putText(resized, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add camera label
            cv2.putText(resized, f'Camera {i+1}', (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            resized_frames.append(resized)
        
        # Combine into 2x2 grid
        top_row = np.hstack([resized_frames[0], resized_frames[1]])
        bottom_row = np.hstack([resized_frames[2], resized_frames[3]])
        multiview = np.vstack([top_row, bottom_row])
        
        return multiview
    
    def create_3d_visualization(self, persons_3d, balls_3d, frame_num):
        """Create 3D visualization"""
        fig = plt.figure(figsize=(10, 8))
        
        # Create subplots: God View (top) and 3D View (bottom)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, projection='3d')
        
        # God View (Top-down)
        ax1.set_xlim(-0.5, 6.5)
        ax1.set_ylim(-0.5, 4.5)
        ax1.set_aspect('equal')
        ax1.set_title(f'God View - Frame {frame_num}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Length (m)')
        ax1.set_ylabel('Width (m)')
        
        # Draw court
        court_corners = np.array([[0, 0], [self.court_length, 0], 
                                 [self.court_length, self.court_width], 
                                 [0, self.court_width], [0, 0]])
        ax1.plot(court_corners[:, 0], court_corners[:, 1], 'k-', linewidth=2)
        
        # Draw net
        net_x = self.court_length / 2
        ax1.plot([net_x, net_x], [0, self.court_width], 'r-', linewidth=3, label='Net')
        
        # Draw non-volley zone
        kitchen_depth = 2.13 / 2  # 7 feet / 2
        ax1.fill_between([net_x - kitchen_depth, net_x + kitchen_depth], 
                        0, self.court_width, alpha=0.2, color='yellow')
        
        # Plot persons
        if persons_3d:
            persons_array = np.array(persons_3d)
            ax1.scatter(persons_array[:, 0], persons_array[:, 1], 
                       c='blue', s=100, alpha=0.7, label=f'Players ({len(persons_3d)})')
        
        # Plot balls
        if balls_3d:
            balls_array = np.array(balls_3d)
            ax1.scatter(balls_array[:, 0], balls_array[:, 1], 
                       c='red', s=50, alpha=0.8, label=f'Ball')
            
            # Add ball to trajectory
            for ball_pos in balls_3d:
                self.ball_trajectory.append(ball_pos)
            
            # Draw ball trajectory
            if len(self.ball_trajectory) > 1:
                traj_array = np.array(list(self.ball_trajectory))
                ax1.plot(traj_array[:, 0], traj_array[:, 1], 'r--', alpha=0.6, linewidth=1)
        
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 3D View
        ax2.set_xlim(0, self.court_length)
        ax2.set_ylim(0, self.court_width)
        ax2.set_zlim(0, 3)
        ax2.set_title('3D Court View', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Length (m)')
        ax2.set_ylabel('Width (m)')
        ax2.set_zlabel('Height (m)')
        
        # Draw court 3D
        court_3d = np.array([[0, 0, 0], [self.court_length, 0, 0], 
                            [self.court_length, self.court_width, 0], 
                            [0, self.court_width, 0]])
        for i in range(4):
            j = (i + 1) % 4
            ax2.plot([court_3d[i, 0], court_3d[j, 0]], 
                    [court_3d[i, 1], court_3d[j, 1]], 
                    [court_3d[i, 2], court_3d[j, 2]], 'k-', linewidth=2)
        
        # Draw net 3D
        ax2.plot([net_x, net_x], [0, 0], [0, self.net_height], 'r-', linewidth=3)
        ax2.plot([net_x, net_x], [0, self.court_width], [self.net_height, self.net_height], 'r-', linewidth=3)
        ax2.plot([net_x, net_x], [self.court_width, self.court_width], [self.net_height, 0], 'r-', linewidth=3)
        
        # Plot 3D positions
        if persons_3d:
            persons_array = np.array(persons_3d)
            ax2.scatter(persons_array[:, 0], persons_array[:, 1], persons_array[:, 2], 
                       c='blue', s=50, alpha=0.7)
        
        if balls_3d:
            balls_array = np.array(balls_3d)
            ax2.scatter(balls_array[:, 0], balls_array[:, 1], balls_array[:, 2], 
                       c='red', s=30, alpha=0.8)
            
            # Draw 3D trajectory
            if len(self.ball_trajectory) > 1:
                traj_array = np.array(list(self.ball_trajectory))
                ax2.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 
                        'r--', alpha=0.6, linewidth=1)
        
        plt.tight_layout()
        
        # Save to memory buffer
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        
        return buf
    
    def run_realtime_visualization(self, max_frames=500):
        """Run real-time visualization"""
        print("🚀 Starting Real-Time Pickleball Visualization...")
        print("📺 Layout: Left = 4 Cameras | Right = 3D Visualization")
        
        # Open video captures
        caps = []
        for i, path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"❌ Cannot open video {i+1}")
                return False
            caps.append(cap)
        
        # Get video properties
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        print(f"📹 Video FPS: {fps}")
        
        # Create output video writer
        output_path = 'realtime_pickleball_visualization.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while frame_count < max_frames:
                # Read frames from all cameras
                frames = []
                all_valid = True
                
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        all_valid = False
                        break
                    frames.append(frame)
                
                if not all_valid or len(frames) < 4:
                    break
                
                # Process frame data
                frame_data = self.process_frame_data(frames)
                
                # Create multiview frame (left side)
                multiview_frame = self.create_multiview_frame(frames, frame_data['detections'])
                
                # Create 3D visualization (right side)
                viz_3d = self.create_3d_visualization(
                    frame_data['persons_3d'], 
                    frame_data['balls_3d'], 
                    frame_count
                )
                
                # Resize 3D visualization to match multiview height
                target_height = multiview_frame.shape[0]
                target_width = int(viz_3d.shape[1] * target_height / viz_3d.shape[0])
                viz_3d_resized = cv2.resize(viz_3d, (target_width, target_height))
                
                # Combine side by side
                combined_frame = np.hstack([multiview_frame, viz_3d_resized])
                
                # Initialize video writer on first frame
                if frame_count == 0:
                    h, w = combined_frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    print(f"📝 Output size: {w}x{h}")
                
                # Write frame
                out.write(combined_frame)
                
                # Show progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    processing_fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    persons_count = len(frame_data['persons_3d'])
                    balls_count = len(frame_data['balls_3d'])
                    
                    print(f"Frame {frame_count}: {persons_count} persons, {balls_count} balls, "
                          f"{processing_fps:.1f} FPS")
                
                frame_count += 1
                
                # Optional: Show real-time preview (uncomment if needed)
                # cv2.imshow('Real-Time Pickleball Visualization', combined_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        
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
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"\n✅ Completed!")
            print(f"📊 Processed {frame_count} frames in {total_time:.1f}s")
            print(f"🎯 Average FPS: {frame_count/total_time:.1f}")
            print(f"📁 Output: {output_path}")
        
        return True

def main():
    """Main function"""
    print("🏓 REAL-TIME PICKLEBALL 3D VISUALIZATION")
    print("=" * 50)
    
    # Video paths
    base_path = r"C:\Users\highp\pickerball"
    video_folders = [
        "e4e66c2058ff-0.0.0.0-3000-1-0-mzle9eCKS2oQvLJa7rOE000000",
        "e4e66c2058ff-0.0.0.0-3000-2-0-vvkoKtKIUN7KS72O4bfR000000", 
        "e4e66c2058ff-0.0.0.0-3000-3-0-a4TtYafdNkjZQjVO5hll000000",
        "e4e66c2058ff-0.0.0.0-3000-4-0-ZhV2hb2DFg8xhbXYcpWn000000"
    ]
    
    # Find video files
    video_paths = []
    for folder in video_folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(folder_path, mp4_files[0])
                video_paths.append(video_path)
                print(f"✅ Found: {mp4_files[0]}")
    
    if len(video_paths) < 4:
        print(f"❌ Need 4 videos, found {len(video_paths)}")
        return
    
    # Create visualizer
    visualizer = RealTimePickleballVisualization(video_paths)
    
    # Run real-time visualization
    success = visualizer.run_realtime_visualization(max_frames=200)
    
    if success:
        print("\n🎉 Real-time visualization completed!")
        print("💡 Output video shows:")
        print("   - Left: 4-camera multiview with tracking")
        print("   - Right: 3D God View + 3D court visualization")

if __name__ == "__main__":
    main()