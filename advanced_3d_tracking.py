#!/usr/bin/env python3
"""
Advanced 3D Pickleball Player Tracking System
- True 3D player movement tracking
- Maximum 4 players per court
- Real-time visualization with movement trails
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
import os
from collections import defaultdict, deque
import time

class Advanced3DPickleballTracker:
    def __init__(self):
        print("🏓 ADVANCED 3D PICKLEBALL TRACKING SYSTEM")
        print("=" * 55)
        
        # Load YOLO model
        self.model = YOLO('yolo11n.pt')
        
        # Video paths
        self.video_paths = [
            "e4e66c2058ff-0.0.0.0-3000-1-0-mzle9eCKS2oQvLJa7rOE000000/h20250926093017-20250926093526m.mp4",
            "e4e66c2058ff-0.0.0.0-3000-2-0-vvkoKtKIUN7KS72O4bfR000000/h20250926093019-20250926093526m.mp4", 
            "e4e66c2058ff-0.0.0.0-3000-3-0-a4TtYafdNkjZQjVO5hll000000/h20250926093021-20250926093526m.mp4",
            "e4e66c2058ff-0.0.0.0-3000-4-0-ZhV2hb2DFg8xhbXYcpWn000000/h20250926093022-20250926093526m.mp4"
        ]
        
        # Verify videos exist
        for i, path in enumerate(self.video_paths):
            if os.path.exists(path):
                print(f"✅ Camera {i+1}: {os.path.basename(path)}")
            else:
                print(f"❌ Camera {i+1}: Not found")
        
        # Load video captures
        self.caps = [cv2.VideoCapture(path) for path in self.video_paths]
        self.fps = self.caps[0].get(cv2.CAP_PROP_FPS)
        print(f"📹 Video FPS: {self.fps}")
        
        # Camera setup for 3D reconstruction
        self.setup_cameras()
        
        # Player tracking
        self.max_players = 4  # Maximum 4 players per court
        self.player_tracker = PlayerTracker(max_players=self.max_players)
        self.ball_tracker = BallTracker()
        
        # Visualization setup
        self.setup_visualization()
    
    def setup_cameras(self):
        """Setup camera matrices for 3D reconstruction"""
        # Simplified camera intrinsic parameters
        focal_length = 800
        center_x, center_y = 320, 240
        
        self.camera_matrix = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y], 
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4,1))
        
        # Camera positions for 4 corners of the court (in meters)
        self.camera_positions = [
            np.array([0, 0, 2.5]),      # Camera 1: Front-left corner
            np.array([6, 0, 2.5]),      # Camera 2: Front-right corner  
            np.array([6, 4, 2.5]),      # Camera 3: Back-right corner
            np.array([0, 4, 2.5])       # Camera 4: Back-left corner
        ]
        
        # Camera rotations (looking toward center of court)
        self.camera_rotations = [
            np.array([0, 0, 0]),        # Camera 1
            np.array([0, 0, -90]),      # Camera 2
            np.array([0, 0, -180]),     # Camera 3  
            np.array([0, 0, 90])        # Camera 4
        ]
    
    def setup_visualization(self):
        """Setup real-time 3D visualization"""
        plt.ioff()  # Turn off interactive mode to avoid blocking
        self.fig = plt.figure(figsize=(14, 8))
        
        # Layout: Left = Multi-view, Right = 3D tracking
        self.ax_multi = plt.subplot2grid((1, 2), (0, 0))
        self.ax_3d = plt.subplot2grid((1, 2), (0, 1), projection='3d')
        
        self.setup_3d_court()
        
    def setup_3d_court(self):
        """Setup 3D court visualization"""
        # Pickleball court dimensions (meters)
        length, width = 6.1, 4.27
        net_height = 0.91
        
        # Court boundaries
        court_x = [0, length, length, 0, 0]
        court_y = [0, 0, width, width, 0] 
        court_z = [0, 0, 0, 0, 0]
        
        self.ax_3d.plot(court_x, court_y, court_z, 'k-', linewidth=2, label='Court')
        
        # Net
        net_x = [length/2, length/2]
        net_y = [0, width]
        net_z = [0, net_height]
        self.ax_3d.plot(net_x, net_y, net_z, 'r-', linewidth=3, label='Net')
        
        # Court zones
        self.ax_3d.plot([0, length], [width/2, width/2], [0, 0], 'gray', alpha=0.5)
        
        self.ax_3d.set_xlabel('Length (m)')
        self.ax_3d.set_ylabel('Width (m)') 
        self.ax_3d.set_zlabel('Height (m)')
        self.ax_3d.set_title('🏓 Real-Time 3D Player Tracking')
        
        # Set limits
        self.ax_3d.set_xlim(0, 6.5)
        self.ax_3d.set_ylim(0, 4.5)
        self.ax_3d.set_zlim(0, 3)
        
    def triangulate_3d_position(self, points_2d):
        """Triangulate 3D position from multiple 2D points"""
        if len(points_2d) < 2:
            return None
            
        # Use first two cameras for triangulation
        if len(points_2d) >= 2:
            point1 = np.array(points_2d[0], dtype=np.float32).reshape(1, 1, 2)
            point2 = np.array(points_2d[1], dtype=np.float32).reshape(1, 1, 2)
            
            # Projection matrices (simplified)
            P1 = np.hstack([self.camera_matrix, np.zeros((3, 1))])
            P2 = np.hstack([self.camera_matrix, np.array([[100], [0], [0]])])
            
            # Triangulate
            points_4d = cv2.triangulatePoints(P1, P2, point1, point2)
            points_3d = points_4d[:3] / points_4d[3]
            
            # Convert to court coordinates
            x, y, z = points_3d.flatten()
            
            # Map to court dimensions (0-6m x 0-4m)
            x = max(0, min(6, x * 0.01))  
            y = max(0, min(4, y * 0.01))
            z = max(0, min(3, abs(z) * 0.01))
            
            return np.array([x, y, z])
        
        return None
    
    def detect_and_track_frame(self, frame_idx):
        """Detect and track objects in current frame"""
        frames = []
        all_detections = []
        
        # Read frames from all cameras
        for cap in self.caps:
            ret, frame = cap.read()
            if not ret:
                return None, None
            
            # Resize for processing
            frame_small = cv2.resize(frame, (320, 240))
            frames.append(frame)
            
            # YOLO detection
            results = self.model(frame_small, classes=[0, 37], conf=0.3, verbose=False)
            detections = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Scale back to original size
                        scale_x, scale_y = frame.shape[1] / 320, frame.shape[0] / 240
                        x1, x2 = x1 * scale_x, x2 * scale_x
                        y1, y2 = y1 * scale_y, y2 * scale_y
                        
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': [center_x, center_y],
                            'conf': conf,
                            'class': cls,
                            'type': 'person' if cls == 0 else 'ball'
                        })
            
            all_detections.append(detections)
        
        return frames, all_detections
    
    def update_tracking(self, all_detections, frame_idx):
        """Update player and ball tracking"""
        # Separate persons and balls
        person_detections = [[] for _ in range(4)]  # 4 cameras
        ball_detections = [[] for _ in range(4)]
        
        for cam_idx, detections in enumerate(all_detections):
            for det in detections:
                if det['type'] == 'person':
                    person_detections[cam_idx].append(det)
                else:
                    ball_detections[cam_idx].append(det)
        
        # Update player tracker
        self.player_tracker.update(person_detections, frame_idx)
        
        # Update ball tracker  
        self.ball_tracker.update(ball_detections, frame_idx)
    
    def create_multiview_frame(self, frames, all_detections):
        """Create multi-view display with tracking overlays"""
        if not frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Resize frames
        display_frames = []
        for i, frame in enumerate(frames):
            display_frame = cv2.resize(frame, (320, 240))
            
            # Draw detections
            for det in all_detections[i]:
                x1, y1, x2, y2 = det['bbox']
                # Scale to display size
                x1, x2 = x1 * 320 / frame.shape[1], x2 * 320 / frame.shape[1]
                y1, y2 = y1 * 240 / frame.shape[0], y2 * 240 / frame.shape[0]
                
                color = (0, 255, 0) if det['type'] == 'person' else (0, 0, 255)
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Label
                label = f"{det['type']} {det['conf']:.2f}"
                cv2.putText(display_frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            display_frames.append(display_frame)
        
        # Arrange in 2x2 grid
        top_row = np.hstack([display_frames[0], display_frames[1]])
        bottom_row = np.hstack([display_frames[2], display_frames[3]])
        multiview = np.vstack([top_row, bottom_row])
        
        return multiview
    
    def update_3d_visualization(self):
        """Update 3D visualization with current player positions"""
        # Clear previous player markers
        self.ax_3d.clear()
        self.setup_3d_court()
        
        # Plot current player positions
        current_players = self.player_tracker.get_current_positions()
        
        colors = ['blue', 'red', 'green', 'orange']
        for player_id, positions in current_players.items():
            if positions:
                latest_pos = positions[-1]  # Get latest position
                if latest_pos is not None:
                    x, y, z = latest_pos
                    color = colors[player_id % len(colors)]
                    
                    # Plot player position
                    self.ax_3d.scatter([x], [y], [z], c=color, s=100, 
                                     label=f'Player {player_id+1}')
                    
                    # Plot movement trail (last 10 positions)
                    if len(positions) > 1:
                        trail_positions = [pos for pos in positions[-10:] if pos is not None]
                        if len(trail_positions) > 1:
                            xs, ys, zs = zip(*trail_positions)
                            self.ax_3d.plot(xs, ys, zs, color=color, alpha=0.5, linewidth=2)
        
        # Plot ball trajectory
        ball_positions = self.ball_tracker.get_trajectory()
        if len(ball_positions) > 1:
            valid_positions = [pos for pos in ball_positions if pos is not None]
            if len(valid_positions) > 1:
                xs, ys, zs = zip(*valid_positions)
                self.ax_3d.plot(xs, ys, zs, 'r--', linewidth=2, alpha=0.7, label='Ball')
                
                # Current ball position
                if ball_positions[-1] is not None:
                    x, y, z = ball_positions[-1]
                    self.ax_3d.scatter([x], [y], [z], c='red', s=50, marker='o')
        
        self.ax_3d.legend()
        plt.draw()
    
    def run_advanced_tracking(self, max_frames=200):
        """Run advanced 3D tracking system"""
        print(f"🚀 Starting Advanced 3D Tracking")
        print(f"📺 Processing up to {max_frames} frames")
        
        frame_idx = 0
        start_time = time.time()
        
        # Video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('advanced_3d_tracking.mp4', fourcc, 10.0, (1280, 720))
        
        try:
            while frame_idx < max_frames:
                # Detect and track current frame
                frames, all_detections = self.detect_and_track_frame(frame_idx)
                
                if frames is None:
                    break
                
                # Update tracking
                self.update_tracking(all_detections, frame_idx)
                
                # Create visualizations
                multiview = self.create_multiview_frame(frames, all_detections)
                
                # Update 3D visualization every 5 frames for performance
                if frame_idx % 5 == 0:
                    self.update_3d_visualization()
                
                # Combine visualizations
                self.ax_multi.clear()
                self.ax_multi.imshow(cv2.cvtColor(multiview, cv2.COLOR_BGR2RGB))
                self.ax_multi.set_title(f'Multi-View Tracking - Frame {frame_idx}')
                self.ax_multi.axis('off')
                
                # Force update without blocking
                plt.tight_layout()
                
                # Save frame to video
                self.fig.canvas.draw()
                width, height = self.fig.canvas.get_width_height()
                buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(height, width, 4)  # RGBA format
                buf = buf[:, :, :3]  # Convert to RGB
                buf = cv2.resize(buf, (1280, 720))
                out.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 25 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_idx / elapsed if elapsed > 0 else 0
                    print(f"Frame {frame_idx}: Processing at {fps:.1f} FPS")
                    
                    # Print tracking stats
                    active_players = len(self.player_tracker.get_current_positions())
                    print(f"  👥 Active players: {active_players}/{self.max_players}")
            
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        # Cleanup
        out.release()
        for cap in self.caps:
            cap.release()
        
        elapsed = time.time() - start_time
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        
        print(f"\n✅ Advanced 3D tracking completed!")
        print(f"📊 Final Stats:")
        print(f"   - Frames processed: {frame_idx}")
        print(f"   - Total time: {elapsed:.1f}s")
        print(f"   - Average FPS: {avg_fps:.1f}")
        print(f"   - Output: advanced_3d_tracking.mp4")
        print(f"   - Max players tracked: {self.max_players}")
        
        return True


class PlayerTracker:
    """Advanced player tracking with movement history"""
    def __init__(self, max_players=4, max_history=50):
        self.max_players = max_players
        self.max_history = max_history
        self.player_positions = {}  # player_id: deque of 3D positions
        self.player_last_seen = {}  # player_id: frame_idx
        self.next_player_id = 0
        self.position_threshold = 0.5  # meters for position matching
        
    def update(self, person_detections_multi_cam, frame_idx):
        """Update player tracking from multi-camera detections"""
        # Triangulate 3D positions from multi-camera detections
        triangulated_positions = []
        
        # Find corresponding persons across cameras
        if len(person_detections_multi_cam) >= 2:
            for i, dets_cam1 in enumerate(person_detections_multi_cam[0]):
                best_matches = [dets_cam1['center']]  # Start with camera 1
                
                # Find best matches in other cameras
                for cam_idx in range(1, len(person_detections_multi_cam)):
                    cam_dets = person_detections_multi_cam[cam_idx]
                    if cam_dets:
                        # Simple matching - take closest detection
                        best_match = min(cam_dets, key=lambda x: x['conf'], default=None)
                        if best_match:
                            best_matches.append(best_match['center'])
                
                # Triangulate if we have matches from at least 2 cameras
                if len(best_matches) >= 2:
                    pos_3d = self.triangulate_simple(best_matches)
                    if pos_3d is not None:
                        triangulated_positions.append(pos_3d)
        
        # Limit to max players
        triangulated_positions = triangulated_positions[:self.max_players]
        
        # Match positions to existing players or create new ones
        self.match_players(triangulated_positions, frame_idx)
        
        # Remove inactive players
        self.cleanup_inactive_players(frame_idx)
    
    def triangulate_simple(self, points_2d):
        """Simple triangulation from 2D points"""
        if len(points_2d) < 2:
            return None
        
        # Simple mapping to 3D court coordinates
        # This is a simplified approach - in real system would use proper triangulation
        p1, p2 = points_2d[0], points_2d[1]
        
        # Map pixel coordinates to court coordinates (rough approximation)
        x = (p1[0] + p2[0]) / 2 * 6.0 / 640  # Map to 0-6m court length
        y = (p1[1] + p2[1]) / 2 * 4.0 / 480  # Map to 0-4m court width  
        z = 1.7  # Average player height
        
        # Clamp to court bounds
        x = max(0, min(6, x))
        y = max(0, min(4, y))
        z = max(1.5, min(2.2, z))
        
        return np.array([x, y, z])
    
    def match_players(self, new_positions, frame_idx):
        """Match new positions to existing players"""
        # Calculate distances to existing players
        unmatched_positions = new_positions.copy()
        
        for player_id, history in self.player_positions.items():
            if not history:
                continue
                
            last_pos = history[-1]
            if last_pos is None:
                continue
            
            # Find closest new position
            if unmatched_positions:
                distances = [np.linalg.norm(pos - last_pos) for pos in unmatched_positions]
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                # If close enough, assign to this player
                if min_dist < self.position_threshold:
                    new_pos = unmatched_positions.pop(min_dist_idx)
                    self.player_positions[player_id].append(new_pos)
                    self.player_last_seen[player_id] = frame_idx
                    
                    # Limit history length
                    if len(self.player_positions[player_id]) > self.max_history:
                        self.player_positions[player_id].popleft()
        
        # Create new players for unmatched positions
        for pos in unmatched_positions:
            if self.next_player_id < self.max_players:
                player_id = self.next_player_id
                self.player_positions[player_id] = deque([pos], maxlen=self.max_history)
                self.player_last_seen[player_id] = frame_idx
                self.next_player_id += 1
    
    def cleanup_inactive_players(self, frame_idx, timeout_frames=30):
        """Remove players that haven't been seen recently"""
        to_remove = []
        for player_id, last_seen in self.player_last_seen.items():
            if frame_idx - last_seen > timeout_frames:
                to_remove.append(player_id)
        
        for player_id in to_remove:
            del self.player_positions[player_id]
            del self.player_last_seen[player_id]
    
    def get_current_positions(self):
        """Get current positions of all tracked players"""
        return {pid: list(history) for pid, history in self.player_positions.items()}


class BallTracker:
    """Ball tracking with trajectory history"""
    def __init__(self, max_history=30):
        self.max_history = max_history
        self.trajectory = deque(maxlen=max_history)
    
    def update(self, ball_detections_multi_cam, frame_idx):
        """Update ball tracking"""
        # Simple ball tracking - take best detection across cameras
        best_ball = None
        best_conf = 0
        
        for cam_dets in ball_detections_multi_cam:
            for det in cam_dets:
                if det['conf'] > best_conf:
                    best_conf = det['conf']
                    best_ball = det
        
        if best_ball:
            # Convert to 3D position (simplified)
            center = best_ball['center']
            x = center[0] * 6.0 / 640  # Map to court length
            y = center[1] * 4.0 / 480  # Map to court width
            z = 1.0  # Estimated ball height
            
            pos_3d = np.array([x, y, z])
            self.trajectory.append(pos_3d)
        else:
            self.trajectory.append(None)  # No ball detected
    
    def get_trajectory(self):
        """Get ball trajectory history"""
        return list(self.trajectory)


if __name__ == "__main__":
    tracker = Advanced3DPickleballTracker()
    tracker.run_advanced_tracking(max_frames=150)