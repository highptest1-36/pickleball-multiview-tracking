"""
2D Court Viewer - Real-time visualization của sân pickleball với heatmap và player tracking

Features:
1. Bird's-eye view của sân pickleball
2. Real-time player tracking với màu sắc khác nhau
3. Ball tracking với trail
4. Heat map overlay
5. Live statistics
6. Court zones visualization
"""

import cv2
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import threading
import queue

class PickleballCourt2D:
    """2D Court visualization class."""
    
    def __init__(self, config_path: str = "config/court_points.json"):
        """
        Initialize 2D court viewer.
        
        Args:
            config_path: Đường dẫn config file
        """
        # Court dimensions (bird's-eye view)
        self.court_width = 1341   # 13.41m * 100 pixels/m
        self.court_height = 610   # 6.1m * 100 pixels/m
        self.padding = 100
        
        # Canvas dimensions
        self.canvas_width = self.court_width + 2 * self.padding
        self.canvas_height = self.court_height + 2 * self.padding
        
        # Court coordinates
        self.court_x = self.padding
        self.court_y = self.padding
        
        # Colors
        self.court_color = (34, 139, 34)  # Forest green
        self.line_color = (255, 255, 255)  # White
        self.net_color = (255, 255, 255)   # White
        self.kitchen_color = (0, 255, 255)  # Yellow
        
        # Player colors (up to 4 players)
        self.player_colors = [
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue  
            (0, 255, 0),    # Green
            (255, 0, 255),  # Magenta
        ]
        
        self.ball_color = (255, 255, 255)  # White
        
        # Tracking data
        self.player_positions = defaultdict(deque)  # player_id -> deque of positions
        self.ball_positions = deque(maxlen=50)      # Ball trail
        self.heatmap_data = defaultdict(list)       # player_id -> [positions]
        
        # Display settings
        self.trail_length = 30
        self.point_size = 8
        self.trail_thickness = 2
        
        # Statistics
        self.player_stats = defaultdict(dict)
        
        # Heatmap
        self.heatmap_resolution = (134, 61)  # 10 pixels per meter
        self.heatmap_alpha = 0.6
        
        print("🏓 2D Court Viewer initialized")
        print(f"📐 Court size: {self.court_width}x{self.court_height}")

    def create_court_canvas(self) -> np.ndarray:
        """
        Tạo canvas với sân pickleball.
        
        Returns:
            Canvas image
        """
        # Create green background
        canvas = np.full((self.canvas_height, self.canvas_width, 3), 
                        (50, 50, 50), dtype=np.uint8)
        
        # Draw court background
        cv2.rectangle(canvas, 
                     (self.court_x, self.court_y),
                     (self.court_x + self.court_width, self.court_y + self.court_height),
                     self.court_color, -1)
        
        # Court boundary
        cv2.rectangle(canvas,
                     (self.court_x, self.court_y),
                     (self.court_x + self.court_width, self.court_y + self.court_height),
                     self.line_color, 3)
        
        # Net (center line)
        net_y = self.court_y + self.court_height // 2
        cv2.line(canvas,
                (self.court_x, net_y),
                (self.court_x + self.court_width, net_y),
                self.net_color, 4)
        
        # Service lines (2.13m from net)
        service_distance = int(2.13 * 100)  # 2.13m * 100 pixels/m
        
        # Top service line
        cv2.line(canvas,
                (self.court_x, self.court_y + service_distance),
                (self.court_x + self.court_width, self.court_y + service_distance),
                self.line_color, 2)
        
        # Bottom service line  
        cv2.line(canvas,
                (self.court_x, self.court_y + self.court_height - service_distance),
                (self.court_x + self.court_width, self.court_y + self.court_height - service_distance),
                self.line_color, 2)
        
        # Center service line
        center_x = self.court_x + self.court_width // 2
        cv2.line(canvas,
                (center_x, self.court_y),
                (center_x, self.court_y + service_distance),
                self.line_color, 2)
        cv2.line(canvas,
                (center_x, self.court_y + self.court_height - service_distance),
                (center_x, self.court_y + self.court_height),
                self.line_color, 2)
        
        # Non-volley zone (kitchen)
        kitchen_y1 = net_y - service_distance // 2
        kitchen_y2 = net_y + service_distance // 2
        cv2.rectangle(canvas,
                     (self.court_x, kitchen_y1),
                     (self.court_x + self.court_width, kitchen_y2),
                     self.kitchen_color, 2)
        
        # Add zone labels
        cv2.putText(canvas, "NON-VOLLEY ZONE", 
                   (self.court_x + 400, net_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.kitchen_color, 2)
        
        return canvas

    def transform_point(self, x: float, y: float, homography: np.ndarray) -> Tuple[int, int]:
        """
        Transform point từ camera view sang court view.
        
        Args:
            x, y: Tọa độ trong camera view
            homography: Homography matrix
            
        Returns:
            (x, y) trong court coordinate system
        """
        # Apply homography transformation
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, homography)
        
        # Convert to canvas coordinates
        court_x = int(transformed[0][0][0]) + self.court_x
        court_y = int(transformed[0][0][1]) + self.court_y
        
        return court_x, court_y

    def add_player_position(self, player_id: int, x: float, y: float, 
                           timestamp: float, homography: np.ndarray):
        """
        Thêm vị trí player mới.
        
        Args:
            player_id: ID của player
            x, y: Tọa độ trong camera view
            timestamp: Timestamp
            homography: Homography matrix
        """
        # Transform to court coordinates
        court_x, court_y = self.transform_point(x, y, homography)
        
        # Add to trail
        self.player_positions[player_id].append((court_x, court_y, timestamp))
        if len(self.player_positions[player_id]) > self.trail_length:
            self.player_positions[player_id].popleft()
        
        # Add to heatmap data
        # Convert to heatmap coordinates (relative to court)
        heatmap_x = int((court_x - self.court_x) / self.court_width * self.heatmap_resolution[0])
        heatmap_y = int((court_y - self.court_y) / self.court_height * self.heatmap_resolution[1])
        
        # Clamp to valid bounds
        heatmap_x = max(0, min(heatmap_x, self.heatmap_resolution[0] - 1))
        heatmap_y = max(0, min(heatmap_y, self.heatmap_resolution[1] - 1))
        
        if 0 <= heatmap_x < self.heatmap_resolution[0] and 0 <= heatmap_y < self.heatmap_resolution[1]:
            self.heatmap_data[player_id].append((heatmap_x, heatmap_y))

    def add_ball_position(self, x: float, y: float, timestamp: float, homography: np.ndarray):
        """
        Thêm vị trí ball mới.
        
        Args:
            x, y: Tọa độ trong camera view
            timestamp: Timestamp  
            homography: Homography matrix
        """
        # Transform to court coordinates
        court_x, court_y = self.transform_point(x, y, homography)
        
        # Add to trail
        self.ball_positions.append((court_x, court_y, timestamp))

    def create_heatmap_overlay(self, player_id: int) -> np.ndarray:
        """
        Tạo heatmap overlay cho player.
        
        Args:
            player_id: ID của player
            
        Returns:
            Heatmap overlay image
        """
        if player_id not in self.heatmap_data or not self.heatmap_data[player_id]:
            return np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        
        # Create heatmap grid
        heatmap = np.zeros(self.heatmap_resolution, dtype=np.float32)
        
        # Fill heatmap with positions
        for hx, hy in self.heatmap_data[player_id]:
            # Ensure bounds
            hx = max(0, min(hx, self.heatmap_resolution[0] - 1))
            hy = max(0, min(hy, self.heatmap_resolution[1] - 1))
            heatmap[hy, hx] += 1
        
        # Apply gaussian filter for smoothing
        heatmap = gaussian_filter(heatmap, sigma=2)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Create colormap
        colormap = plt.cm.hot
        heatmap_colored = colormap(heatmap)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Resize to court size
        heatmap_resized = cv2.resize(heatmap_colored, (self.court_width, self.court_height))
        
        # Create full canvas overlay
        overlay = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        overlay[self.court_y:self.court_y + self.court_height, 
                self.court_x:self.court_x + self.court_width] = heatmap_resized
        
        return overlay

    def draw_players(self, canvas: np.ndarray) -> np.ndarray:
        """
        Vẽ players và trails lên canvas.
        
        Args:
            canvas: Canvas để vẽ
            
        Returns:
            Canvas với players
        """
        result = canvas.copy()
        
        for player_id, positions in self.player_positions.items():
            if not positions:
                continue
            
            # Get player color
            color = self.player_colors[player_id % len(self.player_colors)]
            
            # Draw trail
            if len(positions) > 1:
                trail_points = [(pos[0], pos[1]) for pos in positions]
                for i in range(1, len(trail_points)):
                    # Fade trail
                    alpha = i / len(trail_points)
                    thickness = max(1, int(self.trail_thickness * alpha))
                    
                    cv2.line(result, trail_points[i-1], trail_points[i], color, thickness)
            
            # Draw current position
            if positions:
                current_pos = positions[-1]
                cv2.circle(result, (current_pos[0], current_pos[1]), 
                          self.point_size, color, -1)
                
                # Draw player ID
                cv2.putText(result, f"P{player_id}", 
                           (current_pos[0] + 10, current_pos[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result

    def draw_ball(self, canvas: np.ndarray) -> np.ndarray:
        """
        Vẽ ball và trail lên canvas.
        
        Args:
            canvas: Canvas để vẽ
            
        Returns:
            Canvas với ball
        """
        result = canvas.copy()
        
        if not self.ball_positions:
            return result
        
        # Draw ball trail
        if len(self.ball_positions) > 1:
            trail_points = [(pos[0], pos[1]) for pos in self.ball_positions]
            for i in range(1, len(trail_points)):
                alpha = i / len(trail_points)
                thickness = max(1, int(2 * alpha))
                cv2.line(result, trail_points[i-1], trail_points[i], self.ball_color, thickness)
        
        # Draw current ball position
        current_pos = self.ball_positions[-1]
        cv2.circle(result, (current_pos[0], current_pos[1]), 5, self.ball_color, -1)
        cv2.circle(result, (current_pos[0], current_pos[1]), 8, self.ball_color, 2)
        
        # Ball label
        cv2.putText(result, "BALL", 
                   (current_pos[0] + 12, current_pos[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ball_color, 1)
        
        return result

    def draw_statistics(self, canvas: np.ndarray, frame_info: Dict[str, Any]) -> np.ndarray:
        """
        Vẽ statistics lên canvas.
        
        Args:
            canvas: Canvas để vẽ
            frame_info: Thông tin frame hiện tại
            
        Returns:
            Canvas với statistics
        """
        result = canvas.copy()
        
        # Statistics panel
        panel_x = 10
        panel_y = 10
        line_height = 25
        
        # Background panel
        cv2.rectangle(result, (panel_x - 5, panel_y - 5), 
                     (panel_x + 300, panel_y + 200), (0, 0, 0), -1)
        cv2.rectangle(result, (panel_x - 5, panel_y - 5),
                     (panel_x + 300, panel_y + 200), (255, 255, 255), 2)
        
        # Title
        cv2.putText(result, "PICKLEBALL COURT ANALYSIS", 
                   (panel_x, panel_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame info
        y_pos = panel_y + 50
        cv2.putText(result, f"Frame: {frame_info.get('frame', 0)}", 
                   (panel_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += line_height
        cv2.putText(result, f"Time: {frame_info.get('time', 0):.1f}s", 
                   (panel_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Player count
        y_pos += line_height
        player_count = len([pid for pid in self.player_positions.keys() if self.player_positions[pid]])
        cv2.putText(result, f"Players: {player_count}/4", 
                   (panel_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Ball status
        y_pos += line_height
        ball_status = "YES" if self.ball_positions else "NO"
        cv2.putText(result, f"Ball: {ball_status}", 
                   (panel_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Player legends
        y_pos += line_height + 10
        cv2.putText(result, "PLAYERS:", 
                   (panel_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (player_id, positions) in enumerate(self.player_positions.items()):
            if not positions:
                continue
            
            y_pos += line_height
            color = self.player_colors[player_id % len(self.player_colors)]
            
            # Color dot
            cv2.circle(result, (panel_x + 10, y_pos - 5), 6, color, -1)
            
            # Player info
            pos_count = len(self.heatmap_data.get(player_id, []))
            cv2.putText(result, f"Player {player_id}: {pos_count} pts", 
                       (panel_x + 25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result

    def render_frame(self, frame_info: Dict[str, Any], show_heatmap: bool = True, 
                    heatmap_player: Optional[int] = None) -> np.ndarray:
        """
        Render một frame của court view.
        
        Args:
            frame_info: Thông tin frame
            show_heatmap: Có hiển thị heatmap không
            heatmap_player: Player ID để hiển thị heatmap (None = all)
            
        Returns:
            Rendered frame
        """
        # Create court canvas
        canvas = self.create_court_canvas()
        
        # Add heatmap overlay
        if show_heatmap:
            if heatmap_player is not None:
                # Single player heatmap
                overlay = self.create_heatmap_overlay(heatmap_player)
                canvas = cv2.addWeighted(canvas, 1 - self.heatmap_alpha, overlay, self.heatmap_alpha, 0)
            else:
                # All players heatmap
                for player_id in self.player_positions.keys():
                    overlay = self.create_heatmap_overlay(player_id)
                    canvas = cv2.addWeighted(canvas, 1 - self.heatmap_alpha/4, overlay, self.heatmap_alpha/4, 0)
        
        # Draw players
        canvas = self.draw_players(canvas)
        
        # Draw ball
        canvas = self.draw_ball(canvas)
        
        # Draw statistics
        canvas = self.draw_statistics(canvas, frame_info)
        
        return canvas

class LiveCourtViewer:
    """Live court viewer với video input."""
    
    def __init__(self, video_path: str, camera_name: str = "san1"):
        """
        Initialize live viewer.
        
        Args:
            video_path: Đường dẫn video
            camera_name: Tên camera (san1, san2, etc.)
        """
        self.video_path = video_path
        self.camera_name = camera_name
        
        # Load court calibration
        self.homography = self.load_homography()
        if self.homography is None:
            raise ValueError(f"No calibration found for camera {camera_name}")
        
        # Initialize court
        self.court = PickleballCourt2D()
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Loaded video: {video_path}")
        print(f"📹 FPS: {self.fps}, Frames: {self.total_frames}")

    def load_homography(self) -> Optional[np.ndarray]:
        """
        Load homography matrix từ calibration file.
        
        Returns:
            Homography matrix hoặc None
        """
        try:
            with open("config/court_points.json", 'r') as f:
                config = json.load(f)
            
            if self.camera_name not in config['cameras']:
                return None
            
            camera_config = config['cameras'][self.camera_name]
            if camera_config['calibration_status'] != 'calibrated':
                return None
            
            return np.array(camera_config['homography_matrix'], dtype=np.float32)
            
        except Exception as e:
            print(f"❌ Error loading homography: {e}")
            return None

    def run_live_demo(self, tracking_data_path: str, max_frames: int = 300):
        """
        Chạy live demo với tracking data.
        
        Args:
            tracking_data_path: Đường dẫn CSV tracking data
            max_frames: Số frames tối đa
        """
        # Load tracking data
        try:
            tracking_df = pd.read_csv(tracking_data_path)
            print(f"📊 Loaded {len(tracking_df)} tracking records")
        except Exception as e:
            print(f"❌ Error loading tracking data: {e}")
            return
        
        print("🔴 Starting live court viewer...")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  'h': Toggle heatmap")
        print("  '1-4': Show specific player heatmap")
        print("  '0': Show all players heatmap")
        print("  'q': Quit")
        
        frame_count = 0
        paused = False
        show_heatmap = True
        heatmap_player = None
        
        # Create windows
        cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
        cv2.namedWindow("2D Court View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original Video", 800, 600)
        cv2.resizeWindow("2D Court View", 800, 600)
        
        while frame_count < max_frames:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Get tracking data for current frame
                frame_detections = tracking_df[tracking_df['frame_id'] == frame_count]
                
                # Update court with new positions
                for _, detection in frame_detections.iterrows():
                    obj_id = int(detection['object_id'])
                    center_x = detection['center_x']
                    center_y = detection['center_y'] 
                    timestamp = detection['timestamp']
                    class_name = detection['class']
                    
                    if class_name == 'person' or 'player' in class_name:
                        self.court.add_player_position(obj_id, center_x, center_y, timestamp, self.homography)
                    elif class_name == 'ball':
                        self.court.add_ball_position(center_x, center_y, timestamp, self.homography)
                
                frame_count += 1
            
            # Render court view
            frame_info = {
                'frame': frame_count,
                'time': frame_count / self.fps,
                'detections': len(frame_detections) if not paused else 0
            }
            
            court_view = self.court.render_frame(frame_info, show_heatmap, heatmap_player)
            
            # Show original video with detections
            if 'frame' in locals():
                display_frame = frame.copy()
                
                # Draw detections on original
                if not paused:
                    for _, detection in frame_detections.iterrows():
                        x1, y1 = int(detection['bbox_x1']), int(detection['bbox_y1'])
                        x2, y2 = int(detection['bbox_x2']), int(detection['bbox_y2'])
                        center_x, center_y = int(detection['center_x']), int(detection['center_y'])
                        obj_id = int(detection['object_id'])
                        class_name = detection['class']
                        
                        # Color based on class
                        if class_name == 'person' or 'player' in class_name:
                            color = self.court.player_colors[obj_id % len(self.court.player_colors)]
                        else:
                            color = self.court.ball_color
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(display_frame, (center_x, center_y), 5, color, -1)
                        cv2.putText(display_frame, f"{class_name}_{obj_id}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                cv2.imshow("Original Video", display_frame)
            
            cv2.imshow("2D Court View", court_view)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                status = "PAUSED" if paused else "PLAYING"
                print(f"📺 {status} at frame {frame_count}")
            elif key == ord('h'):
                show_heatmap = not show_heatmap
                print(f"🔥 Heatmap: {'ON' if show_heatmap else 'OFF'}")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                heatmap_player = int(chr(key)) - 1
                print(f"🔥 Showing heatmap for Player {heatmap_player}")
            elif key == ord('0'):
                heatmap_player = None
                print(f"🔥 Showing heatmap for all players")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Demo completed ({frame_count} frames)")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="2D Court Viewer")
    parser.add_argument('--video', type=str, required=True,
                       help='Đường dẫn video')
    parser.add_argument('--camera', type=str, default='san1',
                       help='Tên camera (san1, san2, etc.)')
    parser.add_argument('--tracking', type=str, required=True,
                       help='Đường dẫn tracking data CSV')
    parser.add_argument('--max-frames', type=int, default=300,
                       help='Số frames tối đa')
    
    args = parser.parse_args()
    
    try:
        viewer = LiveCourtViewer(args.video, args.camera)
        viewer.run_live_demo(args.tracking, args.max_frames)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()