import cv2
import numpy as np
import json
from collections import defaultdict, deque
import time
import torch

class OptimizedSan4Analysis:
    def __init__(self):
        # Load calibration
        with open('court_calibration_san4.json', 'r') as f:
            self.calibration = json.load(f)
        
        self.homography = np.array(self.calibration['homography'])
        self.court_width = self.calibration['court_width']
        self.court_length = self.calibration['court_length']
        
        # Video setup
        self.video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Video: {self.total_frames} frames at {self.fps} FPS")
        
        # YOLO setup with GPU optimization
        from ultralytics import YOLO
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {device}")
        
        self.model = YOLO('yolov8n.pt')  # Use lighter model for speed
        self.model.to(device)
        
        # Tracking data
        self.player_tracks = defaultdict(lambda: deque(maxlen=200))  # Reduced memory
        self.ball_tracks = deque(maxlen=100)
        self.current_frame = 0
        self.player_positions_smooth = {}
        
        # Performance optimization
        self.skip_frames = 2  # Process every 2nd frame for speed
        self.last_processed_frame = -1
        
        # 2D Court visualization setup
        self.court_scale = 80  # pixels per meter
        self.court_img_width = int(self.court_width * self.court_scale + 100)
        self.court_img_height = int(self.court_length * self.court_scale + 100)
        self.court_offset_x = 50
        self.court_offset_y = 50
        
        self.setup_court_template()
        
        # Statistics
        self.stats = {
            'total_players': 0,
            'total_balls': 0,
            'left_players': [],
            'right_players': []
        }
        
    def setup_court_template(self):
        """Create court template for visualization"""
        self.court_template = np.zeros((self.court_img_height, self.court_img_width, 3), dtype=np.uint8)
        
        # Court background (green)
        court_rect = (
            self.court_offset_x,
            self.court_offset_y,
            int(self.court_width * self.court_scale),
            int(self.court_length * self.court_scale)
        )
        cv2.rectangle(self.court_template, 
                     (court_rect[0], court_rect[1]),
                     (court_rect[0] + court_rect[2], court_rect[1] + court_rect[3]),
                     (50, 150, 50), -1)
        
        # Court boundary (white)
        cv2.rectangle(self.court_template,
                     (court_rect[0], court_rect[1]),
                     (court_rect[0] + court_rect[2], court_rect[1] + court_rect[3]),
                     (255, 255, 255), 3)
        
        # Net (black line)
        net_y = self.court_offset_y + int(self.court_length * self.court_scale / 2)
        cv2.line(self.court_template,
                (self.court_offset_x, net_y),
                (self.court_offset_x + int(self.court_width * self.court_scale), net_y),
                (0, 0, 0), 8)
        
        # Service lines (dashed)
        service_y1 = self.court_offset_y + int(self.court_length * self.court_scale / 4)
        service_y2 = self.court_offset_y + int(3 * self.court_length * self.court_scale / 4)
        
        # Create dashed lines
        for x in range(self.court_offset_x, self.court_offset_x + int(self.court_width * self.court_scale), 20):
            cv2.line(self.court_template, (x, service_y1), (x + 10, service_y1), (100, 100, 255), 2)
            cv2.line(self.court_template, (x, service_y2), (x + 10, service_y2), (100, 100, 255), 2)
        
        # Center line
        center_x = self.court_offset_x + int(self.court_width * self.court_scale / 2)
        for y in range(self.court_offset_y, self.court_offset_y + int(self.court_length * self.court_scale), 20):
            cv2.line(self.court_template, (center_x, y), (center_x, y + 10), (100, 100, 255), 2)
        
        # Labels
        cv2.putText(self.court_template, 'LEFT SIDE', 
                   (self.court_offset_x + 20, self.court_offset_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        cv2.putText(self.court_template, 'RIGHT SIDE',
                   (self.court_offset_x + int(self.court_width * self.court_scale) - 120, self.court_offset_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
    
    def court_to_image_coords(self, court_pos):
        """Convert court coordinates to image coordinates"""
        img_x = int(court_pos[0] * self.court_scale + self.court_offset_x)
        img_y = int(court_pos[1] * self.court_scale + self.court_offset_y)
        return (img_x, img_y)
    
    def is_on_court(self, court_pos):
        """Check if position is actually on the court"""
        return (0 <= court_pos[0] <= self.court_width and 
                0 <= court_pos[1] <= self.court_length)
    
    def get_side(self, court_pos):
        """Determine which side of court (left=0, right=1)"""
        return 0 if court_pos[0] < self.court_width / 2 else 1
    
    def transform_to_court(self, image_points):
        """Transform image coordinates to court coordinates"""
        if len(image_points) == 0:
            return []
        
        points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        court_points = cv2.perspectiveTransform(points, self.homography)
        return court_points.reshape(-1, 2)
    
    def smooth_position(self, player_id, new_pos):
        """Smooth player position using exponential moving average"""
        alpha = 0.7  # Increased for more responsiveness
        
        if player_id in self.player_positions_smooth:
            old_pos = self.player_positions_smooth[player_id]
            smoothed_pos = [
                alpha * new_pos[0] + (1 - alpha) * old_pos[0],
                alpha * new_pos[1] + (1 - alpha) * old_pos[1]
            ]
        else:
            smoothed_pos = new_pos
        
        self.player_positions_smooth[player_id] = smoothed_pos
        return smoothed_pos
    
    def process_frame(self, frame):
        """Process frame for tracking with GPU optimization"""
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame
            scale = 1.0
        
        # YOLO inference with optimized settings
        results = self.model.track(frame_resized, persist=True, verbose=False, 
                                 conf=0.3, iou=0.6, imgsz=640)  # Smaller image size
        
        players = []
        balls = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                # Handle tracking IDs
                if result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy()
                else:
                    ids = [None] * len(boxes)
                
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
                    class_name = self.model.names[int(cls)]
                    
                    if class_name == 'person' and conf > 0.4:
                        track_id = int(ids[i]) if ids[i] is not None else f"temp_{i}_{self.current_frame}"
                        x, y, w, h = box
                        
                        # Scale back to original coordinates
                        x, y = x / scale, y / scale
                        
                        players.append({
                            'id': track_id,
                            'pos': [x, y],
                            'conf': conf
                        })
                        
                    elif class_name in ['sports ball', 'ball'] and conf > 0.2:
                        x, y, w, h = box
                        ball_size = max(w, h)
                        
                        # Scale back to original coordinates
                        x, y = x / scale, y / scale
                        ball_size = ball_size / scale
                        
                        # Filter reasonable ball sizes
                        if 8 < ball_size < 80:
                            balls.append({
                                'pos': [x, y],
                                'conf': conf,
                                'size': ball_size
                            })
        
        return players, balls
    
    def filter_active_players(self, valid_players):
        """Filter to keep max 2 players per side"""
        left_players = []
        right_players = []
        
        for player_id, court_pos, conf in valid_players:
            side = self.get_side(court_pos)
            
            # Score based on confidence and tracking history
            track_bonus = min(len(self.player_tracks[player_id]) * 0.01, 0.3)
            total_score = conf + track_bonus
            
            if side == 0:
                left_players.append((player_id, court_pos, total_score))
            else:
                right_players.append((player_id, court_pos, total_score))
        
        # Keep top 2 per side
        left_players.sort(key=lambda x: x[2], reverse=True)
        right_players.sort(key=lambda x: x[2], reverse=True)
        
        return left_players[:2] + right_players[:2]
    
    def update_tracking(self, players, balls):
        """Update tracking data"""
        # Process players
        if players:
            player_positions = [p['pos'] for p in players]
            court_positions = self.transform_to_court(player_positions)
            
            # Filter players on court
            valid_players = []
            for player, court_pos in zip(players, court_positions):
                if self.is_on_court(court_pos):
                    valid_players.append((player['id'], court_pos, player['conf']))
            
            # Keep max 2 players per side
            active_players = self.filter_active_players(valid_players)
            
            # Update tracking with smoothing
            current_active_ids = set()
            for player_id, court_pos, conf in active_players:
                current_active_ids.add(player_id)
                smoothed_pos = self.smooth_position(player_id, court_pos)
                
                self.player_tracks[player_id].append({
                    'frame': self.current_frame,
                    'court_pos': smoothed_pos,
                    'confidence': conf
                })
            
            # Clean up inactive players
            inactive_threshold = 60  # Reduced for faster cleanup
            players_to_remove = []
            for player_id in list(self.player_tracks.keys()):
                if (player_id not in current_active_ids and 
                    (len(self.player_tracks[player_id]) == 0 or 
                     self.current_frame - self.player_tracks[player_id][-1]['frame'] > inactive_threshold)):
                    players_to_remove.append(player_id)
            
            for player_id in players_to_remove:
                del self.player_tracks[player_id]
                if player_id in self.player_positions_smooth:
                    del self.player_positions_smooth[player_id]
        
        # Process balls
        if balls:
            ball_positions = [b['pos'] for b in balls]
            court_positions = self.transform_to_court(ball_positions)
            
            for i, court_pos in enumerate(court_positions):
                if self.is_on_court(court_pos):
                    self.ball_tracks.append({
                        'frame': self.current_frame,
                        'court_pos': court_pos,
                        'confidence': balls[i]['conf']
                    })
    
    def create_court_visualization(self):
        """Create 2D court visualization"""
        court_img = self.court_template.copy()
        
        # Player colors
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # Update side statistics
        left_players = []
        right_players = []
        
        # Draw players and trails
        for i, (player_id, track) in enumerate(self.player_tracks.items()):
            if len(track) > 0:
                color = colors[i % len(colors)]
                
                # Current position
                current_pos = track[-1]['court_pos']
                img_pos = self.court_to_image_coords(current_pos)
                
                # Update side stats
                if self.get_side(current_pos) == 0:
                    left_players.append(player_id)
                else:
                    right_players.append(player_id)
                
                # Draw trail (last 50 positions)
                if len(track) > 5:
                    recent_track = list(track)[-50:]
                    for j in range(1, len(recent_track)):
                        pos1 = self.court_to_image_coords(recent_track[j-1]['court_pos'])
                        pos2 = self.court_to_image_coords(recent_track[j]['court_pos'])
                        
                        # Fade effect
                        alpha = j / len(recent_track)
                        fade_color = tuple(int(c * alpha * 0.7) for c in color)
                        
                        cv2.line(court_img, pos1, pos2, fade_color, 2)
                
                # Draw current position
                cv2.circle(court_img, img_pos, 12, color, -1)
                cv2.circle(court_img, img_pos, 12, (255, 255, 255), 2)
                
                # Player ID
                cv2.putText(court_img, f'P{player_id}', 
                           (img_pos[0] + 15, img_pos[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw ball trail
        if len(self.ball_tracks) > 1:
            recent_balls = [b for b in self.ball_tracks if self.current_frame - b['frame'] < 30]
            
            if len(recent_balls) > 1:
                # Ball trail
                for j in range(1, len(recent_balls)):
                    pos1 = self.court_to_image_coords(recent_balls[j-1]['court_pos'])
                    pos2 = self.court_to_image_coords(recent_balls[j]['court_pos'])
                    cv2.line(court_img, pos1, pos2, (0, 255, 255), 3)
                
                # Current ball position
                if recent_balls:
                    ball_pos = self.court_to_image_coords(recent_balls[-1]['court_pos'])
                    cv2.circle(court_img, ball_pos, 8, (0, 255, 255), -1)
                    cv2.circle(court_img, ball_pos, 8, (0, 0, 0), 2)
        
        # Update statistics
        self.stats['left_players'] = left_players
        self.stats['right_players'] = right_players
        self.stats['total_players'] = len(self.player_tracks)
        self.stats['total_balls'] = len([b for b in self.ball_tracks if self.current_frame - b['frame'] < 30])
        
        return court_img
    
    def add_stats_overlay(self, court_img):
        """Add statistics overlay to court image"""
        # Stats background
        overlay = court_img.copy()
        cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, court_img, 0.3, 0, court_img)
        
        # Stats text
        stats_text = [
            f"Frame: {self.current_frame:,}/{self.total_frames:,}",
            f"Progress: {self.current_frame/self.total_frames*100:.1f}%",
            f"Players: {self.stats['total_players']}/4",
            f"Left: {len(self.stats['left_players'])}/2",
            f"Right: {len(self.stats['right_players'])}/2",
            f"Ball detections: {self.stats['total_balls']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(court_img, text, (20, 35 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return court_img
    
    def run(self):
        """Main execution loop"""
        print("🚀 Starting Optimized San4 Analysis...")
        print("⚡ Features: GPU acceleration, optimized visualization, smooth tracking")
        print("⏹️  Press 'q' to quit")
        
        # Window setup
        cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
        cv2.namedWindow('2D Court Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original Video', 800, 600)
        cv2.resizeWindow('2D Court Tracking', 600, 700)
        
        frame_time_buffer = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("🏁 End of video reached")
                    break
                
                self.current_frame += 1
                
                # Process frames (skip some for performance)
                process_this_frame = (self.current_frame % self.skip_frames == 0)
                
                if process_this_frame:
                    players, balls = self.process_frame(frame)
                    self.update_tracking(players, balls)
                    self.last_processed_frame = self.current_frame
                
                # Always show visualization
                # Original video with basic overlay
                display_frame = frame.copy()
                
                # Add frame info
                cv2.putText(display_frame, f"Frame: {self.current_frame:,}/{self.total_frames:,}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Progress: {self.current_frame/self.total_frames*100:.1f}%", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 2D Court visualization
                court_img = self.create_court_visualization()
                court_img = self.add_stats_overlay(court_img)
                
                # Show windows
                cv2.imshow('Original Video', display_frame)
                cv2.imshow('2D Court Tracking', court_img)
                
                # FPS calculation
                frame_time = time.time() - start_time
                frame_time_buffer.append(frame_time)
                avg_fps = 1.0 / (sum(frame_time_buffer) / len(frame_time_buffer))
                
                # Progress report
                if self.current_frame % 100 == 0:
                    print(f"⚡ Frame {self.current_frame:,} | Players: {self.stats['total_players']} | FPS: {avg_fps:.1f}")
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("⏹️  Analysis stopped by user")
                    break
                
        except KeyboardInterrupt:
            print("\n⏹️  Analysis stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("🎉 Analysis complete!")

if __name__ == "__main__":
    analyzer = OptimizedSan4Analysis()
    analyzer.run()