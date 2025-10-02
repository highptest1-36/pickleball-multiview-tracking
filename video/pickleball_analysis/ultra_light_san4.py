import cv2
import numpy as np
import json
from collections import defaultdict, deque
import time
import torch

class UltraLightSan4Analysis:
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
        
        # YOLO setup - ultra light
        from ultralytics import YOLO
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {device}")
        
        self.model = YOLO('yolov8n.pt')  # Lightest model
        self.model.to(device)
        
        # Tracking data - minimal memory
        self.player_tracks = defaultdict(lambda: deque(maxlen=50))  # Very short tracks
        self.ball_tracks = deque(maxlen=30)
        self.current_frame = 0
        self.player_positions_smooth = {}
        
        # Ultra performance optimization
        self.skip_frames = 3  # Process every 3rd frame
        self.process_size = 480  # Very small processing size
        
        # Simple 2D visualization
        self.court_scale = 60  # Smaller scale
        self.court_img_width = int(self.court_width * self.court_scale + 80)
        self.court_img_height = int(self.court_length * self.court_scale + 80)
        self.court_offset = 40
        
        self.setup_simple_court()
        
        # Performance tracking
        self.frame_times = deque(maxlen=10)
        
    def setup_simple_court(self):
        """Create simple court template"""
        self.court_template = np.zeros((self.court_img_height, self.court_img_width, 3), dtype=np.uint8)
        
        # Court (green rectangle)
        court_rect = (
            self.court_offset,
            self.court_offset,
            int(self.court_width * self.court_scale),
            int(self.court_length * self.court_scale)
        )
        cv2.rectangle(self.court_template, 
                     (court_rect[0], court_rect[1]),
                     (court_rect[0] + court_rect[2], court_rect[1] + court_rect[3]),
                     (40, 120, 40), -1)
        
        # Court boundary
        cv2.rectangle(self.court_template,
                     (court_rect[0], court_rect[1]),
                     (court_rect[0] + court_rect[2], court_rect[1] + court_rect[3]),
                     (255, 255, 255), 2)
        
        # Net
        net_y = self.court_offset + int(self.court_length * self.court_scale / 2)
        cv2.line(self.court_template,
                (self.court_offset, net_y),
                (self.court_offset + int(self.court_width * self.court_scale), net_y),
                (255, 255, 255), 4)
        
        # Center line
        center_x = self.court_offset + int(self.court_width * self.court_scale / 2)
        cv2.line(self.court_template,
                (center_x, self.court_offset),
                (center_x, self.court_offset + int(self.court_length * self.court_scale)),
                (100, 100, 255), 1)
        
        # Labels
        cv2.putText(self.court_template, 'LEFT', 
                   (self.court_offset + 10, self.court_offset - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        cv2.putText(self.court_template, 'RIGHT',
                   (self.court_offset + int(self.court_width * self.court_scale) - 50, self.court_offset - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    
    def court_to_image_coords(self, court_pos):
        """Convert court coordinates to image coordinates"""
        img_x = int(court_pos[0] * self.court_scale + self.court_offset)
        img_y = int(court_pos[1] * self.court_scale + self.court_offset)
        return (img_x, img_y)
    
    def is_on_court(self, court_pos):
        """Check if position is on court"""
        return (0 <= court_pos[0] <= self.court_width and 
                0 <= court_pos[1] <= self.court_length)
    
    def get_side(self, court_pos):
        """Get court side (0=left, 1=right)"""
        return 0 if court_pos[0] < self.court_width / 2 else 1
    
    def transform_to_court(self, image_points):
        """Transform to court coordinates"""
        if len(image_points) == 0:
            return []
        
        points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        court_points = cv2.perspectiveTransform(points, self.homography)
        return court_points.reshape(-1, 2)
    
    def smooth_position(self, player_id, new_pos):
        """Simple position smoothing"""
        alpha = 0.8  # High responsiveness
        
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
    
    def process_frame_fast(self, frame):
        """Ultra-fast frame processing"""
        # Heavy downscale for speed
        height, width = frame.shape[:2]
        scale = self.process_size / width
        new_width = self.process_size
        new_height = int(height * scale)
        frame_small = cv2.resize(frame, (new_width, new_height))
        
        # Fast YOLO inference
        results = self.model.track(frame_small, persist=True, verbose=False, 
                                 conf=0.5, iou=0.7, imgsz=320)  # Very small
        
        players = []
        balls = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                if result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy()
                else:
                    ids = [None] * len(boxes)
                
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
                    class_name = self.model.names[int(cls)]
                    
                    if class_name == 'person' and conf > 0.5:
                        track_id = int(ids[i]) if ids[i] is not None else f"t_{i}"
                        x, y, w, h = box
                        
                        # Scale back
                        x, y = x / scale, y / scale
                        
                        players.append({
                            'id': track_id,
                            'pos': [x, y],
                            'conf': conf
                        })
                        
                    elif class_name in ['sports ball', 'ball'] and conf > 0.3:
                        x, y, w, h = box
                        x, y = x / scale, y / scale
                        
                        balls.append({
                            'pos': [x, y],
                            'conf': conf
                        })
        
        return players, balls
    
    def filter_players_fast(self, valid_players):
        """Fast player filtering"""
        left_players = []
        right_players = []
        
        for player_id, court_pos, conf in valid_players:
            side = self.get_side(court_pos)
            
            if side == 0:
                left_players.append((player_id, court_pos, conf))
            else:
                right_players.append((player_id, court_pos, conf))
        
        # Keep top 2 per side by confidence
        left_players.sort(key=lambda x: x[2], reverse=True)
        right_players.sort(key=lambda x: x[2], reverse=True)
        
        return left_players[:2] + right_players[:2]
    
    def update_tracking_fast(self, players, balls):
        """Fast tracking update"""
        # Players
        if players:
            player_positions = [p['pos'] for p in players]
            court_positions = self.transform_to_court(player_positions)
            
            valid_players = []
            for player, court_pos in zip(players, court_positions):
                if self.is_on_court(court_pos):
                    valid_players.append((player['id'], court_pos, player['conf']))
            
            active_players = self.filter_players_fast(valid_players)
            
            # Update with smoothing
            current_ids = set()
            for player_id, court_pos, conf in active_players:
                current_ids.add(player_id)
                smoothed_pos = self.smooth_position(player_id, court_pos)
                
                self.player_tracks[player_id].append({
                    'frame': self.current_frame,
                    'pos': smoothed_pos,
                    'conf': conf
                })
            
            # Quick cleanup
            for player_id in list(self.player_tracks.keys()):
                if (player_id not in current_ids and 
                    (len(self.player_tracks[player_id]) == 0 or 
                     self.current_frame - self.player_tracks[player_id][-1]['frame'] > 30)):
                    del self.player_tracks[player_id]
                    if player_id in self.player_positions_smooth:
                        del self.player_positions_smooth[player_id]
        
        # Balls
        if balls:
            ball_positions = [b['pos'] for b in balls]
            court_positions = self.transform_to_court(ball_positions)
            
            for court_pos in court_positions:
                if self.is_on_court(court_pos):
                    self.ball_tracks.append({
                        'frame': self.current_frame,
                        'pos': court_pos
                    })
    
    def create_simple_court_view(self):
        """Create simple 2D court view"""
        court_img = self.court_template.copy()
        
        # Simple player colors
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
        
        # Draw players
        left_count = 0
        right_count = 0
        
        for i, (player_id, track) in enumerate(self.player_tracks.items()):
            if len(track) > 0:
                color = colors[i % len(colors)]
                
                # Current position
                current_pos = track[-1]['pos']
                img_pos = self.court_to_image_coords(current_pos)
                
                # Count sides
                if self.get_side(current_pos) == 0:
                    left_count += 1
                else:
                    right_count += 1
                
                # Simple trail (last 10 positions)
                if len(track) > 3:
                    recent = list(track)[-10:]
                    for j in range(1, len(recent)):
                        pos1 = self.court_to_image_coords(recent[j-1]['pos'])
                        pos2 = self.court_to_image_coords(recent[j]['pos'])
                        cv2.line(court_img, pos1, pos2, color, 1)
                
                # Player dot
                cv2.circle(court_img, img_pos, 8, color, -1)
                cv2.circle(court_img, img_pos, 8, (255, 255, 255), 1)
                
                # ID
                cv2.putText(court_img, str(player_id), 
                           (img_pos[0] + 10, img_pos[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw ball
        if len(self.ball_tracks) > 0:
            recent_balls = [b for b in self.ball_tracks if self.current_frame - b['frame'] < 15]
            if recent_balls:
                ball_pos = self.court_to_image_coords(recent_balls[-1]['pos'])
                cv2.circle(court_img, ball_pos, 5, (0, 255, 255), -1)
        
        # Simple stats
        cv2.putText(court_img, f"Frame: {self.current_frame}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(court_img, f"Left: {left_count}/2  Right: {right_count}/2", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return court_img
    
    def run(self):
        """Ultra-fast main loop"""
        print("🚀 Starting Ultra-Light San4 Analysis...")
        print("⚡ Maximum performance mode - OpenCV only")
        print("⏹️  Press 'q' to quit")
        
        # Setup windows
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('2D Court', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original', 640, 480)
        cv2.resizeWindow('2D Court', 400, 500)
        
        last_process_frame = 0
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("🏁 Video complete")
                    break
                
                self.current_frame += 1
                
                # Process every N frames
                if self.current_frame - last_process_frame >= self.skip_frames:
                    players, balls = self.process_frame_fast(frame)
                    self.update_tracking_fast(players, balls)
                    last_process_frame = self.current_frame
                
                # Simple original video display
                display_frame = cv2.resize(frame, (640, 480))
                cv2.putText(display_frame, f"Frame: {self.current_frame:,}/{self.total_frames:,}", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"{self.current_frame/self.total_frames*100:.1f}%", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 2D court
                court_img = self.create_simple_court_view()
                
                # Show
                cv2.imshow('Original', display_frame)
                cv2.imshow('2D Court', court_img)
                
                # FPS tracking
                frame_time = time.time() - start_time
                self.frame_times.append(frame_time)
                
                if len(self.frame_times) > 0:
                    avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                    
                    if self.current_frame % 100 == 0:
                        players_count = len(self.player_tracks)
                        print(f"⚡ Frame {self.current_frame:,} | Players: {players_count} | FPS: {avg_fps:.1f}")
                
                # Quick exit check
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("🎉 Complete!")

if __name__ == "__main__":
    analyzer = UltraLightSan4Analysis()
    analyzer.run()