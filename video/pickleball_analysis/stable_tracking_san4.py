import cv2
import numpy as np
import json
from collections import defaultdict, deque
import time
import torch
from scipy.spatial.distance import cdist

class StableTrackingSan4:
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
        
        # YOLO setup
        from ultralytics import YOLO
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {device}")
        
        self.model = YOLO('yolov8n.pt')
        self.model.to(device)
        
        # Fixed player tracking - 4 players total
        self.fixed_players = {
            'left_1': {'id': 1, 'side': 0, 'active': False, 'last_pos': None, 'confidence_history': deque(maxlen=10)},
            'left_2': {'id': 2, 'side': 0, 'active': False, 'last_pos': None, 'confidence_history': deque(maxlen=10)},
            'right_1': {'id': 3, 'side': 1, 'active': False, 'last_pos': None, 'confidence_history': deque(maxlen=10)},
            'right_2': {'id': 4, 'side': 1, 'active': False, 'last_pos': None, 'confidence_history': deque(maxlen=10)}
        }
        
        # Player tracking data
        self.player_tracks = {
            1: deque(maxlen=200),
            2: deque(maxlen=200),
            3: deque(maxlen=200),
            4: deque(maxlen=200)
        }
        
        # Ball tracking
        self.ball_tracks = deque(maxlen=100)
        self.ball_position_smooth = None
        
        self.current_frame = 0
        
        # Performance optimization
        self.skip_frames = 2
        self.process_size = 640
        
        # Visualization setup
        self.court_scale = 70
        self.court_img_width = int(self.court_width * self.court_scale + 100)
        self.court_img_height = int(self.court_length * self.court_scale + 100)
        self.court_offset = 50
        
        self.setup_court_template()
        
        # Assignment parameters
        self.max_assignment_distance = 1.5  # meters
        self.min_confidence_threshold = 0.4
        self.player_inactive_threshold = 60  # frames
        
    def setup_court_template(self):
        """Create court visualization template"""
        self.court_template = np.zeros((self.court_img_height, self.court_img_width, 3), dtype=np.uint8)
        
        # Court background
        court_rect = (
            self.court_offset,
            self.court_offset,
            int(self.court_width * self.court_scale),
            int(self.court_length * self.court_scale)
        )
        cv2.rectangle(self.court_template, 
                     (court_rect[0], court_rect[1]),
                     (court_rect[0] + court_rect[2], court_rect[1] + court_rect[3]),
                     (50, 150, 50), -1)
        
        # Court boundary
        cv2.rectangle(self.court_template,
                     (court_rect[0], court_rect[1]),
                     (court_rect[0] + court_rect[2], court_rect[1] + court_rect[3]),
                     (255, 255, 255), 3)
        
        # Net
        net_y = self.court_offset + int(self.court_length * self.court_scale / 2)
        cv2.line(self.court_template,
                (self.court_offset, net_y),
                (self.court_offset + int(self.court_width * self.court_scale), net_y),
                (0, 0, 0), 6)
        
        # Service lines
        service_y1 = self.court_offset + int(self.court_length * self.court_scale / 4)
        service_y2 = self.court_offset + int(3 * self.court_length * self.court_scale / 4)
        
        for x in range(self.court_offset, self.court_offset + int(self.court_width * self.court_scale), 15):
            cv2.line(self.court_template, (x, service_y1), (x + 8, service_y1), (100, 100, 255), 2)
            cv2.line(self.court_template, (x, service_y2), (x + 8, service_y2), (100, 100, 255), 2)
        
        # Center line
        center_x = self.court_offset + int(self.court_width * self.court_scale / 2)
        for y in range(self.court_offset, self.court_offset + int(self.court_length * self.court_scale), 15):
            cv2.line(self.court_template, (center_x, y), (center_x, y + 8), (100, 100, 255), 2)
        
        # Side labels
        cv2.putText(self.court_template, 'LEFT SIDE', 
                   (self.court_offset + 20, self.court_offset - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        cv2.putText(self.court_template, 'RIGHT SIDE',
                   (self.court_offset + int(self.court_width * self.court_scale) - 120, self.court_offset - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
    
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
    
    def smooth_position(self, current_pos, last_pos, alpha=0.7):
        """Smooth position using exponential moving average"""
        if last_pos is None:
            return current_pos
        
        return [
            alpha * current_pos[0] + (1 - alpha) * last_pos[0],
            alpha * current_pos[1] + (1 - alpha) * last_pos[1]
        ]
    
    def process_frame(self, frame):
        """Process frame for detection"""
        # Resize for processing
        height, width = frame.shape[:2]
        if width > self.process_size:
            scale = self.process_size / width
            new_width = self.process_size
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame
            scale = 1.0
        
        # YOLO inference
        results = self.model(frame_resized, verbose=False, conf=0.3, iou=0.6)
        
        detected_players = []
        detected_balls = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    class_name = self.model.names[int(cls)]
                    
                    if class_name == 'person' and conf > self.min_confidence_threshold:
                        x, y, w, h = box
                        x, y = x / scale, y / scale  # Scale back
                        
                        detected_players.append({
                            'pos': [x, y],
                            'conf': conf,
                            'size': max(w, h) / scale
                        })
                        
                    elif class_name in ['sports ball', 'ball'] and conf > 0.2:
                        x, y, w, h = box
                        x, y = x / scale, y / scale
                        ball_size = max(w, h) / scale
                        
                        # Filter reasonable ball sizes
                        if 8 < ball_size < 80:
                            detected_balls.append({
                                'pos': [x, y],
                                'conf': conf,
                                'size': ball_size
                            })
        
        return detected_players, detected_balls
    
    def assign_players_to_fixed_positions(self, detected_players):
        """Assign detected players to fixed 4-player structure"""
        if not detected_players:
            return
        
        # Transform detections to court coordinates
        player_positions = [p['pos'] for p in detected_players]
        court_positions = self.transform_to_court(player_positions)
        
        # Filter players on court
        valid_detections = []
        for i, court_pos in enumerate(court_positions):
            if self.is_on_court(court_pos):
                detection = detected_players[i].copy()
                detection['court_pos'] = court_pos
                detection['side'] = self.get_side(court_pos)
                valid_detections.append(detection)
        
        if not valid_detections:
            return
        
        # Separate by side
        left_detections = [d for d in valid_detections if d['side'] == 0]
        right_detections = [d for d in valid_detections if d['side'] == 1]
        
        # Assign left side players
        self.assign_side_players(left_detections, ['left_1', 'left_2'])
        
        # Assign right side players
        self.assign_side_players(right_detections, ['right_1', 'right_2'])
        
        # Deactivate players not seen for too long
        self.deactivate_inactive_players()
    
    def assign_side_players(self, detections, player_keys):
        """Assign detections to specific side players"""
        if not detections:
            return
        
        # Get active players on this side
        active_players = []
        inactive_players = []
        
        for key in player_keys:
            if self.fixed_players[key]['active'] and self.fixed_players[key]['last_pos'] is not None:
                active_players.append(key)
            else:
                inactive_players.append(key)
        
        # Calculate distances for active players
        if active_players:
            active_positions = [self.fixed_players[key]['last_pos'] for key in active_players]
            detection_positions = [d['court_pos'] for d in detections]
            
            if len(active_positions) > 0 and len(detection_positions) > 0:
                distances = cdist(active_positions, detection_positions)
                
                # Hungarian assignment (simplified)
                assigned_detections = set()
                
                for i, player_key in enumerate(active_players):
                    if i < len(distances):
                        best_detection_idx = np.argmin(distances[i])
                        min_distance = distances[i][best_detection_idx]
                        
                        if (min_distance < self.max_assignment_distance and 
                            best_detection_idx not in assigned_detections):
                            
                            # Assign this detection to the player
                            detection = detections[best_detection_idx]
                            self.update_player_position(player_key, detection)
                            assigned_detections.add(best_detection_idx)
                
                # Assign remaining detections to inactive players
                remaining_detections = [d for i, d in enumerate(detections) if i not in assigned_detections]
            else:
                remaining_detections = detections
        else:
            remaining_detections = detections
        
        # Assign remaining detections to inactive players
        for i, detection in enumerate(remaining_detections[:len(inactive_players)]):
            if i < len(inactive_players):
                player_key = inactive_players[i]
                self.update_player_position(player_key, detection)
    
    def update_player_position(self, player_key, detection):
        """Update player position with smoothing"""
        player = self.fixed_players[player_key]
        
        # Smooth position
        smoothed_pos = self.smooth_position(
            detection['court_pos'], 
            player['last_pos']
        )
        
        # Update player data
        player['active'] = True
        player['last_pos'] = smoothed_pos
        player['confidence_history'].append(detection['conf'])
        player['last_seen_frame'] = self.current_frame
        
        # Add to tracking history
        player_id = player['id']
        self.player_tracks[player_id].append({
            'frame': self.current_frame,
            'court_pos': smoothed_pos,
            'confidence': detection['conf'],
            'side': player['side']
        })
    
    def deactivate_inactive_players(self):
        """Deactivate players not seen for too long"""
        for player_key, player in self.fixed_players.items():
            if (player['active'] and 
                hasattr(player, 'last_seen_frame') and
                self.current_frame - player.get('last_seen_frame', 0) > self.player_inactive_threshold):
                
                player['active'] = False
                print(f"🔄 Player {player['id']} ({player_key}) deactivated after {self.player_inactive_threshold} frames")
    
    def update_ball_tracking(self, detected_balls):
        """Update ball tracking with smoothing"""
        if not detected_balls:
            return
        
        # Transform balls to court coordinates
        ball_positions = [b['pos'] for b in detected_balls]
        court_positions = self.transform_to_court(ball_positions)
        
        # Filter balls on court and find best one
        valid_balls = []
        for i, court_pos in enumerate(court_positions):
            if self.is_on_court(court_pos):
                ball = detected_balls[i].copy()
                ball['court_pos'] = court_pos
                valid_balls.append(ball)
        
        if not valid_balls:
            return
        
        # Take the ball with highest confidence
        best_ball = max(valid_balls, key=lambda b: b['conf'])
        
        # Smooth ball position
        smoothed_pos = self.smooth_position(
            best_ball['court_pos'],
            self.ball_position_smooth,
            alpha=0.8  # More responsive for ball
        )
        
        self.ball_position_smooth = smoothed_pos
        
        # Add to ball tracking history
        self.ball_tracks.append({
            'frame': self.current_frame,
            'court_pos': smoothed_pos,
            'confidence': best_ball['conf']
        })
    
    def create_visualization(self):
        """Create court visualization"""
        court_img = self.court_template.copy()
        
        # Player colors
        player_colors = {
            1: (0, 0, 255),    # Red for left_1
            2: (255, 0, 0),    # Blue for left_2  
            3: (0, 255, 0),    # Green for right_1
            4: (255, 255, 0)   # Cyan for right_2
        }
        
        # Draw player tracks and current positions
        active_count = {'left': 0, 'right': 0}
        
        for player_key, player in self.fixed_players.items():
            if player['active'] and player['last_pos'] is not None:
                player_id = player['id']
                color = player_colors[player_id]
                
                # Count active players
                if player['side'] == 0:
                    active_count['left'] += 1
                else:
                    active_count['right'] += 1
                
                # Draw trail
                track = self.player_tracks[player_id]
                if len(track) > 3:
                    recent_track = list(track)[-50:]  # Last 50 positions
                    for i in range(1, len(recent_track)):
                        pos1 = self.court_to_image_coords(recent_track[i-1]['court_pos'])
                        pos2 = self.court_to_image_coords(recent_track[i]['court_pos'])
                        
                        # Fade effect
                        alpha = i / len(recent_track)
                        fade_color = tuple(int(c * alpha * 0.6) for c in color)
                        cv2.line(court_img, pos1, pos2, fade_color, 2)
                
                # Draw current position
                current_pos = self.court_to_image_coords(player['last_pos'])
                cv2.circle(court_img, current_pos, 12, color, -1)
                cv2.circle(court_img, current_pos, 12, (255, 255, 255), 2)
                
                # Player ID
                cv2.putText(court_img, f'P{player_id}', 
                           (current_pos[0] + 15, current_pos[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw ball trail and position
        if len(self.ball_tracks) > 0:
            recent_balls = [b for b in self.ball_tracks if self.current_frame - b['frame'] < 30]
            
            if len(recent_balls) > 1:
                # Ball trail
                for i in range(1, len(recent_balls)):
                    pos1 = self.court_to_image_coords(recent_balls[i-1]['court_pos'])
                    pos2 = self.court_to_image_coords(recent_balls[i]['court_pos'])
                    cv2.line(court_img, pos1, pos2, (0, 255, 255), 3)
            
            if recent_balls:
                # Current ball position
                ball_pos = self.court_to_image_coords(recent_balls[-1]['court_pos'])
                cv2.circle(court_img, ball_pos, 8, (0, 255, 255), -1)
                cv2.circle(court_img, ball_pos, 8, (0, 0, 0), 2)
        
        # Add statistics overlay
        overlay = court_img.copy()
        cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, court_img, 0.2, 0, court_img)
        
        # Stats text
        stats = [
            f"Frame: {self.current_frame:,}/{self.total_frames:,}",
            f"Progress: {self.current_frame/self.total_frames*100:.1f}%",
            f"LEFT SIDE: {active_count['left']}/2 players",
            f"RIGHT SIDE: {active_count['right']}/2 players",
            f"Ball detections: {len([b for b in self.ball_tracks if self.current_frame - b['frame'] < 30])}",
            f"Total active: {active_count['left'] + active_count['right']}/4"
        ]
        
        for i, text in enumerate(stats):
            cv2.putText(court_img, text, (20, 35 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return court_img
    
    def run(self):
        """Main execution loop"""
        print("🚀 Starting Stable 4-Player Tracking Analysis...")
        print("👥 Fixed 4-player structure: 2 left + 2 right")
        print("🎾 Ball tracking with cross-court movement")
        print("⏹️  Press 'q' to quit")
        
        # Setup windows
        cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
        cv2.namedWindow('2D Court - Stable Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Original Video', 800, 600)
        cv2.resizeWindow('2D Court - Stable Tracking', 600, 700)
        
        frame_times = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("🏁 Video complete")
                    break
                
                self.current_frame += 1
                
                # Process every N frames
                if self.current_frame % self.skip_frames == 0:
                    detected_players, detected_balls = self.process_frame(frame)
                    self.assign_players_to_fixed_positions(detected_players)
                    self.update_ball_tracking(detected_balls)
                
                # Visualize
                # Original video
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Frame: {self.current_frame:,}/{self.total_frames:,}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Stable 4-Player Tracking", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 2D Court
                court_img = self.create_visualization()
                
                # Show windows
                cv2.imshow('Original Video', display_frame)
                cv2.imshow('2D Court - Stable Tracking', court_img)
                
                # Performance tracking
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                
                # Progress report
                if self.current_frame % 100 == 0:
                    active_players = sum(1 for p in self.fixed_players.values() if p['active'])
                    recent_balls = len([b for b in self.ball_tracks if self.current_frame - b['frame'] < 30])
                    print(f"⚡ Frame {self.current_frame:,} | Active players: {active_players}/4 | Recent balls: {recent_balls} | FPS: {avg_fps:.1f}")
                
                # Exit check
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️  Analysis stopped by user")
                    break
                
        except KeyboardInterrupt:
            print("\n⏹️  Analysis stopped")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("🎉 Analysis complete!")
            
            # Print final statistics
            print("\n📊 FINAL STATISTICS:")
            for player_key, player in self.fixed_players.items():
                track_length = len(self.player_tracks[player['id']])
                print(f"  Player {player['id']} ({player_key}): {track_length} tracked frames")
            print(f"  Ball: {len(self.ball_tracks)} tracked frames")

if __name__ == "__main__":
    analyzer = StableTrackingSan4()
    analyzer.run()