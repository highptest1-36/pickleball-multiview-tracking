import cv2
import numpy as np
import json
from collections import defaultdict, deque
import time
import torch
from scipy.spatial.distance import euclidean

class FixedTrackingSan4:
    def __init__(self):
        # Load calibration
        with open('court_calibration_san4.json', 'r') as f:
            self.calibration = json.load(f)
        
        self.homography = np.array(self.calibration['homography'])
        self.homography_inv = np.linalg.inv(self.homography)
        self.court_width = self.calibration['court_width']
        self.court_length = self.calibration['court_length']
        self.image_points = np.array(self.calibration['image_points'])
        
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
        
        # Fixed player tracking - 4 players only
        self.fixed_players = {
            'left_1': {'id': 'L1', 'side': 'left', 'position': None, 'last_seen': 0, 'track': deque(maxlen=100)},
            'left_2': {'id': 'L2', 'side': 'left', 'position': None, 'last_seen': 0, 'track': deque(maxlen=100)},
            'right_1': {'id': 'R1', 'side': 'right', 'position': None, 'last_seen': 0, 'track': deque(maxlen=100)},
            'right_2': {'id': 'R2', 'side': 'right', 'position': None, 'last_seen': 0, 'track': deque(maxlen=100)}
        }
        
        # Ball tracking
        self.ball_tracks = deque(maxlen=200)
        self.current_frame = 0
        
        # Performance settings
        self.skip_frames = 2
        self.process_size = 640
        
        # Court visualization setup
        self.setup_court_overlay()
        
    def setup_court_overlay(self):
        """Setup court overlay for visualization"""
        # Court boundary points for overlay
        self.court_boundary = np.array(self.image_points, dtype=np.int32)
        
        # Calculate center line and net position in image coordinates
        court_corners = np.array([
            [0, 0],
            [self.court_width, 0], 
            [self.court_width, self.court_length],
            [0, self.court_length]
        ], dtype=np.float32)
        
        # Net line (center horizontal)
        net_court_points = np.array([
            [0, self.court_length/2],
            [self.court_width, self.court_length/2]
        ], dtype=np.float32)
        
        # Transform to image coordinates
        net_image_points = cv2.perspectiveTransform(
            net_court_points.reshape(-1, 1, 2), 
            self.homography_inv
        ).reshape(-1, 2)
        
        self.net_line = net_image_points.astype(np.int32)
        
        # Center line (vertical)
        center_court_points = np.array([
            [self.court_width/2, 0],
            [self.court_width/2, self.court_length]
        ], dtype=np.float32)
        
        center_image_points = cv2.perspectiveTransform(
            center_court_points.reshape(-1, 1, 2),
            self.homography_inv
        ).reshape(-1, 2)
        
        self.center_line = center_image_points.astype(np.int32)
    
    def is_on_court(self, court_pos):
        """Check if position is on court"""
        return (0 <= court_pos[0] <= self.court_width and 
                0 <= court_pos[1] <= self.court_length)
    
    def get_court_side(self, court_pos):
        """Get court side (left/right)"""
        return 'left' if court_pos[0] < self.court_width / 2 else 'right'
    
    def transform_to_court(self, image_points):
        """Transform image coordinates to court coordinates"""
        if len(image_points) == 0:
            return []
        
        points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        court_points = cv2.perspectiveTransform(points, self.homography)
        return court_points.reshape(-1, 2)
    
    def transform_to_image(self, court_points):
        """Transform court coordinates to image coordinates"""
        if len(court_points) == 0:
            return []
        
        points = np.array(court_points, dtype=np.float32).reshape(-1, 1, 2)
        image_points = cv2.perspectiveTransform(points, self.homography_inv)
        return image_points.reshape(-1, 2)
    
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
        
        # YOLO detection
        results = self.model(frame_resized, verbose=False, conf=0.3, imgsz=320)
        
        detections = {'persons': [], 'balls': []}
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    class_name = self.model.names[int(cls)]
                    x, y, w, h = box
                    
                    # Scale back to original coordinates
                    x, y = x / scale, y / scale
                    
                    if class_name == 'person' and conf > 0.4:
                        detections['persons'].append({
                            'pos': [x, y],
                            'conf': conf,
                            'box': [x-w/(2*scale), y-h/(2*scale), w/scale, h/scale]
                        })
                    
                    elif class_name in ['sports ball', 'ball'] and conf > 0.2:
                        ball_size = max(w, h) / scale
                        if 8 < ball_size < 80:
                            detections['balls'].append({
                                'pos': [x, y],
                                'conf': conf,
                                'size': ball_size
                            })
        
        return detections
    
    def assign_players_to_fixed_slots(self, person_detections):
        """Assign detected persons to fixed player slots"""
        if not person_detections:
            return
        
        # Transform detections to court coordinates
        detection_positions = [d['pos'] for d in person_detections]
        court_positions = self.transform_to_court(detection_positions)
        
        # Filter detections on court
        valid_detections = []
        for i, court_pos in enumerate(court_positions):
            if self.is_on_court(court_pos):
                side = self.get_court_side(court_pos)
                valid_detections.append({
                    'image_pos': person_detections[i]['pos'],
                    'court_pos': court_pos,
                    'conf': person_detections[i]['conf'],
                    'side': side,
                    'box': person_detections[i]['box']
                })
        
        # Separate by sides
        left_detections = [d for d in valid_detections if d['side'] == 'left']
        right_detections = [d for d in valid_detections if d['side'] == 'right']
        
        # Assign to fixed slots
        self.assign_to_side_slots(left_detections, ['left_1', 'left_2'])
        self.assign_to_side_slots(right_detections, ['right_1', 'right_2'])
    
    def assign_to_side_slots(self, detections, slot_names):
        """Assign detections to specific side slots"""
        if not detections:
            return
        
        # Get current active slots for this side
        active_slots = []
        for slot_name in slot_names:
            if (self.fixed_players[slot_name]['position'] is not None and 
                self.current_frame - self.fixed_players[slot_name]['last_seen'] < 30):
                active_slots.append(slot_name)
        
        # Sort detections by confidence
        detections.sort(key=lambda x: x['conf'], reverse=True)
        
        # Assign detections to slots
        assigned_detections = []
        
        # First, try to match with existing active slots
        for slot_name in active_slots:
            if not detections:
                break
            
            slot_pos = self.fixed_players[slot_name]['position']
            if slot_pos is not None:
                # Find closest detection
                min_dist = float('inf')
                best_idx = -1
                
                for i, detection in enumerate(detections):
                    dist = euclidean(detection['court_pos'], slot_pos)
                    if dist < min_dist and dist < 1.5:  # Max 1.5m movement
                        min_dist = dist
                        best_idx = i
                
                if best_idx >= 0:
                    detection = detections.pop(best_idx)
                    self.update_player_slot(slot_name, detection)
                    assigned_detections.append(detection)
        
        # Assign remaining detections to empty slots
        empty_slots = [name for name in slot_names if name not in active_slots]
        for slot_name in empty_slots:
            if detections:
                detection = detections.pop(0)  # Take highest confidence
                self.update_player_slot(slot_name, detection)
                assigned_detections.append(detection)
    
    def update_player_slot(self, slot_name, detection):
        """Update a player slot with new detection"""
        player = self.fixed_players[slot_name]
        player['position'] = detection['court_pos']
        player['last_seen'] = self.current_frame
        player['track'].append({
            'frame': self.current_frame,
            'court_pos': detection['court_pos'],
            'image_pos': detection['image_pos'],
            'conf': detection['conf'],
            'box': detection['box']
        })
    
    def update_ball_tracking(self, ball_detections):
        """Update ball tracking"""
        if not ball_detections:
            return
        
        # Transform to court coordinates
        ball_positions = [b['pos'] for b in ball_detections]
        court_positions = self.transform_to_court(ball_positions)
        
        # Add valid balls on court
        for i, court_pos in enumerate(court_positions):
            if self.is_on_court(court_pos):
                self.ball_tracks.append({
                    'frame': self.current_frame,
                    'court_pos': court_pos,
                    'image_pos': ball_detections[i]['pos'],
                    'conf': ball_detections[i]['conf'],
                    'size': ball_detections[i]['size']
                })
    
    def draw_court_overlay(self, frame):
        """Draw court overlay on frame"""
        overlay = frame.copy()
        
        # Draw court boundary
        cv2.polylines(overlay, [self.court_boundary], True, (0, 255, 0), 3)
        
        # Draw net line
        cv2.line(overlay, tuple(self.net_line[0]), tuple(self.net_line[1]), (255, 255, 255), 4)
        
        # Draw center line
        cv2.line(overlay, tuple(self.center_line[0]), tuple(self.center_line[1]), (100, 100, 255), 2)
        
        # Add side labels
        left_center = np.mean(self.court_boundary[:2], axis=0).astype(int)
        right_center = np.mean(self.court_boundary[2:4], axis=0).astype(int)
        
        cv2.putText(overlay, 'LEFT SIDE', tuple(left_center - [60, -20]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        cv2.putText(overlay, 'RIGHT SIDE', tuple(right_center - [60, -20]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        return overlay
    
    def draw_player_tracking(self, frame):
        """Draw player tracking on frame"""
        colors = {
            'left_1': (255, 100, 100),   # Light red
            'left_2': (200, 50, 50),     # Dark red  
            'right_1': (100, 100, 255),  # Light blue
            'right_2': (50, 50, 200)     # Dark blue
        }
        
        for slot_name, player in self.fixed_players.items():
            if (player['position'] is not None and 
                self.current_frame - player['last_seen'] < 10):
                
                color = colors[slot_name]
                
                # Get latest tracking data
                if len(player['track']) > 0:
                    latest = player['track'][-1]
                    image_pos = latest['image_pos']
                    
                    # Draw player circle
                    pos = (int(image_pos[0]), int(image_pos[1]))
                    cv2.circle(frame, pos, 15, color, -1)
                    cv2.circle(frame, pos, 15, (255, 255, 255), 3)
                    
                    # Draw player ID
                    cv2.putText(frame, player['id'], 
                               (pos[0] - 10, pos[1] - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Draw trail (last 20 positions)
                    if len(player['track']) > 3:
                        trail_points = []
                        recent_track = list(player['track'])[-20:]
                        
                        for track_data in recent_track:
                            img_pos = track_data['image_pos']
                            trail_points.append((int(img_pos[0]), int(img_pos[1])))
                        
                        # Draw trail lines
                        for i in range(1, len(trail_points)):
                            alpha = i / len(trail_points)
                            thickness = int(1 + alpha * 3)
                            cv2.line(frame, trail_points[i-1], trail_points[i], color, thickness)
        
        return frame
    
    def draw_ball_tracking(self, frame):
        """Draw ball tracking on frame"""
        if len(self.ball_tracks) == 0:
            return frame
        
        # Get recent ball positions
        recent_balls = [b for b in self.ball_tracks if self.current_frame - b['frame'] < 30]
        
        if recent_balls:
            # Draw ball trail
            if len(recent_balls) > 1:
                trail_points = []
                for ball in recent_balls[-15:]:  # Last 15 positions
                    img_pos = ball['image_pos']
                    trail_points.append((int(img_pos[0]), int(img_pos[1])))
                
                # Draw trail
                for i in range(1, len(trail_points)):
                    alpha = i / len(trail_points)
                    thickness = int(1 + alpha * 4)
                    cv2.line(frame, trail_points[i-1], trail_points[i], (0, 255, 255), thickness)
            
            # Draw current ball position
            latest_ball = recent_balls[-1]
            ball_pos = latest_ball['image_pos']
            pos = (int(ball_pos[0]), int(ball_pos[1]))
            
            cv2.circle(frame, pos, 8, (0, 255, 255), -1)
            cv2.circle(frame, pos, 8, (0, 0, 0), 2)
            cv2.putText(frame, 'BALL', (pos[0] + 12, pos[1] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw statistics overlay"""
        # Count active players per side
        left_count = sum(1 for name in ['left_1', 'left_2'] 
                        if (self.fixed_players[name]['position'] is not None and 
                            self.current_frame - self.fixed_players[name]['last_seen'] < 10))
        
        right_count = sum(1 for name in ['right_1', 'right_2'] 
                         if (self.fixed_players[name]['position'] is not None and 
                             self.current_frame - self.fixed_players[name]['last_seen'] < 10))
        
        recent_balls = len([b for b in self.ball_tracks if self.current_frame - b['frame'] < 30])
        
        # Stats background
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Stats text
        stats = [
            f"Frame: {self.current_frame:,}/{self.total_frames:,} ({self.current_frame/self.total_frames*100:.1f}%)",
            f"Left Side: {left_count}/2 players",
            f"Right Side: {right_count}/2 players", 
            f"Ball detections: {recent_balls}"
        ]
        
        for i, text in enumerate(stats):
            cv2.putText(frame, text, (20, 35 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def cleanup_inactive_players(self):
        """Clean up players not seen for too long"""
        timeout = 60  # frames
        
        for slot_name, player in self.fixed_players.items():
            if (player['position'] is not None and 
                self.current_frame - player['last_seen'] > timeout):
                
                player['position'] = None
                print(f"⚠️  Player {player['id']} timed out")
    
    def run(self):
        """Main execution loop"""
        print("🚀 Starting Fixed Tracking San4 Analysis...")
        print("🔧 Features: Fixed 4-player tracking, court overlay, ball tracking")
        print("⏹️  Press 'q' to quit")
        
        # Window setup
        cv2.namedWindow('San4 Fixed Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('San4 Fixed Tracking', 1200, 800)
        
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
                    detections = self.process_frame(frame)
                    self.assign_players_to_fixed_slots(detections['persons'])
                    self.update_ball_tracking(detections['balls'])
                    self.cleanup_inactive_players()
                
                # Draw overlays
                frame = self.draw_court_overlay(frame)
                frame = self.draw_player_tracking(frame)
                frame = self.draw_ball_tracking(frame)
                frame = self.draw_statistics(frame)
                
                # Show frame
                cv2.imshow('San4 Fixed Tracking', frame)
                
                # Performance tracking
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                
                if len(frame_times) > 0:
                    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    
                    if self.current_frame % 100 == 0:
                        active_players = sum(1 for p in self.fixed_players.values() 
                                           if p['position'] is not None and 
                                           self.current_frame - p['last_seen'] < 10)
                        print(f"⚡ Frame {self.current_frame:,} | Active players: {active_players}/4 | FPS: {avg_fps:.1f}")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
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

if __name__ == "__main__":
    analyzer = FixedTrackingSan4()
    analyzer.run()