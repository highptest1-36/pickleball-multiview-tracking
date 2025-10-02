import cv2
import numpy as np
import json
from collections import defaultdict, deque
import time
import torch
from scipy.spatial.distance import cdist

class FixedTrackingSan4V2:
    def __init__(self):
        # Load calibration
        with open('court_calibration_san4.json', 'r') as f:
            self.calibration = json.load(f)
        
        self.homography = np.array(self.calibration['homography'])
        self.court_width = self.calibration['court_width']
        self.court_length = self.calibration['court_length']
        
        print(f"🏟️ Court: {self.court_width:.1f}m wide × {self.court_length:.1f}m long")
        
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
        
        # CORRECTED 4-PLAYER SYSTEM
        # Sân pickleball: lưới nằm ngang (theo chiều rộng)
        # Chia theo chiều dài (length): gần camera vs xa camera
        self.near_side = (0, self.court_length/2)        # Gần camera (y = 0 to length/2)
        self.far_side = (self.court_length/2, self.court_length)  # Xa camera (y = length/2 to length)
        
        self.players = {
            1: {'active': False, 'position': None, 'side': 'near', 'color': (0, 0, 255)},     # Red - Near
            2: {'active': False, 'position': None, 'side': 'near', 'color': (0, 100, 255)},  # Orange - Near
            3: {'active': False, 'position': None, 'side': 'far', 'color': (255, 0, 0)},     # Blue - Far
            4: {'active': False, 'position': None, 'side': 'far', 'color': (255, 0, 100)}    # Purple - Far
        }
        
        self.player_tracks = {i: deque(maxlen=100) for i in range(1, 5)}
        
        # Ball tracking - can move freely
        self.ball_tracks = deque(maxlen=60)
        self.last_ball_pos = None
        
        self.current_frame = 0
        
        # Net position - HORIZONTAL (across width)
        self.net_y = self.court_length / 2  # Net at middle of length
        
        print(f"🏟️ CORRECTED ZONES:")
        print(f"   Near camera [y: 0-{self.net_y:.1f}m] = P1, P2")
        print(f"   Far camera [y: {self.net_y:.1f}-{self.court_length:.1f}m] = P3, P4")
        print(f"   Net runs across width at y={self.net_y:.1f}m")
        
    def is_on_court(self, court_pos):
        """Check if position is on court"""
        return (0 <= court_pos[0] <= self.court_width and 
                0 <= court_pos[1] <= self.court_length)
    
    def get_court_side(self, court_pos):
        """Get which side of court based on distance from camera"""
        # Near camera = y < net_y, Far camera = y > net_y
        return 'near' if court_pos[1] < self.net_y else 'far'
    
    def transform_to_court(self, image_points):
        """Transform image coordinates to court coordinates"""
        if len(image_points) == 0:
            return []
        
        points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        court_points = cv2.perspectiveTransform(points, self.homography)
        return court_points.reshape(-1, 2)
    
    def detect_objects(self, frame):
        """Detect persons and balls"""
        # Resize for speed
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame
            scale = 1.0
        
        # YOLO detection
        results = self.model(frame_resized, verbose=False, conf=0.25)
        
        persons = []
        balls = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    class_name = self.model.names[int(cls)]
                    
                    if class_name == 'person' and conf > 0.4:
                        x, y, w, h = box
                        x, y = x / scale, y / scale
                        w, h = w / scale, h / scale
                        
                        persons.append({
                            'center': [x, y],
                            'bbox': [x-w/2, y-h/2, w, h],
                            'conf': conf
                        })
                        
                    elif class_name in ['sports ball', 'ball'] and conf > 0.15:
                        x, y, w, h = box
                        x, y = x / scale, y / scale
                        w, h = w / scale, h / scale
                        
                        ball_size = max(w, h)
                        if 5 < ball_size < 120:
                            balls.append({
                                'center': [x, y],
                                'bbox': [x-w/2, y-h/2, w, h],
                                'conf': conf,
                                'size': ball_size
                            })
        
        return persons, balls
    
    def assign_players_by_camera_distance(self, detected_persons):
        """Assign players based on distance from camera (near vs far)"""
        if not detected_persons:
            return
        
        # Transform to court coordinates
        person_centers = [p['center'] for p in detected_persons]
        court_positions = self.transform_to_court(person_centers)
        
        # Filter only on-court detections
        valid_detections = []
        for person, court_pos in zip(detected_persons, court_positions):
            if self.is_on_court(court_pos):
                person['court_pos'] = court_pos
                person['side'] = self.get_court_side(court_pos)
                valid_detections.append(person)
        
        if not valid_detections:
            return
        
        # Separate by camera distance
        near_detections = [p for p in valid_detections if p['side'] == 'near']
        far_detections = [p for p in valid_detections if p['side'] == 'far']
        
        print(f"🔍 Frame {self.current_frame}: Near={len(near_detections)}, Far={len(far_detections)}")
        
        # Assign near side (P1, P2)
        self.assign_side_players(near_detections, [1, 2])
        
        # Assign far side (P3, P4)  
        self.assign_side_players(far_detections, [3, 4])
    
    def assign_side_players(self, detections, player_ids):
        """Assign detections to specific side players"""
        if not detections:
            # Mark players as inactive if no detections for too long
            for pid in player_ids:
                if self.players[pid]['active']:
                    if (len(self.player_tracks[pid]) == 0 or 
                        self.current_frame - self.player_tracks[pid][-1]['frame'] > 90):
                        self.players[pid]['active'] = False
                        self.players[pid]['position'] = None
                        print(f"❌ Player {pid} inactive (no detection)")
            return
        
        # Get currently active players on this side
        active_players = {pid: self.players[pid] for pid in player_ids if self.players[pid]['active']}
        
        # If no active players, assign first detections
        if not active_players:
            for i, detection in enumerate(detections[:2]):  # Max 2 per side
                pid = player_ids[i]
                self.assign_player(pid, detection)
                print(f"✅ Player {pid} activated on {detection['side']} side")
            return
        
        # Match detections to existing active players
        if len(active_players) > 0:
            detection_positions = np.array([d['court_pos'] for d in detections])
            active_ids = list(active_players.keys())
            active_positions = np.array([self.players[pid]['position']['court_pos'] for pid in active_ids])
            
            # Calculate distances
            distances = cdist(detection_positions, active_positions)
            
            # Greedy assignment based on closest distance
            used_detections = set()
            used_players = set()
            
            # Create assignment pairs sorted by distance
            assignments = []
            for i in range(distances.shape[0]):
                for j in range(distances.shape[1]):
                    assignments.append((distances[i,j], i, j))
            
            assignments.sort()  # Sort by distance
            
            # Assign closest matches first
            for dist, det_idx, player_idx in assignments:
                if det_idx not in used_detections and player_idx not in used_players:
                    if dist < 2.0:  # Max movement threshold
                        pid = active_ids[player_idx]
                        self.assign_player(pid, detections[det_idx])
                        used_detections.add(det_idx)
                        used_players.add(player_idx)
            
            # Handle unassigned detections - assign to inactive slots
            remaining_detections = [detections[i] for i in range(len(detections)) if i not in used_detections]
            inactive_slots = [pid for pid in player_ids if not self.players[pid]['active']]
            
            for i, detection in enumerate(remaining_detections[:len(inactive_slots)]):
                pid = inactive_slots[i]
                self.assign_player(pid, detection)
                print(f"🔄 Player {pid} re-activated on {detection['side']} side")
    
    def assign_player(self, player_id, detection):
        """Assign detection to specific player"""
        self.players[player_id]['active'] = True
        self.players[player_id]['position'] = detection
        
        # Add to tracking history
        self.player_tracks[player_id].append({
            'frame': self.current_frame,
            'court_pos': detection['court_pos'],
            'image_pos': detection['center'],
            'bbox': detection['bbox'],
            'conf': detection['conf']
        })
    
    def update_ball_tracking(self, detected_balls):
        """Update ball tracking"""
        if not detected_balls:
            return
        
        # Transform to court coordinates
        ball_centers = [b['center'] for b in detected_balls]
        court_positions = self.transform_to_court(ball_centers)
        
        # Filter on-court balls
        valid_balls = []
        for ball, court_pos in zip(detected_balls, court_positions):
            if self.is_on_court(court_pos):
                ball['court_pos'] = court_pos
                valid_balls.append(ball)
        
        if not valid_balls:
            return
        
        # Choose best ball candidate
        best_ball = None
        
        if self.last_ball_pos is not None and len(valid_balls) > 0:
            # Find ball closest to last known position
            min_dist = float('inf')
            for ball in valid_balls:
                dist = np.linalg.norm(np.array(ball['court_pos']) - np.array(self.last_ball_pos))
                if dist < min_dist and dist < 4.0:  # Max 4m movement
                    min_dist = dist
                    best_ball = ball
        
        # If no good continuity, pick highest confidence
        if best_ball is None and valid_balls:
            best_ball = max(valid_balls, key=lambda x: x['conf'])
        
        if best_ball:
            self.ball_tracks.append({
                'frame': self.current_frame,
                'court_pos': best_ball['court_pos'],
                'image_pos': best_ball['center'],
                'bbox': best_ball['bbox'],
                'conf': best_ball['conf']
            })
            self.last_ball_pos = best_ball['court_pos']
    
    def draw_corrected_court_zones(self, frame):
        """Draw court boundaries with CORRECT orientation"""
        # Court corners
        court_corners = np.array([
            [0, 0], [self.court_width, 0],
            [self.court_width, self.court_length], [0, self.court_length]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform to image coordinates
        image_corners = cv2.perspectiveTransform(court_corners, np.linalg.inv(self.homography))
        image_corners = image_corners.reshape(-1, 2).astype(int)
        
        # Draw court boundary (green)
        cv2.polylines(frame, [image_corners], True, (0, 255, 0), 4)
        
        # Draw net line HORIZONTALLY (across width at middle of length)
        net_start = np.array([[0, self.net_y]], dtype=np.float32).reshape(-1, 1, 2)
        net_end = np.array([[self.court_width, self.net_y]], dtype=np.float32).reshape(-1, 1, 2)
        
        net_start_img = cv2.perspectiveTransform(net_start, np.linalg.inv(self.homography))[0][0].astype(int)
        net_end_img = cv2.perspectiveTransform(net_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        # Draw HORIZONTAL net (white, thick)
        cv2.line(frame, tuple(net_start_img), tuple(net_end_img), (255, 255, 255), 8)
        
        # Draw service lines (non-volley zones)
        nvz_distance = 2.13  # 7 feet
        
        # Near side non-volley line
        nvz_near_start = np.array([[0, self.net_y - nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        nvz_near_end = np.array([[self.court_width, self.net_y - nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        
        nvz_near_start_img = cv2.perspectiveTransform(nvz_near_start, np.linalg.inv(self.homography))[0][0].astype(int)
        nvz_near_end_img = cv2.perspectiveTransform(nvz_near_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        # Far side non-volley line  
        nvz_far_start = np.array([[0, self.net_y + nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        nvz_far_end = np.array([[self.court_width, self.net_y + nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        
        nvz_far_start_img = cv2.perspectiveTransform(nvz_far_start, np.linalg.inv(self.homography))[0][0].astype(int)
        nvz_far_end_img = cv2.perspectiveTransform(nvz_far_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        # Draw non-volley zone lines (blue)
        cv2.line(frame, tuple(nvz_near_start_img), tuple(nvz_near_end_img), (255, 100, 0), 3)
        cv2.line(frame, tuple(nvz_far_start_img), tuple(nvz_far_end_img), (255, 100, 0), 3)
        
        # Draw center service line (vertical)
        center_service_start = np.array([[self.court_width/2, self.net_y - nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        center_service_end = np.array([[self.court_width/2, self.net_y + nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        
        center_service_start_img = cv2.perspectiveTransform(center_service_start, np.linalg.inv(self.homography))[0][0].astype(int)
        center_service_end_img = cv2.perspectiveTransform(center_service_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        cv2.line(frame, tuple(center_service_start_img), tuple(center_service_end_img), (0, 255, 255), 3)
        
        # Zone labels with CORRECTED positions
        cv2.putText(frame, 'NEAR CAMERA (P1,P2)', (30, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        cv2.putText(frame, 'FAR CAMERA (P3,P4)', (30, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        
        return frame
    
    def draw_all_tracking(self, frame):
        """Draw all tracking on frame"""
        # Draw corrected court zones
        frame = self.draw_corrected_court_zones(frame)
        
        # Draw players
        for player_id in range(1, 5):
            if self.players[player_id]['active']:
                player = self.players[player_id]['position']
                color = self.players[player_id]['color']
                
                # Bounding box
                x, y, w, h = player['bbox']
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 3)
                
                # Center point
                center = player['center']
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, color, -1)
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, (255, 255, 255), 2)
                
                # Player ID with side info
                side_label = 'NEAR' if self.players[player_id]['side'] == 'near' else 'FAR'
                label = f'P{player_id}-{side_label} ({player["conf"]:.2f})'
                cv2.putText(frame, label, 
                           (int(center[0])+15, int(center[1])-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Trail
                if len(self.player_tracks[player_id]) > 3:
                    trail_points = []
                    for track in list(self.player_tracks[player_id])[-25:]:
                        trail_points.append(track['image_pos'])
                    
                    # Draw trail with fading effect
                    for i in range(1, len(trail_points)):
                        alpha = i / len(trail_points)
                        trail_color = tuple(int(c * alpha * 0.8) for c in color)
                        pt1 = (int(trail_points[i-1][0]), int(trail_points[i-1][1]))
                        pt2 = (int(trail_points[i][0]), int(trail_points[i][1]))
                        cv2.line(frame, pt1, pt2, trail_color, 3)
        
        # Draw ball
        if len(self.ball_tracks) > 0:
            current_ball = self.ball_tracks[-1]
            center = current_ball['image_pos']
            bbox = current_ball['bbox']
            
            # Ball bounding box (yellow)
            x, y, w, h = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 255), 2)
            
            # Ball center
            cv2.circle(frame, (int(center[0]), int(center[1])), 8, (0, 255, 255), -1)
            cv2.circle(frame, (int(center[0]), int(center[1])), 8, (0, 0, 0), 2)
            
            # Ball label
            cv2.putText(frame, f'BALL ({current_ball["conf"]:.2f})', 
                       (int(center[0])+12, int(center[1])-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Ball trail
            if len(self.ball_tracks) > 3:
                ball_trail = []
                for track in list(self.ball_tracks)[-20:]:
                    ball_trail.append(track['image_pos'])
                
                for i in range(1, len(ball_trail)):
                    pt1 = (int(ball_trail[i-1][0]), int(ball_trail[i-1][1]))
                    pt2 = (int(ball_trail[i][0]), int(ball_trail[i][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        return frame
    
    def add_stats_panel(self, frame):
        """Add statistics panel"""
        # Background panel
        panel_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Statistics
        active_near = sum(1 for pid in [1, 2] if self.players[pid]['active'])
        active_far = sum(1 for pid in [3, 4] if self.players[pid]['active'])
        ball_recent = len([b for b in self.ball_tracks if self.current_frame - b['frame'] < 30])
        
        stats = [
            f"🎬 Frame: {self.current_frame:,} / {self.total_frames:,}",
            f"📊 Progress: {self.current_frame/self.total_frames*100:.1f}%",
            f"⏱️  Time: {self.current_frame/self.fps:.1f}s",
            "",
            f"👥 CORRECTED PLAYER TRACKING:",
            f"   📹 Near Camera: {active_near}/2 players (P1,P2)",
            f"   🎥 Far Camera: {active_far}/2 players (P3,P4)", 
            f"   🎾 Ball: {ball_recent} recent detections",
            f"   🏟️ Net: HORIZONTAL across court width"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 35 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main execution with corrected tracking"""
        print("🚀 Starting CORRECTED San4 Analysis...")
        print("🔧 FIXED: Net now runs HORIZONTALLY (across width)")
        print("👥 Near camera = P1,P2 | Far camera = P3,P4")
        print("🎾 Ball = Free movement across court")
        print("⏹️  Press 'q' to quit")
        
        # Setup window
        cv2.namedWindow('CORRECTED San4 Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CORRECTED San4 Tracking', 1400, 900)
        
        frame_times = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("🏁 Video finished")
                    break
                
                self.current_frame += 1
                
                # Detect objects
                detected_persons, detected_balls = self.detect_objects(frame)
                
                # Update tracking with corrected logic
                self.assign_players_by_camera_distance(detected_persons)
                self.update_ball_tracking(detected_balls)
                
                # Draw everything
                frame = self.draw_all_tracking(frame)
                frame = self.add_stats_panel(frame)
                
                # Display
                cv2.imshow('CORRECTED San4 Tracking', frame)
                
                # Performance monitoring
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                
                # Progress updates
                if self.current_frame % 120 == 0:
                    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    active_count = sum(1 for p in self.players.values() if p['active'])
                    ball_count = len(self.ball_tracks)
                    near_count = sum(1 for pid in [1, 2] if self.players[pid]['active'])
                    far_count = sum(1 for pid in [3, 4] if self.players[pid]['active'])
                    print(f"⚡ Frame {self.current_frame:,} | Near: {near_count}/2, Far: {far_count}/2 | Ball: {ball_count} | FPS: {avg_fps:.1f}")
                
                # Exit check
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️  User quit")
                    break
                
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("🎉 Analysis finished!")

if __name__ == "__main__":
    analyzer = FixedTrackingSan4V2()
    analyzer.run()