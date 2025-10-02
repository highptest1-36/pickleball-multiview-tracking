import cv2
import numpy as np
import json
from collections import defaultdict, deque
import time
import torch
from scipy.spatial.distance import cdist

class EnhancedTrackingSan4:
    """
    Enhanced tracking với improved far camera detection
    Fix: Better player detection ở sân xa camera
    """
    def __init__(self):
        # Load calibration
        with open('court_calibration_san4.json', 'r') as f:
            self.calibration = json.load(f)
        
        self.homography = np.array(self.calibration['homography'])
        self.court_width = self.calibration['court_width']
        self.court_length = self.calibration['court_length']
        
        # Load yellow polygon points if available
        self.yellow_polygon = None
        if 'yellow_polygon' in self.calibration:
            self.yellow_polygon = np.array(self.calibration['yellow_polygon'], dtype=np.int32)
            print(f"🟨 Using yellow polygon: {len(self.yellow_polygon)} points")
        elif 'image_points' in self.calibration:
            self.yellow_polygon = np.array(self.calibration['image_points'], dtype=np.int32)
            print(f"🟨 Using image_points: {len(self.yellow_polygon)} points")
        
        # Load net line if available
        self.net_line = None
        if 'net_line' in self.calibration:
            self.net_line = np.array(self.calibration['net_line'], dtype=np.int32)
            print(f"🎾 Using net line: 2 points")
        
        # Video setup
        self.video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Video: {self.total_frames} frames at {self.fps} FPS")
        
        # YOLO setup - Using YOLO11x for better detection
        from ultralytics import YOLO
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {device}")
        
        self.model = YOLO('yolo11x.pt')  # Upgraded to YOLO11x for better accuracy
        self.model.to(device)
        print(f"🎯 Model: YOLO11x (high accuracy)")
        
        # SIMPLE 4-player slots - không có logic phức tạp
        # Chỉ cần: detect → assign → track
        self.players = {
            1: {'active': False, 'bbox': None, 'court_pos': None, 'zone': None, 'color': (0, 0, 255), 'id': 1},      # RED
            2: {'active': False, 'bbox': None, 'court_pos': None, 'zone': None, 'color': (0, 255, 0), 'id': 2},      # GREEN  
            3: {'active': False, 'bbox': None, 'court_pos': None, 'zone': None, 'color': (255, 0, 0), 'id': 3},      # BLUE
            4: {'active': False, 'bbox': None, 'court_pos': None, 'zone': None, 'color': (0, 255, 255), 'id': 4}     # YELLOW
        }
        
        self.player_tracks = {i: deque(maxlen=150) for i in range(1, 5)}
        
        # Ball tracking
        self.ball_tracks = deque(maxlen=80)
        self.last_ball_pos = None
        
        self.current_frame = 0
        
        # Court zones - SPLIT BY WIDTH (X-axis), not length!
        # Left half vs Right half of court
        self.near_zone = (0, self.court_width * 0.5)  # Left half (0-50% of width)
        self.far_zone = (self.court_width * 0.5, self.court_width)  # Right half (50-100% of width)
        
        print(f"🏟️ COURT SPLIT BY WIDTH:")
        print(f"   Sân 1 (Left/Near) - P1, P2: 0 - {self.court_width * 0.5:.2f}m")
        print(f"   Sân 2 (Right/Far) - P3, P4: {self.court_width * 0.5:.2f}m - {self.court_width:.2f}m")
        
    def get_adaptive_confidence(self, court_pos):
        """
        Adaptive confidence threshold based on X position (left/right split)
        Left = higher threshold, Right = lower threshold (far from camera)
        """
        # Calculate distance ratio based on X-coordinate (width)
        x_ratio = court_pos[0] / self.court_width
        
        # Adaptive confidence: Left = 0.40, Right = 0.20 (increased to reduce false positives)
        # YOLO11x is more accurate, can use lower thresholds but increased for stability
        base_conf = 0.20
        max_conf = 0.40
        adaptive_conf = max_conf - (max_conf - base_conf) * x_ratio
        
        return max(0.15, min(0.45, adaptive_conf))
    
    def is_on_court(self, court_pos):
        """Check if position is on court"""
        return (0 <= court_pos[0] <= self.court_width and 
                0 <= court_pos[1] <= self.court_length)
    
    def get_player_zone(self, court_pos):
        """
        Determine player zone - split by WIDTH (X-axis), not length!
        Uses <= for near zone to handle edge case when player is exactly on split line
        """
        x = court_pos[0]  # Use X coordinate, not Y!
        
        # Left half (inclusive of split line at 3.05m)
        if x <= self.near_zone[1]:
            return 'near'
        # Right half (exclusive, starts after 3.05m)
        else:
            return 'far'
    
    def transform_to_court(self, image_points):
        """Transform image coordinates to court coordinates"""
        if len(image_points) == 0:
            return []
        
        points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        court_points = cv2.perspectiveTransform(points, self.homography)
        return court_points.reshape(-1, 2)
    
    def detect_objects_adaptive(self, frame):
        """
        Enhanced detection with focus on far camera region
        """
        height, width = frame.shape[:2]
        
        # Full frame detection
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame
            scale = 1.0
        
        # YOLO detection with LOWER confidence for far objects
        results = self.model(frame_resized, verbose=False, conf=0.25, iou=0.5)  # Increased from 0.2 to reduce false positives
        
        persons = []
        balls = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    class_name = self.model.names[int(cls)]
                    
                    if class_name == 'person':
                        x, y, w, h = box
                        x, y = x / scale, y / scale
                        w, h = w / scale, h / scale
                        
                        persons.append({
                            'center': [x, y],
                            'bbox': [x-w/2, y-h/2, w, h],
                            'conf': conf,
                            'size': w * h  # Person size for distance estimation
                        })
                        
                    elif class_name in ['sports ball', 'ball'] and conf > 0.08:  # Very low for ball detection
                        x, y, w, h = box
                        x, y = x / scale, y / scale
                        w, h = w / scale, h / scale
                        
                        ball_size = max(w, h)
                        if 5 < ball_size < 150:
                            balls.append({
                                'center': [x, y],
                                'bbox': [x-w/2, y-h/2, w, h],
                                'conf': conf,
                                'size': ball_size
                            })
        
        return persons, balls
    
    def calculate_player_features(self, detection):
        """
        Calculate player features for better re-identification
        """
        features = {
            'position': detection['center'],
            'size': detection.get('size', 0),
            'confidence': detection['conf']
        }
        return features
    
    def assign_players_enhanced(self, detected_persons):
        """
        SUPER SIMPLE LOGIC:
        1. Detect người → phân loại zone (near/far based on net line)
        2. Mỗi zone tối đa 2 người
        3. Cứ detect thấy → vẽ bounding box + tracking
        4. KHÔNG có re-assignment, KHÔNG có matching phức tạp
        """
        if not detected_persons:
            return
        
        # Transform to court coordinates
        person_centers = [p['center'] for p in detected_persons]
        court_positions = self.transform_to_court(person_centers)
        
        # Filter valid detections on court
        valid_detections = []
        for person, court_pos in zip(detected_persons, court_positions):
            if self.is_on_court(court_pos):
                person['court_pos'] = court_pos
                person['zone'] = self.get_player_zone(court_pos)
                valid_detections.append(person)
        
        if not valid_detections:
            return
        
        # Separate by zone
        near_detections = sorted([p for p in valid_detections if p['zone'] == 'near'], 
                                key=lambda x: x['conf'], reverse=True)[:2]  # Max 2
        far_detections = sorted([p for p in valid_detections if p['zone'] == 'far'], 
                               key=lambda x: x['conf'], reverse=True)[:2]  # Max 2
        
        # Simple assignment - just fill slots
        self.simple_assign(near_detections, 'near')
        self.simple_assign(far_detections, 'far')
        
        # Debug - Show clear zone assignments
        if self.current_frame % 30 == 0:
            p1_status = "✅" if self.players[1]['active'] else "❌"
            p2_status = "✅" if self.players[2]['active'] else "❌"
            p3_status = "✅" if self.players[3]['active'] else "❌"
            p4_status = "✅" if self.players[4]['active'] else "❌"
            print(f"🏟️ Frame {self.current_frame}: Sân 1: {p1_status}P1 {p2_status}P2 | Sân 2: {p3_status}P3 {p4_status}P4")
    
    def simple_assign(self, detections, zone_name):
        """
        STABLE TRACKING with zone lock:
        - Track people continuously within their zone
        - Use simple distance matching (< 1.5m)
        - Never jump to opposite side
        - No flickering, no re-linking
        """
        # Determine which player IDs belong to this zone
        if zone_name == 'near':
            zone_player_ids = [1, 2]  # P1, P2 for near side (Sân 1)
        else:  # far
            zone_player_ids = [3, 4]  # P3, P4 for far side (Sân 2)
        
        if not detections:
            # No detections - deactivate all players in this zone
            for pid in zone_player_ids:
                self.players[pid]['active'] = False
            return
        
        # Get currently active players in this zone
        active_players = [(pid, self.players[pid]) for pid in zone_player_ids 
                         if self.players[pid]['active']]
        
        # Simple distance matching for active players
        assigned_detections = set()
        assigned_players = set()
        
        if active_players:
            # Match detections to existing players by distance
            for pid, player in active_players:
                if player['court_pos'] is None:
                    continue
                    
                best_detection_idx = None
                best_distance = float('inf')
                
                for i, detection in enumerate(detections):
                    if i in assigned_detections:
                        continue
                    
                    # Calculate distance
                    dist = np.linalg.norm(
                        np.array(detection['court_pos']) - np.array(player['court_pos'])
                    )
                    
                    # Match if close enough (< 1.5m movement per frame)
                    if dist < 1.5 and dist < best_distance:
                        best_distance = dist
                        best_detection_idx = i
                
                # Assign best match
                if best_detection_idx is not None:
                    self._update_player(pid, detections[best_detection_idx])
                    assigned_detections.add(best_detection_idx)
                    assigned_players.add(pid)
                else:
                    # Lost tracking - deactivate
                    self.players[pid]['active'] = False
        
        # Assign remaining detections to inactive slots
        unassigned_detections = [d for i, d in enumerate(detections) 
                                if i not in assigned_detections]
        inactive_players = [pid for pid in zone_player_ids 
                           if pid not in assigned_players]
        
        # Sort by confidence
        unassigned_detections.sort(key=lambda x: x['conf'], reverse=True)
        
        for detection, pid in zip(unassigned_detections, inactive_players):
            self._update_player(pid, detection)
            assigned_players.add(pid)
        
        # Deactivate remaining inactive players
        for pid in zone_player_ids:
            if pid not in assigned_players:
                self.players[pid]['active'] = False
    
    def _update_player(self, pid, detection):
        """Update single player with detection data"""
        player = self.players[pid]
        
        player['active'] = True
        player['bbox'] = detection['bbox']
        player['court_pos'] = detection['court_pos']
        player['zone'] = detection['zone']
        player['conf'] = detection['conf']
        
        # Add to tracking history for trail
        self.player_tracks[pid].append({
            'frame': self.current_frame,
            'bbox': detection['bbox'],
            'center': detection['center'],
            'court_pos': detection['court_pos']
        })
    
    def update_ball_tracking(self, detected_balls):
        """Enhanced ball tracking with trajectory prediction"""
        if not detected_balls:
            return
        
        ball_centers = [b['center'] for b in detected_balls]
        court_positions = self.transform_to_court(ball_centers)
        
        valid_balls = []
        for ball, court_pos in zip(detected_balls, court_positions):
            if self.is_on_court(court_pos):
                ball['court_pos'] = court_pos
                valid_balls.append(ball)
        
        if not valid_balls:
            return
        
        best_ball = None
        
        # Trajectory-based selection
        if self.last_ball_pos is not None and len(valid_balls) > 0:
            min_dist = float('inf')
            for ball in valid_balls:
                dist = np.linalg.norm(np.array(ball['court_pos']) - np.array(self.last_ball_pos))
                # Weighted by confidence
                weighted_dist = dist / (ball['conf'] + 0.1)
                if weighted_dist < min_dist and dist < 5.0:
                    min_dist = weighted_dist
                    best_ball = ball
        
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
        else:
            # Clear last ball position if not detected for 10 frames
            if len(self.ball_tracks) > 0:
                last_frame = self.ball_tracks[-1]['frame']
                if self.current_frame - last_frame > 10:
                    self.last_ball_pos = None
    
    def draw_court_zones(self, frame):
        """Draw court boundaries and enhanced zones"""
        # Draw YELLOW POLYGON if available (the actual user-selected boundary)
        if self.yellow_polygon is not None and len(self.yellow_polygon) > 0:
            # Draw yellow polygon (user's exact selection)
            cv2.polylines(frame, [self.yellow_polygon], True, (0, 255, 255), 4)
        else:
            # Fallback: Court corners from homography
            court_corners = np.array([
                [0, 0], [self.court_width, 0],
                [self.court_width, self.court_length], [0, self.court_length]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            image_corners = cv2.perspectiveTransform(court_corners, np.linalg.inv(self.homography))
            image_corners = image_corners.reshape(-1, 2).astype(int)
            
            # Court boundary (green fallback)
            cv2.polylines(frame, [image_corners], True, (0, 255, 0), 4)
        
        # Draw NET LINE if available (WHITE thick line)
        if self.net_line is not None and len(self.net_line) == 2:
            pt1 = tuple(self.net_line[0])
            pt2 = tuple(self.net_line[1])
            cv2.line(frame, pt1, pt2, (255, 255, 255), 6)
        
        # Zone separation line removed - not needed
        
        # Zone labels
        cv2.putText(frame, 'LEFT HALF (P1,P2)', (30, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        cv2.putText(frame, 'RIGHT HALF (P3,P4)', (frame.shape[1]-280, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        
        return frame
    
    def draw_dotted_line(self, img, pt1, pt2, color, thickness, gap):
        """Draw dotted line"""
        dist = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0]*(1-r) + pt2[0]*r) + 0.5)
            y = int((pt1[1]*(1-r) + pt2[1]*r) + 0.5)
            pts.append((x, y))
        
        for i in range(0, len(pts), 2):
            if i+1 < len(pts):
                cv2.line(img, pts[i], pts[i+1], color, thickness)
    
    def draw_all_tracking(self, frame):
        """Draw enhanced tracking visualization"""
        frame = self.draw_court_zones(frame)
        
        # Draw players with enhanced info
        for player_id in range(1, 5):
            if self.players[player_id]['active']:
                bbox = self.players[player_id]['bbox']
                court_pos = self.players[player_id]['court_pos']
                color = self.players[player_id]['color']
                zone = self.players[player_id]['zone']
                
                # Bounding box
                x, y, w, h = bbox
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 3)
                
                # Center point (calculate from bbox)
                center_x = x + w/2
                center_y = y + h/2
                cv2.circle(frame, (int(center_x), int(center_y)), 10, color, -1)
                cv2.circle(frame, (int(center_x), int(center_y)), 10, (255, 255, 255), 2)
                
                # Enhanced label with zone and court position
                label = f'P{player_id} [{zone}] ({court_pos[0]:.1f},{court_pos[1]:.1f})'
                cv2.putText(frame, label, 
                           (int(center_x)+15, int(center_y)-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Trail
                if len(self.player_tracks[player_id]) > 3:
                    trail_points = []
                    for track in list(self.player_tracks[player_id])[-30:]:
                        trail_points.append(track['center'])  # Use 'center' from simple_assign
                    
                    for i in range(1, len(trail_points)):
                        alpha = i / len(trail_points)
                        trail_color = tuple(int(c * alpha * 0.8) for c in color)
                        pt1 = (int(trail_points[i-1][0]), int(trail_points[i-1][1]))
                        pt2 = (int(trail_points[i][0]), int(trail_points[i][1]))
                        cv2.line(frame, pt1, pt2, trail_color, 3)
        
        # Draw ball - ONLY if detected in recent frames (within 5 frames)
        if len(self.ball_tracks) > 0:
            current_ball = self.ball_tracks[-1]
            frame_gap = self.current_frame - current_ball['frame']
            
            # ONLY show if detected within last 5 frames
            if frame_gap <= 5:
                center = current_ball['image_pos']
                bbox = current_ball['bbox']
                
                x, y, w, h = bbox
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 255), 2)
                
                cv2.circle(frame, (int(center[0]), int(center[1])), 8, (0, 255, 255), -1)
                cv2.circle(frame, (int(center[0]), int(center[1])), 8, (0, 0, 0), 2)
                
                cv2.putText(frame, f'BALL ({current_ball["conf"]:.2f})', 
                           (int(center[0])+12, int(center[1])-12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Ball trail - only recent tracks
                if len(self.ball_tracks) > 3:
                    ball_trail = []
                    for track in list(self.ball_tracks)[-15:]:
                        if self.current_frame - track['frame'] <= 15:
                            ball_trail.append(track['image_pos'])
                    
                    for i in range(1, len(ball_trail)):
                        pt1 = (int(ball_trail[i-1][0]), int(ball_trail[i-1][1]))
                        pt2 = (int(ball_trail[i][0]), int(ball_trail[i][1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        return frame
    
    def add_stats_panel(self, frame):
        """Enhanced statistics panel"""
        panel_height = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        near_count = sum(1 for pid in [1, 2] if self.players[pid]['active'])
        far_count = sum(1 for pid in [3, 4] if self.players[pid]['active'])
        ball_recent = len([b for b in self.ball_tracks if self.current_frame - b['frame'] < 30])
        
        stats = [
            f"Frame: {self.current_frame:,} / {self.total_frames:,}",
            f"Progress: {self.current_frame/self.total_frames*100:.1f}%",
            f"",
            f"ENHANCED PLAYER TRACKING:",
            f"  Near Camera: {near_count}/2 players",
            f"  Far Camera: {far_count}/2 players",
            f"",
            f"Ball: {ball_recent} recent detections",
            f"",
            f"Features: Adaptive confidence, Enhanced re-ID"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 35 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main execution with enhanced tracking"""
        print("🚀 Starting ENHANCED San4 Analysis...")
        print("✨ Features:")
        print("   - Adaptive confidence thresholds")
        print("   - Enhanced far camera detection")
        print("   - Better player re-identification")
        print("   - Improved ball tracking")
        print("⏹️  Press 'q' to quit")
        
        cv2.namedWindow('Enhanced San4 Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Enhanced San4 Tracking', 1400, 900)
        
        frame_times = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("🏁 Video finished")
                    break
                
                self.current_frame += 1
                
                # Enhanced detection
                detected_persons, detected_balls = self.detect_objects_adaptive(frame)
                
                # Enhanced tracking
                self.assign_players_enhanced(detected_persons)
                self.update_ball_tracking(detected_balls)
                
                # Draw everything
                frame = self.draw_all_tracking(frame)
                frame = self.add_stats_panel(frame)
                
                cv2.imshow('Enhanced San4 Tracking', frame)
                
                # Performance
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                
                if self.current_frame % 120 == 0:
                    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    active_count = sum(1 for p in self.players.values() if p['active'])
                    near_active = sum(1 for pid in [1,2] if self.players[pid]['active'])
                    far_active = sum(1 for pid in [3,4] if self.players[pid]['active'])
                    print(f"⚡ Frame {self.current_frame:,} | Near: {near_active}/2 | Far: {far_active}/2 | Ball: {len(self.ball_tracks)} | FPS: {avg_fps:.1f}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
            print("🎉 Enhanced analysis complete!")

if __name__ == "__main__":
    analyzer = EnhancedTrackingSan4()
    analyzer.run()