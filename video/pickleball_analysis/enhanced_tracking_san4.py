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
        
        # Enhanced 4-player system with confidence tracking
        self.players = {
            1: {'active': False, 'position': None, 'zone': 'near', 'confidence': 0.0, 'color': (0, 0, 255), 'history': deque(maxlen=30)},
            2: {'active': False, 'position': None, 'zone': 'near', 'confidence': 0.0, 'color': (0, 100, 255), 'history': deque(maxlen=30)},
            3: {'active': False, 'position': None, 'zone': 'far', 'confidence': 0.0, 'color': (255, 0, 0), 'history': deque(maxlen=30)},
            4: {'active': False, 'position': None, 'zone': 'far', 'confidence': 0.0, 'color': (255, 0, 100), 'history': deque(maxlen=30)}
        }
        
        self.player_tracks = {i: deque(maxlen=150) for i in range(1, 5)}
        
        # Ball tracking
        self.ball_tracks = deque(maxlen=80)
        self.last_ball_pos = None
        
        self.current_frame = 0
        
        # Court zones - based on camera perspective
        self.near_zone = (0, self.court_length * 0.55)  # Near camera (0-55%)
        self.far_zone = (self.court_length * 0.45, self.court_length)  # Far camera (45-100%)
        
        print(f"🏟️ ENHANCED ZONES:")
        print(f"   Near Camera: 0 - {self.court_length * 0.55:.1f}m")
        print(f"   Far Camera: {self.court_length * 0.45:.1f}m - {self.court_length:.1f}m")
        
    def get_adaptive_confidence(self, court_pos):
        """
        Adaptive confidence threshold based on distance from camera
        Far = lower threshold, Near = higher threshold
        """
        # Calculate distance from camera (y-coordinate)
        distance_ratio = court_pos[1] / self.court_length
        
        # Adaptive confidence: Far = 0.25, Near = 0.45
        base_conf = 0.25
        max_conf = 0.45
        adaptive_conf = base_conf + (max_conf - base_conf) * (1 - distance_ratio)
        
        return max(0.2, min(0.5, adaptive_conf))
    
    def is_on_court(self, court_pos):
        """Check if position is on court"""
        return (0 <= court_pos[0] <= self.court_width and 
                0 <= court_pos[1] <= self.court_length)
    
    def get_player_zone(self, court_pos):
        """Determine player zone with overlap buffer"""
        y = court_pos[1]
        
        # Near zone with buffer
        if y < self.near_zone[1]:
            return 'near'
        # Far zone
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
        results = self.model(frame_resized, verbose=False, conf=0.2, iou=0.5)  # Lowered from 0.3
        
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
                        
                    elif class_name in ['sports ball', 'ball'] and conf > 0.12:  # Even lower for ball
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
        Enhanced player assignment with adaptive confidence and better re-ID
        """
        if not detected_persons:
            # Decay confidence for inactive players
            for pid in range(1, 5):
                if self.players[pid]['active']:
                    self.players[pid]['confidence'] *= 0.95
                    if self.players[pid]['confidence'] < 0.1:
                        self.players[pid]['active'] = False
                        self.players[pid]['position'] = None
            return
        
        # Transform to court coordinates
        person_centers = [p['center'] for p in detected_persons]
        court_positions = self.transform_to_court(person_centers)
        
        # Filter and enhance detections
        valid_detections = []
        for person, court_pos in zip(detected_persons, court_positions):
            if self.is_on_court(court_pos):
                person['court_pos'] = court_pos
                person['zone'] = self.get_player_zone(court_pos)
                person['adaptive_conf'] = self.get_adaptive_confidence(court_pos)
                person['features'] = self.calculate_player_features(person)
                
                # Accept if meets adaptive threshold
                if person['conf'] >= person['adaptive_conf']:
                    valid_detections.append(person)
        
        if not valid_detections:
            return
        
        # Separate by zone
        near_detections = [p for p in valid_detections if p['zone'] == 'near']
        far_detections = [p for p in valid_detections if p['zone'] == 'far']
        
        # Assign near camera players (P1, P2)
        self.assign_zone_players(near_detections, [1, 2])
        
        # Assign far camera players (P3, P4) with enhanced matching
        self.assign_zone_players(far_detections, [3, 4])
    
    def assign_zone_players(self, detections, player_ids):
        """
        Assign detections to zone players with enhanced matching
        """
        if not detections:
            return
        
        # Get currently active players in this zone
        active_players = {pid: self.players[pid] for pid in player_ids if self.players[pid]['active']}
        
        # If no active players, assign first detections
        if not active_players:
            for i, detection in enumerate(detections[:2]):
                pid = player_ids[i]
                self.assign_player(pid, detection)
                print(f"✅ Player {pid} activated (zone: {detection['zone']})")
            return
        
        # Enhanced matching with multiple factors
        if len(active_players) > 0:
            detection_positions = np.array([d['court_pos'] for d in detections])
            active_ids = list(active_players.keys())
            active_positions = np.array([self.players[pid]['position']['court_pos'] for pid in active_ids])
            
            # Calculate distance matrix
            distances = cdist(detection_positions, active_positions)
            
            # Enhanced scoring: distance + confidence + size consistency
            scores = distances.copy()
            for i, detection in enumerate(detections):
                for j, pid in enumerate(active_ids):
                    # Distance penalty
                    dist_penalty = distances[i, j]
                    
                    # Confidence bonus
                    conf_bonus = -detection['conf'] * 2.0
                    
                    # Size consistency bonus (if available)
                    if len(self.players[pid]['history']) > 0:
                        avg_size = np.mean([h.get('size', 0) for h in self.players[pid]['history']])
                        size_diff = abs(detection.get('size', 0) - avg_size) / max(avg_size, 1)
                        size_penalty = size_diff * 0.5
                    else:
                        size_penalty = 0
                    
                    scores[i, j] = dist_penalty + conf_bonus + size_penalty
            
            # Greedy assignment
            used_detections = set()
            used_players = set()
            
            assignments = []
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    assignments.append((scores[i, j], i, j))
            
            assignments.sort()
            
            # Assign with relaxed threshold for far camera
            max_distance = 2.5 if detections[0]['zone'] == 'far' else 1.8
            
            for score, det_idx, player_idx in assignments:
                if det_idx not in used_detections and player_idx not in used_players:
                    if distances[det_idx, player_idx] < max_distance:
                        pid = active_ids[player_idx]
                        self.assign_player(pid, detections[det_idx])
                        used_detections.add(det_idx)
                        used_players.add(player_idx)
            
            # Assign remaining detections to inactive slots
            available_slots = [pid for pid in player_ids if not self.players[pid]['active']]
            remaining_detections = [detections[i] for i in range(len(detections)) if i not in used_detections]
            
            for i, detection in enumerate(remaining_detections[:len(available_slots)]):
                pid = available_slots[i]
                self.assign_player(pid, detection)
                print(f"🔄 Player {pid} re-activated (zone: {detection['zone']})")
    
    def assign_player(self, player_id, detection):
        """Assign detection to player with confidence tracking"""
        self.players[player_id]['active'] = True
        self.players[player_id]['position'] = detection
        self.players[player_id]['zone'] = detection['zone']
        self.players[player_id]['confidence'] = detection['conf']
        
        # Add to history for feature averaging
        self.players[player_id]['history'].append({
            'court_pos': detection['court_pos'],
            'size': detection.get('size', 0),
            'conf': detection['conf']
        })
        
        # Add to tracking history
        self.player_tracks[player_id].append({
            'frame': self.current_frame,
            'court_pos': detection['court_pos'],
            'image_pos': detection['center'],
            'bbox': detection['bbox'],
            'conf': detection['conf'],
            'zone': detection['zone']
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
    
    def draw_court_zones(self, frame):
        """Draw court boundaries and enhanced zones"""
        # Court corners
        court_corners = np.array([
            [0, 0], [self.court_width, 0],
            [self.court_width, self.court_length], [0, self.court_length]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        image_corners = cv2.perspectiveTransform(court_corners, np.linalg.inv(self.homography))
        image_corners = image_corners.reshape(-1, 2).astype(int)
        
        # Court boundary (green)
        cv2.polylines(frame, [image_corners], True, (0, 255, 0), 4)
        
        # Net line (horizontal - white)
        net_start = np.array([[0, self.court_length/2]], dtype=np.float32).reshape(-1, 1, 2)
        net_end = np.array([[self.court_width, self.court_length/2]], dtype=np.float32).reshape(-1, 1, 2)
        
        net_start_img = cv2.perspectiveTransform(net_start, np.linalg.inv(self.homography))[0][0].astype(int)
        net_end_img = cv2.perspectiveTransform(net_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        cv2.line(frame, tuple(net_start_img), tuple(net_end_img), (255, 255, 255), 6)
        
        # Zone separation line (dotted)
        zone_y = (self.near_zone[1] + self.far_zone[0]) / 2
        zone_start = np.array([[0, zone_y]], dtype=np.float32).reshape(-1, 1, 2)
        zone_end = np.array([[self.court_width, zone_y]], dtype=np.float32).reshape(-1, 1, 2)
        
        zone_start_img = cv2.perspectiveTransform(zone_start, np.linalg.inv(self.homography))[0][0].astype(int)
        zone_end_img = cv2.perspectiveTransform(zone_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        # Dotted line
        self.draw_dotted_line(frame, tuple(zone_start_img), tuple(zone_end_img), (255, 255, 0), 2, 20)
        
        # Zone labels
        cv2.putText(frame, 'NEAR CAMERA (P1,P2)', (30, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        cv2.putText(frame, 'FAR CAMERA (P3,P4)', (frame.shape[1]-280, 40), 
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
                player = self.players[player_id]['position']
                color = self.players[player_id]['color']
                conf = self.players[player_id]['confidence']
                zone = self.players[player_id]['zone']
                
                # Bounding box
                x, y, w, h = player['bbox']
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 3)
                
                # Center point
                center = player['center']
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, color, -1)
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, (255, 255, 255), 2)
                
                # Enhanced label with confidence and zone
                label = f'P{player_id} {conf:.2f} [{zone}]'
                cv2.putText(frame, label, 
                           (int(center[0])+15, int(center[1])-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Trail
                if len(self.player_tracks[player_id]) > 3:
                    trail_points = []
                    for track in list(self.player_tracks[player_id])[-30:]:
                        trail_points.append(track['image_pos'])
                    
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
            
            x, y, w, h = bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 255), 2)
            
            cv2.circle(frame, (int(center[0]), int(center[1])), 8, (0, 255, 255), -1)
            cv2.circle(frame, (int(center[0]), int(center[1])), 8, (0, 0, 0), 2)
            
            cv2.putText(frame, f'BALL ({current_ball["conf"]:.2f})', 
                       (int(center[0])+12, int(center[1])-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Ball trail
            if len(self.ball_tracks) > 3:
                ball_trail = []
                for track in list(self.ball_tracks)[-25:]:
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