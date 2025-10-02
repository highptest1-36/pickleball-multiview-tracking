import cv2
import numpy as np
import json
from collections import defaultdict, deque
import time
import torch
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter

class BallTracker:
    """
    Advanced ball tracker with Kalman Filter and trajectory prediction
    """
    def __init__(self):
        # Kalman Filter setup for 2D ball tracking
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State: [x, y, vx, vy]
        self.kf.x = np.array([0., 0., 0., 0.])
        
        # State transition matrix (constant velocity model)
        dt = 1/30.0  # Assuming 30 FPS
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 0.5
        
        # Process noise
        self.kf.Q *= 0.01
        
        # Initial covariance
        self.kf.P *= 100
        
        self.initialized = False
        self.missing_count = 0
        self.max_missing = 10  # Max frames without detection before reset
        
    def update(self, detection):
        """Update Kalman Filter with new detection"""
        if not self.initialized:
            self.kf.x[:2] = detection
            self.initialized = True
            self.missing_count = 0
        else:
            self.kf.predict()
            self.kf.update(detection)
            self.missing_count = 0
    
    def predict(self):
        """Predict next position"""
        if not self.initialized:
            return None
        
        self.kf.predict()
        self.missing_count += 1
        
        if self.missing_count > self.max_missing:
            self.reset()
            return None
        
        return self.kf.x[:2]
    
    def get_position(self):
        """Get current estimated position"""
        if not self.initialized:
            return None
        return self.kf.x[:2]
    
    def get_velocity(self):
        """Get current estimated velocity"""
        if not self.initialized:
            return None
        return self.kf.x[2:]
    
    def reset(self):
        """Reset tracker"""
        self.initialized = False
        self.missing_count = 0
        self.kf.x = np.array([0., 0., 0., 0.])
        self.kf.P *= 100

class AdvancedTrackingSan4:
    """
    Advanced tracking với Enhanced Ball Tracking using Kalman Filter
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
        
        # Enhanced 4-player system
        self.players = {
            1: {'active': False, 'position': None, 'zone': 'near', 'confidence': 0.0, 'color': (0, 0, 255), 'history': deque(maxlen=30)},
            2: {'active': False, 'position': None, 'zone': 'near', 'confidence': 0.0, 'color': (0, 100, 255), 'history': deque(maxlen=30)},
            3: {'active': False, 'position': None, 'zone': 'far', 'confidence': 0.0, 'color': (255, 0, 0), 'history': deque(maxlen=30)},
            4: {'active': False, 'position': None, 'zone': 'far', 'confidence': 0.0, 'color': (255, 0, 100), 'history': deque(maxlen=30)}
        }
        
        self.player_tracks = {i: deque(maxlen=150) for i in range(1, 5)}
        
        # ADVANCED BALL TRACKING with Kalman Filter
        self.ball_tracker = BallTracker()
        self.ball_tracks = deque(maxlen=100)
        self.ball_detections = deque(maxlen=50)  # Raw detections
        self.last_ball_detection = None
        self.ball_detected_count = 0
        
        self.current_frame = 0
        
        # Court zones
        self.near_zone = (0, self.court_length * 0.55)
        self.far_zone = (self.court_length * 0.45, self.court_length)
        
        print(f"🏟️ ADVANCED BALL TRACKING:")
        print(f"   - Kalman Filter prediction")
        print(f"   - Trajectory smoothing")
        print(f"   - Gap filling (up to {self.ball_tracker.max_missing} frames)")
        
    def get_adaptive_confidence(self, court_pos):
        """Adaptive confidence threshold"""
        distance_ratio = court_pos[1] / self.court_length
        base_conf = 0.25
        max_conf = 0.45
        adaptive_conf = base_conf + (max_conf - base_conf) * (1 - distance_ratio)
        return max(0.2, min(0.5, adaptive_conf))
    
    def is_on_court(self, court_pos):
        """Check if position is on court"""
        return (0 <= court_pos[0] <= self.court_width and 
                0 <= court_pos[1] <= self.court_length)
    
    def get_player_zone(self, court_pos):
        """Determine player zone"""
        y = court_pos[1]
        if y < self.near_zone[1]:
            return 'near'
        else:
            return 'far'
    
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
        image_points = cv2.perspectiveTransform(points, np.linalg.inv(self.homography))
        return image_points.reshape(-1, 2)
    
    def detect_objects_adaptive(self, frame):
        """Enhanced detection"""
        height, width = frame.shape[:2]
        
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame
            scale = 1.0
        
        # YOLO detection with lower confidence for ball
        results = self.model(frame_resized, verbose=False, conf=0.2, iou=0.5)
        
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
                            'size': w * h
                        })
                        
                    elif class_name in ['sports ball', 'ball'] and conf > 0.10:  # Very low threshold
                        x, y, w, h = box
                        x, y = x / scale, y / scale
                        w, h = w / scale, h / scale
                        
                        ball_size = max(w, h)
                        # More relaxed size filter
                        if 4 < ball_size < 200:
                            balls.append({
                                'center': [x, y],
                                'bbox': [x-w/2, y-h/2, w, h],
                                'conf': conf,
                                'size': ball_size
                            })
        
        return persons, balls
    
    def update_ball_tracking_advanced(self, detected_balls):
        """
        Advanced ball tracking with Kalman Filter
        """
        # Transform all ball detections to court coordinates
        valid_balls = []
        if detected_balls:
            ball_centers = [b['center'] for b in detected_balls]
            court_positions = self.transform_to_court(ball_centers)
            
            for ball, court_pos in zip(detected_balls, court_positions):
                if self.is_on_court(court_pos):
                    ball['court_pos'] = court_pos
                    valid_balls.append(ball)
        
        # Select best ball candidate
        best_ball = None
        
        if valid_balls:
            # Use Kalman Filter prediction to select best candidate
            predicted_pos = self.ball_tracker.predict()
            
            if predicted_pos is not None:
                # Find detection closest to prediction
                min_dist = float('inf')
                for ball in valid_balls:
                    dist = np.linalg.norm(ball['court_pos'] - predicted_pos)
                    # Weight by confidence and distance
                    weighted_score = dist / (ball['conf'] + 0.1)
                    if weighted_score < min_dist and dist < 3.0:
                        min_dist = weighted_score
                        best_ball = ball
            
            # If no good match with prediction, use highest confidence
            if best_ball is None:
                best_ball = max(valid_balls, key=lambda x: x['conf'])
            
            # Update Kalman Filter
            if best_ball:
                self.ball_tracker.update(best_ball['court_pos'])
                self.last_ball_detection = best_ball
                self.ball_detected_count += 1
                
                # Store detection
                self.ball_detections.append({
                    'frame': self.current_frame,
                    'court_pos': best_ball['court_pos'],
                    'image_pos': best_ball['center'],
                    'bbox': best_ball['bbox'],
                    'conf': best_ball['conf'],
                    'type': 'detected'
                })
        else:
            # No detection - use Kalman prediction
            predicted_pos = self.ball_tracker.predict()
            
            if predicted_pos is not None:
                # Transform prediction to image coordinates
                predicted_image = self.transform_to_image([predicted_pos])[0]
                
                # Store predicted position
                self.ball_detections.append({
                    'frame': self.current_frame,
                    'court_pos': predicted_pos,
                    'image_pos': predicted_image,
                    'bbox': None,
                    'conf': 0.5,  # Predicted confidence
                    'type': 'predicted'
                })
        
        # Update tracking history with smoothed trajectory
        if len(self.ball_detections) > 0:
            current_pos = self.ball_tracker.get_position()
            if current_pos is not None:
                image_pos = self.transform_to_image([current_pos])[0]
                
                self.ball_tracks.append({
                    'frame': self.current_frame,
                    'court_pos': current_pos,
                    'image_pos': image_pos,
                    'velocity': self.ball_tracker.get_velocity(),
                    'detected': len(valid_balls) > 0
                })
    
    def assign_players_enhanced(self, detected_persons):
        """Enhanced player assignment (same as before)"""
        if not detected_persons:
            for pid in range(1, 5):
                if self.players[pid]['active']:
                    self.players[pid]['confidence'] *= 0.95
                    if self.players[pid]['confidence'] < 0.1:
                        self.players[pid]['active'] = False
                        self.players[pid]['position'] = None
            return
        
        person_centers = [p['center'] for p in detected_persons]
        court_positions = self.transform_to_court(person_centers)
        
        valid_detections = []
        for person, court_pos in zip(detected_persons, court_positions):
            if self.is_on_court(court_pos):
                person['court_pos'] = court_pos
                person['zone'] = self.get_player_zone(court_pos)
                person['adaptive_conf'] = self.get_adaptive_confidence(court_pos)
                
                if person['conf'] >= person['adaptive_conf']:
                    valid_detections.append(person)
        
        if not valid_detections:
            return
        
        near_detections = [p for p in valid_detections if p['zone'] == 'near']
        far_detections = [p for p in valid_detections if p['zone'] == 'far']
        
        self.assign_zone_players(near_detections, [1, 2])
        self.assign_zone_players(far_detections, [3, 4])
    
    def assign_zone_players(self, detections, player_ids):
        """Assign detections to zone players"""
        if not detections:
            return
        
        active_players = {pid: self.players[pid] for pid in player_ids if self.players[pid]['active']}
        
        if not active_players:
            for i, detection in enumerate(detections[:2]):
                pid = player_ids[i]
                self.assign_player(pid, detection)
            return
        
        if len(active_players) > 0:
            detection_positions = np.array([d['court_pos'] for d in detections])
            active_ids = list(active_players.keys())
            active_positions = np.array([self.players[pid]['position']['court_pos'] for pid in active_ids])
            
            distances = cdist(detection_positions, active_positions)
            
            used_detections = set()
            used_players = set()
            
            assignments = []
            for i in range(distances.shape[0]):
                for j in range(distances.shape[1]):
                    assignments.append((distances[i, j], i, j))
            
            assignments.sort()
            
            max_distance = 2.5 if detections[0]['zone'] == 'far' else 1.8
            
            for dist, det_idx, player_idx in assignments:
                if det_idx not in used_detections and player_idx not in used_players:
                    if dist < max_distance:
                        pid = active_ids[player_idx]
                        self.assign_player(pid, detections[det_idx])
                        used_detections.add(det_idx)
                        used_players.add(player_idx)
            
            available_slots = [pid for pid in player_ids if not self.players[pid]['active']]
            remaining_detections = [detections[i] for i in range(len(detections)) if i not in used_detections]
            
            for i, detection in enumerate(remaining_detections[:len(available_slots)]):
                pid = available_slots[i]
                self.assign_player(pid, detection)
    
    def assign_player(self, player_id, detection):
        """Assign detection to player"""
        self.players[player_id]['active'] = True
        self.players[player_id]['position'] = detection
        self.players[player_id]['zone'] = detection['zone']
        self.players[player_id]['confidence'] = detection['conf']
        
        self.players[player_id]['history'].append({
            'court_pos': detection['court_pos'],
            'size': detection.get('size', 0),
            'conf': detection['conf']
        })
        
        self.player_tracks[player_id].append({
            'frame': self.current_frame,
            'court_pos': detection['court_pos'],
            'image_pos': detection['center'],
            'bbox': detection['bbox'],
            'conf': detection['conf'],
            'zone': detection['zone']
        })
    
    def draw_court_zones(self, frame):
        """Draw court boundaries"""
        court_corners = np.array([
            [0, 0], [self.court_width, 0],
            [self.court_width, self.court_length], [0, self.court_length]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        image_corners = cv2.perspectiveTransform(court_corners, np.linalg.inv(self.homography))
        image_corners = image_corners.reshape(-1, 2).astype(int)
        
        cv2.polylines(frame, [image_corners], True, (0, 255, 0), 4)
        
        # Net line
        net_start = np.array([[0, self.court_length/2]], dtype=np.float32).reshape(-1, 1, 2)
        net_end = np.array([[self.court_width, self.court_length/2]], dtype=np.float32).reshape(-1, 1, 2)
        
        net_start_img = cv2.perspectiveTransform(net_start, np.linalg.inv(self.homography))[0][0].astype(int)
        net_end_img = cv2.perspectiveTransform(net_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        cv2.line(frame, tuple(net_start_img), tuple(net_end_img), (255, 255, 255), 6)
        
        return frame
    
    def draw_all_tracking(self, frame):
        """Draw enhanced tracking with advanced ball visualization"""
        frame = self.draw_court_zones(frame)
        
        # Draw players
        for player_id in range(1, 5):
            if self.players[player_id]['active']:
                player = self.players[player_id]['position']
                color = self.players[player_id]['color']
                
                x, y, w, h = player['bbox']
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 3)
                
                center = player['center']
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, color, -1)
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, (255, 255, 255), 2)
                
                label = f'P{player_id}'
                cv2.putText(frame, label, (int(center[0])+15, int(center[1])-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
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
        
        # Draw ADVANCED BALL TRACKING
        if len(self.ball_tracks) > 0:
            # Draw smooth trajectory
            if len(self.ball_tracks) > 2:
                trajectory_points = []
                for track in list(self.ball_tracks)[-40:]:
                    pos = track['image_pos']
                    trajectory_points.append((int(pos[0]), int(pos[1])))
                
                # Draw trajectory line with gradient
                for i in range(1, len(trajectory_points)):
                    alpha = i / len(trajectory_points)
                    color_intensity = int(255 * alpha)
                    color = (0, color_intensity, 255)
                    thickness = int(2 + alpha * 2)
                    cv2.line(frame, trajectory_points[i-1], trajectory_points[i], color, thickness)
            
            # Draw current ball position
            current_ball = self.ball_tracks[-1]
            pos = current_ball['image_pos']
            
            # Different visualization for detected vs predicted
            if current_ball.get('detected', False):
                # Detected ball (solid yellow)
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 10, (0, 255, 255), -1)
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 10, (0, 0, 0), 2)
                label = 'BALL (tracked)'
                label_color = (0, 255, 255)
            else:
                # Predicted ball (dashed circle)
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 10, (100, 200, 255), 2)
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 3, (255, 255, 0), -1)
                label = 'BALL (predicted)'
                label_color = (100, 200, 255)
            
            cv2.putText(frame, label, (int(pos[0])+12, int(pos[1])-12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
            
            # Draw velocity vector
            velocity = current_ball.get('velocity')
            if velocity is not None:
                vx, vy = velocity
                speed = np.sqrt(vx**2 + vy**2)
                if speed > 0.1:  # Only draw if significant velocity
                    # Scale velocity for visualization
                    end_x = int(pos[0] + vx * 50)
                    end_y = int(pos[1] + vy * 50)
                    cv2.arrowedLine(frame, (int(pos[0]), int(pos[1])), (end_x, end_y),
                                  (0, 165, 255), 2, tipLength=0.3)
        
        return frame
    
    def add_stats_panel(self, frame):
        """Enhanced statistics panel with ball tracking info"""
        panel_height = 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (520, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        near_count = sum(1 for pid in [1, 2] if self.players[pid]['active'])
        far_count = sum(1 for pid in [3, 4] if self.players[pid]['active'])
        
        # Ball statistics
        recent_detections = len([b for b in self.ball_detections if 
                                self.current_frame - b['frame'] < 30 and b['type'] == 'detected'])
        recent_predictions = len([b for b in self.ball_detections if 
                                 self.current_frame - b['frame'] < 30 and b['type'] == 'predicted'])
        
        ball_speed = 0
        if len(self.ball_tracks) > 0:
            velocity = self.ball_tracks[-1].get('velocity')
            if velocity is not None:
                ball_speed = np.sqrt(velocity[0]**2 + velocity[1]**2) * self.fps  # m/s
        
        stats = [
            f"Frame: {self.current_frame:,} / {self.total_frames:,}",
            f"Progress: {self.current_frame/self.total_frames*100:.1f}%",
            f"",
            f"PLAYER TRACKING:",
            f"  Near: {near_count}/2 | Far: {far_count}/2",
            f"",
            f"ADVANCED BALL TRACKING:",
            f"  Total detections: {self.ball_detected_count}",
            f"  Recent detected: {recent_detections}",
            f"  Recent predicted: {recent_predictions}",
            f"  Ball speed: {ball_speed:.1f} m/s",
            f"  Trajectory points: {len(self.ball_tracks)}",
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 35 + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main execution"""
        print("🚀 Starting ADVANCED San4 Analysis...")
        print("🎾 Advanced Ball Tracking with Kalman Filter")
        print("⏹️  Press 'q' to quit")
        
        cv2.namedWindow('Advanced San4 Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Advanced San4 Tracking', 1400, 900)
        
        frame_times = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("🏁 Video finished")
                    break
                
                self.current_frame += 1
                
                # Detection and tracking
                detected_persons, detected_balls = self.detect_objects_adaptive(frame)
                self.assign_players_enhanced(detected_persons)
                self.update_ball_tracking_advanced(detected_balls)
                
                # Visualization
                frame = self.draw_all_tracking(frame)
                frame = self.add_stats_panel(frame)
                
                cv2.imshow('Advanced San4 Tracking', frame)
                
                # Performance
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                
                if self.current_frame % 150 == 0:
                    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    print(f"⚡ Frame {self.current_frame:,} | Ball detections: {self.ball_detected_count} | FPS: {avg_fps:.1f}")
                
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
            
            # Final statistics
            print("\n📊 Final Ball Tracking Statistics:")
            print(f"   Total ball detections: {self.ball_detected_count}")
            print(f"   Detection rate: {self.ball_detected_count / max(self.current_frame, 1) * 100:.1f}%")
            print("🎉 Advanced analysis complete!")

if __name__ == "__main__":
    analyzer = AdvancedTrackingSan4()
    analyzer.run()