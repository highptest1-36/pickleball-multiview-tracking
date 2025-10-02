import cv2
import numpy as np
import json
from collections import defaultdict, deque
import time
import torch
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter

class PlayerTracker:
    """
    Advanced player tracker with Re-ID features to prevent ID switching
    """
    def __init__(self, player_id, zone, color):
        self.player_id = player_id
        self.zone = zone
        self.color = color
        
        # Kalman Filter for player motion prediction
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0., 0., 0., 0.])  # [x, y, vx, vy]
        
        dt = 1/30.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.kf.R *= 1.0  # Measurement noise
        self.kf.Q *= 0.1  # Process noise
        self.kf.P *= 100
        
        # Re-ID features
        self.color_histogram = None
        self.avg_size = None
        self.appearance_history = deque(maxlen=10)
        
        # Tracking state
        self.active = False
        self.confidence = 0.0
        self.position = None
        self.missing_count = 0
        self.max_missing = 15
        
        # History
        self.history = deque(maxlen=30)
        self.tracks = deque(maxlen=150)
        
        self.kf_initialized = False
    
    def extract_appearance_features(self, frame, bbox):
        """
        Extract appearance features for Re-ID
        """
        try:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Ensure valid bbox
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            
            if x2 <= x or y2 <= y:
                return None
            
            roi = frame[y:y2, x:x2]
            
            if roi.size == 0:
                return None
            
            # Color histogram (HSV)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            return {
                'color_hist': hist,
                'size': w * h
            }
            
        except Exception as e:
            return None
    
    def update_appearance(self, features):
        """Update appearance model"""
        if features is None:
            return
        
        self.appearance_history.append(features)
        
        # Update average color histogram
        if len(self.appearance_history) > 0:
            hists = [f['color_hist'] for f in self.appearance_history]
            self.color_histogram = np.mean(hists, axis=0)
            
            sizes = [f['size'] for f in self.appearance_history]
            self.avg_size = np.mean(sizes)
    
    def compute_appearance_similarity(self, features):
        """
        Compute similarity between stored appearance and new detection
        Returns: similarity score [0, 1], higher is better
        """
        if features is None or self.color_histogram is None:
            return 0.5
        
        # Color histogram similarity (correlation)
        hist_sim = cv2.compareHist(
            self.color_histogram.astype(np.float32),
            features['color_hist'].astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        hist_sim = (hist_sim + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Size consistency
        if self.avg_size is not None and self.avg_size > 0:
            size_ratio = features['size'] / self.avg_size
            size_sim = 1.0 - min(abs(1.0 - size_ratio), 1.0)
        else:
            size_sim = 0.5
        
        # Weighted combination
        appearance_sim = 0.7 * hist_sim + 0.3 * size_sim
        
        return max(0.0, min(1.0, appearance_sim))
    
    def predict_position(self):
        """Predict next position using Kalman Filter"""
        if not self.kf_initialized:
            return None
        
        self.kf.predict()
        return self.kf.x[:2]
    
    def update(self, detection, frame, current_frame):
        """Update tracker with new detection"""
        # Update Kalman Filter
        if not self.kf_initialized:
            self.kf.x[:2] = detection['court_pos']
            self.kf_initialized = True
        else:
            self.kf.predict()
            self.kf.update(detection['court_pos'])
        
        # Update appearance
        features = self.extract_appearance_features(frame, detection['bbox'])
        self.update_appearance(features)
        
        # Update state
        self.active = True
        self.position = detection
        self.confidence = min(1.0, self.confidence + 0.1)
        self.missing_count = 0
        
        # Update history
        self.history.append({
            'court_pos': detection['court_pos'],
            'size': detection.get('size', 0),
            'conf': detection['conf']
        })
        
        self.tracks.append({
            'frame': current_frame,
            'court_pos': detection['court_pos'],
            'image_pos': detection['center'],
            'bbox': detection['bbox'],
            'conf': detection['conf'],
            'zone': self.zone
        })
    
    def miss(self):
        """Handle missed detection"""
        self.missing_count += 1
        self.confidence *= 0.9
        
        if self.missing_count > self.max_missing:
            self.reset()
    
    def reset(self):
        """Reset tracker"""
        self.active = False
        self.position = None
        self.confidence = 0.0
        self.missing_count = 0
        self.kf_initialized = False
        self.kf.x = np.array([0., 0., 0., 0.])
        self.kf.P *= 100
        self.color_histogram = None
        self.avg_size = None
        self.appearance_history.clear()

class StableReIDTrackingSan4:
    """
    Stable tracking with Re-Identification to prevent ID switching
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
        
        # Initialize Player Trackers with Re-ID
        self.player_trackers = {
            1: PlayerTracker(1, 'near', (0, 0, 255)),
            2: PlayerTracker(2, 'near', (0, 100, 255)),
            3: PlayerTracker(3, 'far', (255, 0, 0)),
            4: PlayerTracker(4, 'far', (255, 0, 100))
        }
        
        self.current_frame = 0
        self.frame_store = None  # Store current frame for appearance extraction
        
        # Court zones
        self.near_zone = (0, self.court_length * 0.55)
        self.far_zone = (self.court_length * 0.45, self.court_length)
        
        # Statistics
        self.id_switches = 0
        self.total_assignments = 0
        
        print(f"🏟️ STABLE RE-ID TRACKING:")
        print(f"   - Appearance-based Re-Identification")
        print(f"   - Kalman Filter motion prediction")
        print(f"   - Anti ID-switching algorithm")
        print(f"   - Color histogram + size consistency")
    
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
        
        results = self.model(frame_resized, verbose=False, conf=0.2, iou=0.5)
        
        persons = []
        
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
        
        return persons
    
    def assign_players_with_reid(self, detected_persons):
        """
        Enhanced player assignment with Re-ID to prevent ID switching
        """
        if not detected_persons:
            # Handle missing detections
            for pid, tracker in self.player_trackers.items():
                if tracker.active:
                    tracker.miss()
            return
        
        # Transform to court coordinates
        person_centers = [p['center'] for p in detected_persons]
        court_positions = self.transform_to_court(person_centers)
        
        # Filter valid detections
        valid_detections = []
        for person, court_pos in zip(detected_persons, court_positions):
            if self.is_on_court(court_pos):
                person['court_pos'] = court_pos
                person['zone'] = self.get_player_zone(court_pos)
                person['adaptive_conf'] = self.get_adaptive_confidence(court_pos)
                
                if person['conf'] >= person['adaptive_conf']:
                    valid_detections.append(person)
        
        if not valid_detections:
            for pid, tracker in self.player_trackers.items():
                if tracker.active:
                    tracker.miss()
            return
        
        # Separate by zone
        near_detections = [p for p in valid_detections if p['zone'] == 'near']
        far_detections = [p for p in valid_detections if p['zone'] == 'far']
        
        # Assign with Re-ID
        self.assign_zone_with_reid(near_detections, [1, 2])
        self.assign_zone_with_reid(far_detections, [3, 4])
    
    def assign_zone_with_reid(self, detections, player_ids):
        """
        Advanced assignment using Re-ID features
        """
        if not detections:
            for pid in player_ids:
                if self.player_trackers[pid].active:
                    self.player_trackers[pid].miss()
            return
        
        # Get active trackers in this zone
        active_trackers = {pid: self.player_trackers[pid] 
                          for pid in player_ids 
                          if self.player_trackers[pid].active}
        
        if not active_trackers:
            # Initialize new trackers
            for i, detection in enumerate(detections[:len(player_ids)]):
                pid = player_ids[i]
                self.player_trackers[pid].update(detection, self.frame_store, self.current_frame)
                print(f"✅ Player {pid} initialized (zone: {detection['zone']})")
            return
        
        # Build cost matrix with multiple factors
        detection_positions = np.array([d['court_pos'] for d in detections])
        active_ids = list(active_trackers.keys())
        
        # Predict positions for active trackers
        predicted_positions = []
        for pid in active_ids:
            pred_pos = self.player_trackers[pid].predict_position()
            if pred_pos is None:
                pred_pos = self.player_trackers[pid].position['court_pos']
            predicted_positions.append(pred_pos)
        
        predicted_positions = np.array(predicted_positions)
        
        # Calculate distance to predictions
        spatial_distances = cdist(detection_positions, predicted_positions)
        
        # Calculate appearance similarities
        appearance_scores = np.zeros((len(detections), len(active_ids)))
        
        for i, detection in enumerate(detections):
            features = self.player_trackers[1].extract_appearance_features(
                self.frame_store, detection['bbox']
            )
            
            for j, pid in enumerate(active_ids):
                sim = self.player_trackers[pid].compute_appearance_similarity(features)
                appearance_scores[i, j] = sim
        
        # Combine scores (lower is better)
        # Normalize distances to [0, 1]
        max_dist = 3.0
        normalized_distances = np.clip(spatial_distances / max_dist, 0, 1)
        
        # Convert appearance similarity to cost (higher sim = lower cost)
        appearance_costs = 1.0 - appearance_scores
        
        # Weighted combination
        total_costs = 0.6 * normalized_distances + 0.4 * appearance_costs
        
        # Hungarian assignment
        used_detections = set()
        used_trackers = set()
        
        assignments = []
        for i in range(total_costs.shape[0]):
            for j in range(total_costs.shape[1]):
                assignments.append((total_costs[i, j], i, j))
        
        assignments.sort()
        
        # Assign with cost threshold
        cost_threshold = 0.7
        
        for cost, det_idx, tracker_idx in assignments:
            if det_idx not in used_detections and tracker_idx not in used_trackers:
                if cost < cost_threshold:
                    pid = active_ids[tracker_idx]
                    self.player_trackers[pid].update(
                        detections[det_idx], self.frame_store, self.current_frame
                    )
                    used_detections.add(det_idx)
                    used_trackers.add(tracker_idx)
                    self.total_assignments += 1
        
        # Handle unmatched detections (new players or ID switches)
        available_slots = [pid for pid in player_ids 
                          if not self.player_trackers[pid].active]
        remaining_detections = [detections[i] for i in range(len(detections)) 
                               if i not in used_detections]
        
        for i, detection in enumerate(remaining_detections[:len(available_slots)]):
            pid = available_slots[i]
            self.player_trackers[pid].update(detection, self.frame_store, self.current_frame)
            print(f"✅ Player {pid} activated (zone: {detection['zone']})")
        
        # Handle unmatched trackers
        unmatched_trackers = [active_ids[i] for i in range(len(active_ids)) 
                             if i not in used_trackers]
        for pid in unmatched_trackers:
            self.player_trackers[pid].miss()
    
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
        """Draw tracking with confidence indicators"""
        frame = self.draw_court_zones(frame)
        
        # Draw players
        for pid, tracker in self.player_trackers.items():
            if tracker.active and tracker.position is not None:
                player = tracker.position
                color = tracker.color
                
                x, y, w, h = player['bbox']
                
                # Box thickness based on confidence
                thickness = int(2 + tracker.confidence * 3)
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)
                
                center = player['center']
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, color, -1)
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, (255, 255, 255), 2)
                
                # Label with confidence
                label = f'P{pid} ({tracker.confidence:.0%})'
                cv2.putText(frame, label, (int(center[0])+15, int(center[1])-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Trail
                if len(tracker.tracks) > 3:
                    trail_points = []
                    for track in list(tracker.tracks)[-30:]:
                        trail_points.append(track['image_pos'])
                    
                    for i in range(1, len(trail_points)):
                        alpha = i / len(trail_points)
                        trail_color = tuple(int(c * alpha * 0.8) for c in color)
                        pt1 = (int(trail_points[i-1][0]), int(trail_points[i-1][1]))
                        pt2 = (int(trail_points[i][0]), int(trail_points[i][1]))
                        cv2.line(frame, pt1, pt2, trail_color, 3)
        
        return frame
    
    def add_stats_panel(self, frame):
        """Enhanced statistics with Re-ID metrics"""
        panel_height = 240
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (550, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        near_count = sum(1 for pid in [1, 2] if self.player_trackers[pid].active)
        far_count = sum(1 for pid in [3, 4] if self.player_trackers[pid].active)
        
        # Calculate average confidence
        active_confidences = [t.confidence for t in self.player_trackers.values() if t.active]
        avg_confidence = np.mean(active_confidences) if active_confidences else 0
        
        stats = [
            f"Frame: {self.current_frame:,} / {self.total_frames:,}",
            f"Progress: {self.current_frame/self.total_frames*100:.1f}%",
            f"",
            f"STABLE RE-ID TRACKING:",
            f"  Near: {near_count}/2 | Far: {far_count}/2",
            f"  Avg Confidence: {avg_confidence:.0%}",
            f"",
            f"RE-ID METRICS:",
            f"  Total assignments: {self.total_assignments}",
            f"  ID switches: {self.id_switches}",
            f"  Switch rate: {self.id_switches/max(self.total_assignments,1)*100:.2f}%",
            f"",
            f"TRACKER STATUS:"
        ]
        
        for pid, tracker in self.player_trackers.items():
            status = "ACTIVE" if tracker.active else "inactive"
            conf = tracker.confidence if tracker.active else 0
            missing = tracker.missing_count if tracker.active else 0
            stats.append(f"  P{pid}: {status:8s} | Conf: {conf:.0%} | Miss: {missing}")
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 35 + i * 16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main execution"""
        print("🚀 Starting Stable Re-ID San4 Analysis...")
        print("🎯 Anti ID-Switching Algorithm Active")
        print("⏹️  Press 'q' to quit")
        
        cv2.namedWindow('Stable Re-ID Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stable Re-ID Tracking', 1400, 900)
        
        frame_times = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("🏁 Video finished")
                    break
                
                self.current_frame += 1
                self.frame_store = frame.copy()
                
                # Detection and tracking
                detected_persons = self.detect_objects_adaptive(frame)
                self.assign_players_with_reid(detected_persons)
                
                # Visualization
                frame = self.draw_all_tracking(frame)
                frame = self.add_stats_panel(frame)
                
                cv2.imshow('Stable Re-ID Tracking', frame)
                
                # Performance
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                
                if self.current_frame % 150 == 0:
                    avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
                    switch_rate = self.id_switches / max(self.total_assignments, 1) * 100
                    print(f"⚡ Frame {self.current_frame:,} | Assignments: {self.total_assignments} | Switches: {self.id_switches} ({switch_rate:.2f}%) | FPS: {avg_fps:.1f}")
                
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
            print("\n📊 Final Re-ID Statistics:")
            print(f"   Total assignments: {self.total_assignments}")
            print(f"   ID switches detected: {self.id_switches}")
            if self.total_assignments > 0:
                print(f"   ID stability: {(1 - self.id_switches/self.total_assignments)*100:.2f}%")
            print("🎉 Stable Re-ID analysis complete!")

if __name__ == "__main__":
    analyzer = StableReIDTrackingSan4()
    analyzer.run()