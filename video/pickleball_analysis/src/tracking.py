"""
Multi-Object Tracking Module

Module này xử lý:
1. Track nhiều người chơi và bóng qua các frame
2. Gán ID cố định cho mỗi đối tượng
3. Xử lý lost tracks và re-identification
4. Export tracking data
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import pandas as pd
from collections import defaultdict
import math

from .utils import Timer, calculate_distance

class SimpleTracker:
    """
    Simple multi-object tracker sử dụng centroid tracking và Kalman filter.
    Đây là implementation đơn giản, có thể thay thế bằng ByteTrack, DeepSORT sau.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo SimpleTracker.
        
        Args:
            config: Cấu hình từ config.yaml
        """
        self.config = config
        self.tracking_config = config['tracking']
        
        # Tracking parameters
        self.max_disappeared = self.tracking_config['max_disappeared']
        self.max_distance = self.tracking_config['max_distance']
        self.min_hits = self.tracking_config['min_hits']
        
        # Tracking state
        self.next_object_id = 0
        self.objects = {}  # object_id -> {'centroid': (x, y), 'disappeared': int, 'class': str, 'hits': int}
        self.disappeared = {}  # object_id -> frames_disappeared
        
        logger.info(f"SimpleTracker khởi tạo với max_distance={self.max_distance}")

    def register(self, centroid: Tuple[float, float], class_name: str) -> int:
        """
        Đăng ký object mới.
        
        Args:
            centroid: Tọa độ center của object
            class_name: Loại object ('player' hoặc 'ball')
            
        Returns:
            ID của object mới
        """
        object_id = self.next_object_id
        self.objects[object_id] = {
            'centroid': centroid,
            'class': class_name,
            'hits': 1,
            'age': 0
        }
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        
        logger.debug(f"Đăng ký object mới: ID={object_id}, class={class_name}, centroid={centroid}")
        return object_id

    def deregister(self, object_id: int) -> None:
        """
        Xóa object khỏi tracking.
        
        Args:
            object_id: ID của object cần xóa
        """
        if object_id in self.objects:
            del self.objects[object_id]
            del self.disappeared[object_id]
            logger.debug(f"Xóa object: ID={object_id}")

    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Update tracker với detections mới.
        
        Args:
            detections: List detections từ YOLO
            
        Returns:
            Dictionary tracking results: {object_id: {'centroid': (x,y), 'class': str, 'bbox': [x1,y1,x2,y2]}}
        """
        # Nếu không có detection
        if len(detections) == 0:
            # Increase disappeared count cho tất cả objects
            object_ids_to_delete = []
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    object_ids_to_delete.append(object_id)
            
            # Xóa objects đã mất quá lâu
            for object_id in object_ids_to_delete:
                self.deregister(object_id)
            
            return self._get_current_objects()
        
        # Nếu chưa có objects nào được track
        if len(self.objects) == 0:
            for detection in detections:
                centroid = tuple(detection['center'])
                class_name = detection['class_name']
                self.register(centroid, class_name)
        else:
            # Có objects và detections -> cần matching
            self._match_detections_to_objects(detections)
        
        return self._get_current_objects()

    def _match_detections_to_objects(self, detections: List[Dict[str, Any]]) -> None:
        """
        Match detections với existing objects.
        
        Args:
            detections: List detections
        """
        # Tách detections theo class
        player_detections = [d for d in detections if d['class_name'] == 'player']
        ball_detections = [d for d in detections if d['class_name'] == 'ball']
        
        # Match players
        self._match_by_class(player_detections, 'player')
        
        # Match balls
        self._match_by_class(ball_detections, 'ball')

    def _match_by_class(self, detections: List[Dict[str, Any]], class_name: str) -> None:
        """
        Match detections của một class cụ thể.
        
        Args:
            detections: List detections của class
            class_name: Tên class
        """
        if not detections:
            return
            
        # Lấy existing objects của class này
        existing_objects = {
            obj_id: obj_data for obj_id, obj_data in self.objects.items()
            if obj_data['class'] == class_name
        }
        
        if not existing_objects:
            # Không có existing objects -> register tất cả detections
            for detection in detections:
                centroid = tuple(detection['center'])
                self.register(centroid, class_name)
            return
        
        # Tính distance matrix
        object_ids = list(existing_objects.keys())
        detection_centroids = [detection['center'] for detection in detections]
        existing_centroids = [existing_objects[obj_id]['centroid'] for obj_id in object_ids]
        
        distance_matrix = self._compute_distance_matrix(existing_centroids, detection_centroids)
        
        # Hungarian algorithm (đơn giản hóa bằng greedy matching)
        used_detection_indices = set()
        used_object_indices = set()
        
        # Greedy matching
        matches = []
        for i in range(len(existing_centroids)):
            if i in used_object_indices:
                continue
                
            min_distance = float('inf')
            best_j = -1
            
            for j in range(len(detection_centroids)):
                if j in used_detection_indices:
                    continue
                    
                distance = distance_matrix[i][j]
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_j = j
            
            if best_j != -1:
                matches.append((i, best_j))
                used_object_indices.add(i)
                used_detection_indices.add(best_j)
        
        # Update matched objects
        for obj_idx, det_idx in matches:
            object_id = object_ids[obj_idx]
            detection = detections[det_idx]
            
            # Update object
            self.objects[object_id]['centroid'] = tuple(detection['center'])
            self.objects[object_id]['hits'] += 1
            self.objects[object_id]['age'] += 1
            self.disappeared[object_id] = 0
            
            # Store additional info
            self.objects[object_id]['bbox'] = detection['bbox']
            self.objects[object_id]['confidence'] = detection['confidence']
        
        # Handle unmatched objects
        for i, object_id in enumerate(object_ids):
            if i not in used_object_indices:
                self.disappeared[object_id] += 1
                self.objects[object_id]['age'] += 1
                
                # Remove if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        # Register new objects for unmatched detections
        for j, detection in enumerate(detections):
            if j not in used_detection_indices:
                centroid = tuple(detection['center'])
                self.register(centroid, class_name)

    def _compute_distance_matrix(self, centroids1: List[Tuple[float, float]], 
                                centroids2: List[Tuple[float, float]]) -> List[List[float]]:
        """
        Tính ma trận khoảng cách giữa 2 sets of centroids.
        
        Args:
            centroids1: List centroids của existing objects
            centroids2: List centroids của detections
            
        Returns:
            Ma trận khoảng cách
        """
        distance_matrix = []
        
        for c1 in centroids1:
            row = []
            for c2 in centroids2:
                distance = calculate_distance(c1, c2)
                row.append(distance)
            distance_matrix.append(row)
            
        return distance_matrix

    def _get_current_objects(self) -> Dict[int, Dict[str, Any]]:
        """
        Lấy trạng thái hiện tại của tất cả objects.
        
        Returns:
            Dictionary với current objects
        """
        current_objects = {}
        
        for object_id, obj_data in self.objects.items():
            # Chỉ return objects đã được confirm (hits >= min_hits)
            if obj_data['hits'] >= self.min_hits:
                current_objects[object_id] = obj_data.copy()
                current_objects[object_id]['object_id'] = object_id
                
        return current_objects

    def get_tracking_history(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Lấy lịch sử tracking của tất cả objects.
        
        Returns:
            Dictionary: {object_id: [(x, y), ...]}
        """
        # Implementation sẽ cần lưu history trong quá trình tracking
        # Đây là placeholder
        return {}

class PickleballTracker:
    """
    Main tracker class cho pickleball analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo PickleballTracker.
        
        Args:
            config: Cấu hình từ config.yaml
        """
        self.config = config
        self.tracker = SimpleTracker(config)
        
        # Tracking history
        self.tracking_history = defaultdict(list)  # object_id -> [{'frame': int, 'centroid': (x,y), 'timestamp': float}]
        self.frame_results = []  # List tất cả frame tracking results
        
        logger.info("PickleballTracker đã được khởi tạo")

    def process_video_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process toàn bộ video detections để tạo tracking results.
        
        Args:
            detections: List frame detections từ YOLO
            
        Returns:
            List tracking results theo format:
            {
                'frame_id': int,
                'timestamp': float,
                'tracked_objects': {object_id: object_data}
            }
        """
        logger.info(f"Bắt đầu tracking {len(detections)} frames")
        
        tracking_results = []
        
        with Timer("Multi-object tracking"):
            for frame_data in detections:
                frame_id = frame_data['frame_id']
                timestamp = frame_data['timestamp']
                frame_detections = frame_data['detections']
                
                # Update tracker
                tracked_objects = self.tracker.update(frame_detections)
                
                # Store frame result
                frame_result = {
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'tracked_objects': tracked_objects
                }
                
                tracking_results.append(frame_result)
                self.frame_results.append(frame_result)
                
                # Update tracking history
                self._update_tracking_history(frame_id, timestamp, tracked_objects)
                
                # Log progress
                if frame_id % 100 == 0:
                    logger.info(f"Tracked frame {frame_id}: {len(tracked_objects)} objects")
        
        logger.info(f"Hoàn thành tracking: {len(tracking_results)} frames")
        return tracking_results

    def _update_tracking_history(self, frame_id: int, timestamp: float, 
                                tracked_objects: Dict[int, Dict[str, Any]]) -> None:
        """
        Update tracking history cho tất cả objects.
        
        Args:
            frame_id: ID của frame
            timestamp: Timestamp của frame
            tracked_objects: Objects được track trong frame này
        """
        for object_id, obj_data in tracked_objects.items():
            history_entry = {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'centroid': obj_data['centroid'],
                'bbox': obj_data.get('bbox', []),
                'confidence': obj_data.get('confidence', 0),
                'class': obj_data['class']
            }
            
            self.tracking_history[object_id].append(history_entry)

    def export_tracking_data(self, output_path: str) -> pd.DataFrame:
        """
        Export tracking data thành CSV.
        
        Args:
            output_path: Đường dẫn file output
            
        Returns:
            DataFrame chứa tracking data
        """
        logger.info(f"Export tracking data vào {output_path}")
        
        # Tạo DataFrame từ tracking history
        rows = []
        
        for object_id, history in self.tracking_history.items():
            for entry in history:
                row = {
                    'frame_id': entry['frame_id'],
                    'timestamp': entry['timestamp'],
                    'object_id': object_id,
                    'class': entry['class'],
                    'center_x': entry['centroid'][0],
                    'center_y': entry['centroid'][1],
                    'confidence': entry['confidence']
                }
                
                # Add bbox info if available
                if entry['bbox']:
                    bbox = entry['bbox']
                    row.update({
                        'bbox_x1': bbox[0],
                        'bbox_y1': bbox[1], 
                        'bbox_x2': bbox[2],
                        'bbox_y2': bbox[3]
                    })
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by frame_id and object_id
        df = df.sort_values(['frame_id', 'object_id']).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Đã export {len(df)} tracking records")
        return df

    def get_tracking_statistics(self) -> Dict[str, Any]:
        """
        Tính toán thống kê tracking.
        
        Returns:
            Dictionary chứa thống kê
        """
        stats = {
            'total_objects_tracked': len(self.tracking_history),
            'total_frames': len(self.frame_results),
            'player_objects': 0,
            'ball_objects': 0,
            'avg_track_length': 0,
            'longest_track': 0,
            'shortest_track': float('inf'),
            'objects_per_frame': []
        }
        
        # Phân tích theo object
        track_lengths = []
        for object_id, history in self.tracking_history.items():
            track_length = len(history)
            track_lengths.append(track_length)
            
            if track_length > stats['longest_track']:
                stats['longest_track'] = track_length
            if track_length < stats['shortest_track']:
                stats['shortest_track'] = track_length
                
            # Count by class
            if history and history[0]['class'] == 'player':
                stats['player_objects'] += 1
            elif history and history[0]['class'] == 'ball':
                stats['ball_objects'] += 1
        
        if track_lengths:
            stats['avg_track_length'] = sum(track_lengths) / len(track_lengths)
        else:
            stats['shortest_track'] = 0
        
        # Phân tích theo frame
        for frame_result in self.frame_results:
            stats['objects_per_frame'].append(len(frame_result['tracked_objects']))
        
        if stats['objects_per_frame']:
            stats['avg_objects_per_frame'] = sum(stats['objects_per_frame']) / len(stats['objects_per_frame'])
        else:
            stats['avg_objects_per_frame'] = 0
        
        return stats

    def visualize_tracking(self, frame: np.ndarray, 
                          tracked_objects: Dict[int, Dict[str, Any]],
                          show_trail: bool = True,
                          trail_length: int = 10) -> np.ndarray:
        """
        Vẽ tracking results lên frame.
        
        Args:
            frame: Input frame
            tracked_objects: Objects được track
            show_trail: Hiển thị trail của objects
            trail_length: Độ dài trail
            
        Returns:
            Frame với tracking visualization
        """
        annotated_frame = frame.copy()
        
        # Colors cho từng object ID
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        for object_id, obj_data in tracked_objects.items():
            # Chọn màu theo object ID
            color = colors[object_id % len(colors)]
            
            # Vẽ bounding box nếu có
            if 'bbox' in obj_data and obj_data['bbox']:
                bbox = obj_data['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ center point
            centroid = obj_data['centroid']
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(annotated_frame, (cx, cy), 5, color, -1)
            
            # Vẽ object ID
            label = f"ID:{object_id} ({obj_data['class']})"
            cv2.putText(annotated_frame, label, (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Vẽ trail nếu được yêu cầu
            if show_trail and object_id in self.tracking_history:
                history = self.tracking_history[object_id]
                if len(history) > 1:
                    # Lấy trail_length points cuối
                    recent_points = history[-trail_length:]
                    points = [entry['centroid'] for entry in recent_points]
                    
                    # Vẽ trail
                    for i in range(1, len(points)):
                        pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                        pt2 = (int(points[i][0]), int(points[i][1]))
                        alpha = i / len(points)  # Fade effect
                        trail_color = tuple(int(c * alpha) for c in color)
                        cv2.line(annotated_frame, pt1, pt2, trail_color, 2)
        
        return annotated_frame

def main():
    """Test function cho tracking module."""
    from .utils import load_config, setup_logging
    from .detection import PickleballDetector
    
    # Load config
    config = load_config()
    setup_logging(config)
    
    # Test tracking với detection data
    detector = PickleballDetector(config)
    tracker = PickleballTracker(config)
    
    # Detect first 50 frames
    video_path = "../data_video/san1.mp4"
    detections = detector.detect_video(video_path, max_frames=50)
    
    # Track objects
    tracking_results = tracker.process_video_detections(detections)
    
    # Export tracking data
    df = tracker.export_tracking_data("output/test_tracking.csv")
    
    # Print statistics
    stats = tracker.get_tracking_statistics()
    print("Tracking Statistics:")
    for key, value in stats.items():
        if key != 'objects_per_frame':  # Skip list values
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()