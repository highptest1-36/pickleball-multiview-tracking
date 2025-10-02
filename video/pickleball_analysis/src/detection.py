"""
YOLO Detection Module

Module này xử lý:
1. Load và configure YOLOv11x model
2. Detect người chơi và bóng trong video frames
3. Post-processing và filtering detections
4. Export detection results
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import time

from .utils import Timer

class PickleballDetector:
    """Class chính để xử lý YOLO detection cho pickleball."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo PickleballDetector.
        
        Args:
            config: Cấu hình từ config.yaml
        """
        self.config = config
        self.detection_config = config['detection']
        
        # YOLO model settings
        self.model_path = self.detection_config['model']
        self.confidence_threshold = self.detection_config['confidence_threshold']
        self.iou_threshold = self.detection_config['iou_threshold']
        self.device = self.detection_config['device']
        
        # Class mapping
        self.target_classes = self.detection_config['classes']
        
        # Initialize model
        self.model = None
        self._load_model()
        
        logger.info(f"PickleballDetector khởi tạo với device: {self.device}")

    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            with Timer("Loading YOLO model"):
                self.model = YOLO(self.model_path)
                
                # Set device
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model.to('cuda')
                    logger.info("Sử dụng GPU cho inference")
                else:
                    self.model.to('cpu')
                    logger.info("Sử dụng CPU cho inference")
                    
        except Exception as e:
            logger.error(f"Lỗi khi load YOLO model: {e}")
            raise

    def detect_frame(self, frame: np.ndarray, 
                    filter_classes: bool = True) -> List[Dict[str, Any]]:
        """
        Detect objects trong một frame.
        
        Args:
            frame: Input frame
            filter_classes: Có filter theo classes cần thiết không
            
        Returns:
            List các detection dict với format:
            {
                'bbox': [x1, y1, x2, y2],
                'center': [cx, cy], 
                'confidence': float,
                'class_id': int,
                'class_name': str
            }
        """
        if self.model is None:
            logger.error("Model chưa được load")
            return []
            
        try:
            # Run inference
            results = self.model(frame, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get box coordinates
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box
                        
                        # Get confidence and class
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Filter classes if needed
                        if filter_classes and not self._is_target_class(class_id):
                            continue
                            
                        # Calculate center point
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Get class name
                        class_name = self._get_class_name(class_id)
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'center': [center_x, center_y],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Lỗi khi detect frame: {e}")
            return []

    def _is_target_class(self, class_id: int) -> bool:
        """
        Kiểm tra class có phải target class không.
        
        Args:
            class_id: ID của class
            
        Returns:
            True nếu là target class
        """
        # COCO classes: person=0, sports ball=32
        return class_id in [0, 32]  # person and sports ball

    def _get_class_name(self, class_id: int) -> str:
        """
        Lấy tên class từ class ID.
        
        Args:
            class_id: ID của class
            
        Returns:
            Tên class
        """
        if class_id == 0:
            return "player"
        elif class_id == 32:
            return "ball"
        else:
            return f"class_{class_id}"

    def detect_video(self, video_path: str, 
                    output_callback: Optional[callable] = None,
                    max_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Detect objects trong toàn bộ video.
        
        Args:
            video_path: Đường dẫn video
            output_callback: Callback function để xử lý mỗi frame
            max_frames: Số frame tối đa để process (None = tất cả)
            
        Returns:
            List tất cả detections theo format:
            {
                'frame_id': int,
                'timestamp': float,
                'detections': List[detection_dict]
            }
        """
        logger.info(f"Bắt đầu detect video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Không thể mở video: {video_path}")
            return []
            
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
            
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")
        
        all_detections = []
        frame_id = 0
        
        with Timer(f"Processing {total_frames} frames"):
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_id >= max_frames):
                    break
                    
                # Calculate timestamp
                timestamp = frame_id / fps
                
                # Detect objects in frame
                detections = self.detect_frame(frame, filter_classes=True)
                
                frame_data = {
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'detections': detections
                }
                
                all_detections.append(frame_data)
                
                # Call output callback if provided
                if output_callback:
                    output_callback(frame, frame_data)
                
                # Log progress
                if frame_id % 100 == 0:
                    logger.info(f"Processed {frame_id}/{total_frames} frames")
                    
                frame_id += 1
        
        cap.release()
        logger.info(f"Hoàn thành detect video: {len(all_detections)} frames processed")
        
        return all_detections

    def filter_detections_by_area(self, detections: List[Dict[str, Any]], 
                                 min_area: float = 100,
                                 max_area: float = 50000) -> List[Dict[str, Any]]:
        """
        Filter detections theo diện tích bounding box.
        
        Args:
            detections: List detections
            min_area: Diện tích tối thiểu
            max_area: Diện tích tối đa
            
        Returns:
            List detections đã được filter
        """
        filtered = []
        
        for detection in detections:
            area = detection.get('area', 0)
            if min_area <= area <= max_area:
                filtered.append(detection)
                
        logger.debug(f"Filtered {len(detections)} -> {len(filtered)} detections by area")
        return filtered

    def filter_detections_by_position(self, detections: List[Dict[str, Any]],
                                    frame_width: int, frame_height: int,
                                    margin_ratio: float = 0.05) -> List[Dict[str, Any]]:
        """
        Filter detections ở rìa frame (có thể là noise).
        
        Args:
            detections: List detections
            frame_width: Chiều rộng frame
            frame_height: Chiều cao frame
            margin_ratio: Tỷ lệ margin (0.05 = 5%)
            
        Returns:
            List detections đã được filter
        """
        margin_x = frame_width * margin_ratio
        margin_y = frame_height * margin_ratio
        
        filtered = []
        
        for detection in detections:
            center_x, center_y = detection['center']
            
            # Kiểm tra center point có trong vùng hợp lệ không
            if (margin_x <= center_x <= frame_width - margin_x and
                margin_y <= center_y <= frame_height - margin_y):
                filtered.append(detection)
                
        logger.debug(f"Filtered {len(detections)} -> {len(filtered)} detections by position")
        return filtered

    def separate_players_and_balls(self, detections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Tách detections thành players và balls.
        
        Args:
            detections: List tất cả detections
            
        Returns:
            Tuple (player_detections, ball_detections)
        """
        players = []
        balls = []
        
        for detection in detections:
            if detection['class_name'] == 'player':
                players.append(detection)
            elif detection['class_name'] == 'ball':
                balls.append(detection)
                
        return players, balls

    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Dict[str, Any]],
                       show_confidence: bool = True,
                       show_center: bool = True) -> np.ndarray:
        """
        Vẽ detections lên frame.
        
        Args:
            frame: Input frame
            detections: List detections
            show_confidence: Hiển thị confidence score
            show_center: Hiển thị center point
            
        Returns:
            Frame với detections được vẽ
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            center = detection['center']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cx, cy = [int(coord) for coord in center]
            
            # Color theo class
            if class_name == 'player':
                color = (0, 255, 0)  # Green
            elif class_name == 'ball':
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 255)  # White
            
            # Vẽ bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ center point
            if show_center:
                cv2.circle(annotated_frame, (cx, cy), 3, color, -1)
            
            # Vẽ label
            label = class_name
            if show_confidence:
                label += f" {confidence:.2f}"
                
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated_frame

    def get_detection_statistics(self, all_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tính toán thống kê detection.
        
        Args:
            all_detections: List tất cả frame detections
            
        Returns:
            Dictionary chứa thống kê
        """
        stats = {
            'total_frames': len(all_detections),
            'total_detections': 0,
            'player_detections': 0,
            'ball_detections': 0,
            'avg_detections_per_frame': 0,
            'avg_players_per_frame': 0,
            'avg_balls_per_frame': 0,
            'frames_with_detections': 0,
            'frames_with_players': 0,
            'frames_with_balls': 0
        }
        
        for frame_data in all_detections:
            detections = frame_data['detections']
            
            if detections:
                stats['frames_with_detections'] += 1
                
            players, balls = self.separate_players_and_balls(detections)
            
            stats['total_detections'] += len(detections)
            stats['player_detections'] += len(players)
            stats['ball_detections'] += len(balls)
            
            if players:
                stats['frames_with_players'] += 1
            if balls:
                stats['frames_with_balls'] += 1
        
        # Calculate averages
        if stats['total_frames'] > 0:
            stats['avg_detections_per_frame'] = stats['total_detections'] / stats['total_frames']
            stats['avg_players_per_frame'] = stats['player_detections'] / stats['total_frames']
            stats['avg_balls_per_frame'] = stats['ball_detections'] / stats['total_frames']
        
        return stats

    def save_detection_video(self, video_path: str, detections: List[Dict[str, Any]], 
                           output_path: str) -> None:
        """
        Lưu video với detections được vẽ.
        
        Args:
            video_path: Đường dẫn video gốc
            detections: List frame detections
            output_path: Đường dẫn video output
        """
        logger.info(f"Bắt đầu lưu detection video: {output_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Không thể mở video: {video_path}")
            return
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while frame_idx < len(detections):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get detections for current frame
            frame_detections = detections[frame_idx]['detections']
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, frame_detections)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        logger.info(f"Đã lưu detection video: {output_path}")

def main():
    """Test function cho detection module."""
    from .utils import load_config, setup_logging
    
    # Load config
    config = load_config()
    setup_logging(config)
    
    # Initialize detector
    detector = PickleballDetector(config)
    
    # Test với một video
    video_path = "../data_video/san1.mp4"
    
    # Detect first 100 frames
    detections = detector.detect_video(video_path, max_frames=100)
    
    # Print statistics
    stats = detector.get_detection_statistics(detections)
    print("Detection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()