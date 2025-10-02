"""
Court Detection và Homography Transformation Module

Module này xử lý:
1. Phát hiện 4 góc sân pickleball từ video
2. Hiệu chỉnh thủ công các điểm góc sân  
3. Áp dụng homography transformation để chuyển về bird's-eye view
4. Lưu và load các điểm chuẩn
"""

import cv2
import numpy as np
import json
import os
import argparse
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import matplotlib.pyplot as plt

from .utils import load_config, load_court_points, save_court_points

class CourtDetector:
    """Class chính để xử lý detection và homography transformation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Khởi tạo CourtDetector.
        
        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        self.config = load_config(config_path)
        self.court_points = load_court_points("config/court_points.json")
        
        # Thông số sân pickleball
        self.court_width_m = self.config['court']['width_meters']
        self.court_height_m = self.config['court']['height_meters']
        
        # Bird's-eye view target dimensions (10 pixels = 1 meter)
        self.target_width = int(self.court_width_m * 10)  # 134 pixels
        self.target_height = int(self.court_height_m * 10)  # 61 pixels
        
        # Target corners for bird's-eye view
        self.target_corners = np.array([
            [0, 0],  # top-left
            [self.target_width, 0],  # top-right  
            [0, self.target_height],  # bottom-left
            [self.target_width, self.target_height]  # bottom-right
        ], dtype=np.float32)
        
        logger.info(f"CourtDetector khởi tạo với target size: {self.target_width}x{self.target_height}")

    def calibrate_camera(self, video_path: str, camera_name: str) -> bool:
        """
        Hiệu chỉnh thủ công điểm góc sân cho một camera.
        
        Args:
            video_path: Đường dẫn đến video
            camera_name: Tên camera (san1, san2, san3, san4)
            
        Returns:
            True nếu hiệu chỉnh thành công
        """
        logger.info(f"Bắt đầu hiệu chỉnh camera {camera_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Không thể mở video: {video_path}")
            return False
            
        # Đọc frame đầu tiên
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"Không thể đọc frame từ video: {video_path}")
            return False
            
        # Clone frame để vẽ
        display_frame = frame.copy()
        selected_points = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_points, display_frame
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(selected_points) < 4:
                    selected_points.append([x, y])
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"{len(selected_points)}", 
                              (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Labels cho từng điểm
                    labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
                    if len(selected_points) <= 4:
                        logger.info(f"Điểm {len(selected_points)}: {labels[len(selected_points)-1]} = ({x}, {y})")
                    
                    cv2.imshow("Court Calibration", display_frame)
        
        # Tạo window và set mouse callback
        cv2.namedWindow("Court Calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Court Calibration", mouse_callback)
        
        # Hiển thị hướng dẫn
        instructions = [
            "HƯỚNG DẪN HIỆU CHỈNH SÂN:",
            "1. Click vào 4 góc sân theo thứ tự:",
            "   - Top-Left (góc trên trái)",
            "   - Top-Right (góc trên phải)", 
            "   - Bottom-Left (góc dưới trái)",
            "   - Bottom-Right (góc dưới phải)",
            "2. Nhấn 's' để lưu",
            "3. Nhấn 'r' để reset",
            "4. Nhấn 'q' để thoát"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Court Calibration", display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save
                if len(selected_points) == 4:
                    # Lưu vào court_points
                    self.court_points['cameras'][camera_name]['court_corners'] = {
                        'top_left': selected_points[0],
                        'top_right': selected_points[1], 
                        'bottom_left': selected_points[2],
                        'bottom_right': selected_points[3]
                    }
                    self.court_points['cameras'][camera_name]['calibration_status'] = 'calibrated'
                    
                    save_court_points(self.court_points)
                    logger.info(f"Đã lưu calibration cho {camera_name}")
                    
                    # Hiển thị preview homography
                    self._preview_homography(frame, selected_points, camera_name)
                    break
                else:
                    logger.warning("Cần chọn đủ 4 điểm trước khi lưu!")
                    
            elif key == ord('r'):  # Reset
                selected_points = []
                display_frame = frame.copy()
                for i, instruction in enumerate(instructions):
                    cv2.putText(display_frame, instruction, (10, 30 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Court Calibration", display_frame)
                logger.info("Reset điểm đã chọn")
                
            elif key == ord('q'):  # Quit
                logger.info("Thoát calibration")
                break
        
        cv2.destroyAllWindows()
        return len(selected_points) == 4
    
    def _preview_homography(self, frame: np.ndarray, source_points: List[List[int]], 
                           camera_name: str) -> None:
        """
        Preview kết quả homography transformation.
        
        Args:
            frame: Frame gốc
            source_points: 4 điểm góc sân đã chọn
            camera_name: Tên camera
        """
        try:
            # Convert points to correct format
            src_pts = np.array(source_points, dtype=np.float32)
            
            # Compute homography
            H = cv2.getPerspectiveTransform(src_pts, self.target_corners)
            
            # Apply transformation
            warped = cv2.warpPerspective(frame, H, (self.target_width, self.target_height))
            
            # Resize để hiển thị dễ nhìn
            display_warped = cv2.resize(warped, (400, 200))
            
            cv2.imshow(f"Birds Eye View - {camera_name}", display_warped)
            cv2.waitKey(3000)  # Hiển thị 3 giây
            cv2.destroyWindow(f"Birds Eye View - {camera_name}")
            
            logger.info(f"Preview homography cho {camera_name} thành công")
            
        except Exception as e:
            logger.error(f"Lỗi khi preview homography: {e}")

    def get_homography_matrix(self, camera_name: str) -> Optional[np.ndarray]:
        """
        Lấy ma trận homography cho camera đã calibrate.
        
        Args:
            camera_name: Tên camera
            
        Returns:
            Ma trận homography hoặc None nếu chưa calibrate
        """
        if self.court_points['cameras'][camera_name]['calibration_status'] != 'calibrated':
            logger.warning(f"Camera {camera_name} chưa được calibrate")
            return None
            
        corners = self.court_points['cameras'][camera_name]['court_corners']
        src_pts = np.array([
            corners['top_left'],
            corners['top_right'],
            corners['bottom_left'], 
            corners['bottom_right']
        ], dtype=np.float32)
        
        try:
            H = cv2.getPerspectiveTransform(src_pts, self.target_corners)
            return H
        except Exception as e:
            logger.error(f"Lỗi khi tính homography cho {camera_name}: {e}")
            return None

    def transform_frame(self, frame: np.ndarray, camera_name: str) -> Optional[np.ndarray]:
        """
        Transform frame thành bird's-eye view.
        
        Args:
            frame: Frame gốc
            camera_name: Tên camera
            
        Returns:
            Frame đã transform hoặc None nếu lỗi
        """
        H = self.get_homography_matrix(camera_name)
        if H is None:
            return None
            
        try:
            warped = cv2.warpPerspective(frame, H, (self.target_width, self.target_height))
            return warped
        except Exception as e:
            logger.error(f"Lỗi khi transform frame: {e}")
            return None

    def transform_point(self, point: Tuple[float, float], camera_name: str) -> Optional[Tuple[float, float]]:
        """
        Transform một điểm từ camera view sang bird's-eye view.
        
        Args:
            point: Điểm (x, y) trong camera view
            camera_name: Tên camera
            
        Returns:
            Điểm (x, y) trong bird's-eye view hoặc None nếu lỗi
        """
        H = self.get_homography_matrix(camera_name)
        if H is None:
            return None
            
        try:
            # Convert point to homogeneous coordinates
            pt = np.array([[point[0], point[1]]], dtype=np.float32)
            pt = pt.reshape(-1, 1, 2)
            
            # Transform point
            transformed = cv2.perspectiveTransform(pt, H)
            
            return (float(transformed[0][0][0]), float(transformed[0][0][1]))
            
        except Exception as e:
            logger.error(f"Lỗi khi transform point: {e}")
            return None

    def batch_transform_points(self, points: List[Tuple[float, float]], 
                              camera_name: str) -> List[Optional[Tuple[float, float]]]:
        """
        Transform nhiều điểm cùng lúc.
        
        Args:
            points: Danh sách điểm [(x, y), ...]
            camera_name: Tên camera
            
        Returns:
            Danh sách điểm đã transform
        """
        H = self.get_homography_matrix(camera_name)
        if H is None:
            return [None] * len(points)
            
        try:
            if not points:
                return []
                
            # Convert to numpy array
            pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform all points
            transformed = cv2.perspectiveTransform(pts, H)
            
            # Convert back to list of tuples
            result = []
            for pt in transformed:
                result.append((float(pt[0][0]), float(pt[0][1])))
                
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi batch transform points: {e}")
            return [None] * len(points)

    def visualize_court_setup(self, output_path: str = "output/court_visualization.png") -> None:
        """
        Tạo visualization cho setup của các camera.
        
        Args:
            output_path: Đường dẫn lưu hình ảnh
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Court Setup Visualization', fontsize=16)
        
        camera_names = ['san1', 'san2', 'san3', 'san4']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for idx, (camera_name, (row, col)) in enumerate(zip(camera_names, positions)):
            ax = axes[row, col]
            
            # Vẽ sân pickleball chuẩn
            court_rect = plt.Rectangle((0, 0), self.court_width_m, self.court_height_m, 
                                     fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(court_rect)
            
            # Vẽ net
            ax.plot([0, self.court_width_m], [self.court_height_m/2, self.court_height_m/2], 
                   'r-', linewidth=3, label='Net')
            
            # Vẽ các line của sân
            # Service lines
            ax.plot([0, self.court_width_m], [1.83, 1.83], 'b--', alpha=0.7)
            ax.plot([0, self.court_width_m], [self.court_height_m-1.83, self.court_height_m-1.83], 'b--', alpha=0.7)
            
            # Center line
            ax.plot([self.court_width_m/2, self.court_width_m/2], [0, self.court_height_m], 'b--', alpha=0.7)
            
            # Hiển thị status
            status = self.court_points['cameras'][camera_name]['calibration_status']
            color = 'green' if status == 'calibrated' else 'red'
            ax.set_title(f'{camera_name.upper()} - {status}', color=color, fontweight='bold')
            
            # Nếu đã calibrate, hiển thị góc nhìn
            if status == 'calibrated':
                corners = self.court_points['cameras'][camera_name]['court_corners']
                # Vẽ field of view (đơn giản hóa)
                fov_color = ['red', 'blue', 'orange', 'purple'][idx]
                ax.scatter([0, self.court_width_m, 0, self.court_width_m], 
                          [0, 0, self.court_height_m, self.court_height_m], 
                          c=fov_color, s=100, alpha=0.7, label=f'FOV {camera_name}')
            
            ax.set_xlim(-1, self.court_width_m + 1)
            ax.set_ylim(-1, self.court_height_m + 1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Đã lưu court visualization vào {output_path}")

    def validate_calibration(self) -> Dict[str, bool]:
        """
        Kiểm tra trạng thái calibration của tất cả camera.
        
        Returns:
            Dictionary với trạng thái calibration của từng camera
        """
        validation_results = {}
        
        for camera_name in ['san1', 'san2', 'san3', 'san4']:
            status = self.court_points['cameras'][camera_name]['calibration_status']
            validation_results[camera_name] = (status == 'calibrated')
            
        calibrated_count = sum(validation_results.values())
        logger.info(f"Calibration status: {calibrated_count}/4 cameras được calibrate")
        
        return validation_results

def main():
    """Entry point cho court calibration."""
    parser = argparse.ArgumentParser(description="Court Detection và Calibration")
    parser.add_argument('--calibrate', action='store_true', 
                       help='Chạy calibration cho tất cả camera')
    parser.add_argument('--camera', type=str, choices=['san1', 'san2', 'san3', 'san4'],
                       help='Calibrate chỉ một camera cụ thể') 
    parser.add_argument('--visualize', action='store_true',
                       help='Tạo visualization cho court setup')
    parser.add_argument('--validate', action='store_true',
                       help='Kiểm tra trạng thái calibration')
    
    args = parser.parse_args()
    
    # Setup logging
    from .utils import setup_logging
    config = load_config()
    setup_logging(config)
    
    detector = CourtDetector()
    
    if args.calibrate or args.camera:
        # Lấy danh sách video
        video_paths = config['video']['input_videos']
        camera_names = ['san1', 'san2', 'san3', 'san4']
        
        if args.camera:
            # Calibrate chỉ một camera
            idx = camera_names.index(args.camera)
            video_path = video_paths[idx]
            detector.calibrate_camera(video_path, args.camera)
        else:
            # Calibrate tất cả camera
            for camera_name, video_path in zip(camera_names, video_paths):
                logger.info(f"Calibrating {camera_name}...")
                detector.calibrate_camera(video_path, camera_name)
    
    if args.visualize:
        detector.visualize_court_setup()
    
    if args.validate:
        results = detector.validate_calibration()
        for camera, status in results.items():
            print(f"{camera}: {'✓' if status else '✗'}")

if __name__ == "__main__":
    main()