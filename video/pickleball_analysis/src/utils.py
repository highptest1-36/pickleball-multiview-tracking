"""
Utility functions cho Pickleball Analysis Pipeline

Các hàm tiện ích chung được sử dụng trong toàn bộ pipeline.
"""

import os
import json
import yaml
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import time
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load cấu hình từ file YAML.
    
    Args:
        config_path: Đường dẫn đến file config
        
    Returns:
        Dictionary chứa cấu hình
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"Đã load config từ {config_path}")
        return config
    except Exception as e:
        logger.error(f"Lỗi khi load config: {e}")
        raise

def load_court_points(court_points_path: str = "config/court_points.json") -> Dict[str, Any]:
    """
    Load toạ độ góc sân từ file JSON.
    
    Args:
        court_points_path: Đường dẫn đến file court points
        
    Returns:
        Dictionary chứa toạ độ góc sân
    """
    try:
        with open(court_points_path, 'r', encoding='utf-8') as file:
            court_points = json.load(file)
        logger.info(f"Đã load court points từ {court_points_path}")
        return court_points
    except Exception as e:
        logger.error(f"Lỗi khi load court points: {e}")
        raise

def save_court_points(court_points: Dict[str, Any], 
                     court_points_path: str = "config/court_points.json") -> None:
    """
    Lưu toạ độ góc sân vào file JSON.
    
    Args:
        court_points: Dictionary chứa toạ độ góc sân
        court_points_path: Đường dẫn đến file lưu
    """
    try:
        with open(court_points_path, 'w', encoding='utf-8') as file:
            json.dump(court_points, file, indent=2, ensure_ascii=False)
        logger.info(f"Đã lưu court points vào {court_points_path}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu court points: {e}")
        raise

def create_output_dirs(base_output_dir: str, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Tạo các thư mục output theo cấu hình.
    
    Args:
        base_output_dir: Thư mục output chính
        config: Cấu hình từ config.yaml
        
    Returns:
        Dictionary chứa đường dẫn các thư mục output
    """
    output_dirs = {}
    
    try:
        # Tạo thư mục chính
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Tạo các thư mục con
        for subdir_key, subdir_name in config['output']['subdirs'].items():
            dir_path = os.path.join(base_output_dir, subdir_name)
            os.makedirs(dir_path, exist_ok=True)
            output_dirs[subdir_key] = dir_path
            logger.debug(f"Tạo thư mục: {dir_path}")
            
        logger.info(f"Đã tạo {len(output_dirs)} thư mục output")
        return output_dirs
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo thư mục output: {e}")
        raise

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Cấu hình logging system.
    
    Args:
        config: Cấu hình từ config.yaml
    """
    try:
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=config['logging']['level'],
            format=config['logging']['format']
        )
        
        # Add file handlers
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Main log file
        logger.add(
            f"{log_dir}/main.log",
            level=config['logging']['level'],
            format=config['logging']['format'],
            rotation=config['logging']['file_rotation'],
            retention=config['logging']['file_retention']
        )
        
        # Error log file  
        logger.add(
            f"{log_dir}/error.log",
            level="ERROR",
            format=config['logging']['format'],
            rotation=config['logging']['file_rotation'],
            retention=config['logging']['file_retention']
        )
        
        logger.info("Logging system đã được cấu hình")
        
    except Exception as e:
        print(f"Lỗi khi setup logging: {e}")
        raise

def validate_video_files(video_paths: List[str]) -> List[str]:
    """
    Kiểm tra và validate các file video đầu vào.
    
    Args:
        video_paths: Danh sách đường dẫn video
        
    Returns:
        Danh sách các file video hợp lệ
    """
    valid_videos = []
    
    for video_path in video_paths:
        if not os.path.exists(video_path):
            logger.warning(f"File video không tồn tại: {video_path}")
            continue
            
        # Kiểm tra có thể mở video không
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Không thể mở file video: {video_path}")
            cap.release()
            continue
            
        # Lấy thông tin video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if frame_count > 0:
            valid_videos.append(video_path)
            logger.info(f"Video hợp lệ: {video_path} ({frame_count} frames, {fps:.1f} FPS, {width}x{height})")
        else:
            logger.warning(f"Video không có frame: {video_path}")
    
    logger.info(f"Có {len(valid_videos)}/{len(video_paths)} video hợp lệ")
    return valid_videos

def pixel_to_meters(pixel_coords: Tuple[float, float], 
                   court_width_pixels: float, 
                   court_height_pixels: float,
                   court_width_meters: float = 13.41,
                   court_height_meters: float = 6.1) -> Tuple[float, float]:
    """
    Chuyển đổi toạ độ pixel sang mét trên sân thực.
    
    Args:
        pixel_coords: Toạ độ pixel (x, y)
        court_width_pixels: Chiều rộng sân trong pixel
        court_height_pixels: Chiều cao sân trong pixel
        court_width_meters: Chiều rộng sân thực (mét)
        court_height_meters: Chiều cao sân thực (mét)
        
    Returns:
        Toạ độ trong mét (x, y)
    """
    x_pixel, y_pixel = pixel_coords
    
    x_meters = (x_pixel / court_width_pixels) * court_width_meters
    y_meters = (y_pixel / court_height_pixels) * court_height_meters
    
    return (x_meters, y_meters)

def meters_to_pixel(meter_coords: Tuple[float, float],
                   court_width_pixels: float,
                   court_height_pixels: float, 
                   court_width_meters: float = 13.41,
                   court_height_meters: float = 6.1) -> Tuple[int, int]:
    """
    Chuyển đổi toạ độ mét sang pixel.
    
    Args:
        meter_coords: Toạ độ mét (x, y)
        court_width_pixels: Chiều rộng sân trong pixel
        court_height_pixels: Chiều cao sân trong pixel
        court_width_meters: Chiều rộng sân thực (mét)
        court_height_meters: Chiều cao sân thực (mét)
        
    Returns:
        Toạ độ pixel (x, y)
    """
    x_meters, y_meters = meter_coords
    
    x_pixel = int((x_meters / court_width_meters) * court_width_pixels)
    y_pixel = int((y_meters / court_height_meters) * court_height_pixels)
    
    return (x_pixel, y_pixel)

def calculate_distance(point1: Tuple[float, float], 
                      point2: Tuple[float, float]) -> float:
    """
    Tính khoảng cách Euclidean giữa 2 điểm.
    
    Args:
        point1: Điểm 1 (x, y)
        point2: Điểm 2 (x, y)
        
    Returns:
        Khoảng cách
    """
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_velocity(positions: List[Tuple[float, float]], 
                      timestamps: List[float],
                      smoothing_window: int = 3) -> List[Tuple[float, float]]:
    """
    Tính vận tốc từ chuỗi vị trí và timestamp.
    
    Args:
        positions: Danh sách vị trí [(x, y), ...]
        timestamps: Danh sách timestamp [t1, t2, ...]
        smoothing_window: Cửa sổ làm mượt
        
    Returns:
        Danh sách vận tốc [(vx, vy), ...]
    """
    if len(positions) < 2:
        return [(0, 0)] * len(positions)
    
    velocities = []
    
    for i in range(len(positions)):
        if i == 0:
            # First frame - use forward difference
            dt = timestamps[i+1] - timestamps[i]
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
        elif i == len(positions) - 1:
            # Last frame - use backward difference
            dt = timestamps[i] - timestamps[i-1]
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
        else:
            # Middle frames - use central difference
            dt = timestamps[i+1] - timestamps[i-1]
            dx = positions[i+1][0] - positions[i-1][0]
            dy = positions[i+1][1] - positions[i-1][1]
        
        if dt > 0:
            vx = dx / dt
            vy = dy / dt
        else:
            vx, vy = 0, 0
            
        velocities.append((vx, vy))
    
    # Apply smoothing if requested
    if smoothing_window > 1:
        velocities = smooth_velocity(velocities, smoothing_window)
    
    return velocities

def smooth_velocity(velocities: List[Tuple[float, float]], 
                   window_size: int) -> List[Tuple[float, float]]:
    """
    Làm mượt vận tốc bằng moving average.
    
    Args:
        velocities: Danh sách vận tốc [(vx, vy), ...]
        window_size: Kích thước cửa sổ
        
    Returns:
        Danh sách vận tốc đã làm mượt
    """
    if len(velocities) < window_size:
        return velocities
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(velocities)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(velocities), i + half_window + 1)
        
        vx_sum = sum(v[0] for v in velocities[start_idx:end_idx])
        vy_sum = sum(v[1] for v in velocities[start_idx:end_idx])
        count = end_idx - start_idx
        
        smoothed.append((vx_sum / count, vy_sum / count))
    
    return smoothed

def save_tracking_data(tracking_data: pd.DataFrame, 
                      output_path: str,
                      format_type: str = "csv") -> None:
    """
    Lưu dữ liệu tracking vào file.
    
    Args:
        tracking_data: DataFrame chứa dữ liệu tracking
        output_path: Đường dẫn file output
        format_type: Loại format ("csv", "json", "parquet")
    """
    try:
        if format_type.lower() == "csv":
            tracking_data.to_csv(output_path, index=False)
        elif format_type.lower() == "json":
            tracking_data.to_json(output_path, orient="records", indent=2)
        elif format_type.lower() == "parquet":
            tracking_data.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        logger.info(f"Đã lưu tracking data vào {output_path}")
        
    except Exception as e:
        logger.error(f"Lỗi khi lưu tracking data: {e}")
        raise

def load_tracking_data(input_path: str) -> pd.DataFrame:
    """
    Load dữ liệu tracking từ file.
    
    Args:
        input_path: Đường dẫn file input
        
    Returns:
        DataFrame chứa dữ liệu tracking
    """
    try:
        if input_path.endswith('.csv'):
            data = pd.read_csv(input_path)
        elif input_path.endswith('.json'):
            data = pd.read_json(input_path)
        elif input_path.endswith('.parquet'):
            data = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
            
        logger.info(f"Đã load tracking data từ {input_path}")
        return data
        
    except Exception as e:
        logger.error(f"Lỗi khi load tracking data: {e}")
        raise

class Timer:
    """Context manager để đo thời gian thực hiện."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Bắt đầu: {self.description}")
        return self
        
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        logger.info(f"Hoàn thành: {self.description} ({duration:.2f}s)")

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Lấy thông tin chi tiết của video.
    
    Args:
        video_path: Đường dẫn video
        
    Returns:
        Dictionary chứa thông tin video
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")
    
    info = {
        'path': video_path,
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_seconds': 0,
        'file_size_mb': 0
    }
    
    if info['fps'] > 0:
        info['duration_seconds'] = info['frame_count'] / info['fps']
    
    try:
        info['file_size_mb'] = os.path.getsize(video_path) / (1024 * 1024)
    except:
        info['file_size_mb'] = 0
    
    cap.release()
    return info

# Constants
PICKLEBALL_COURT_WIDTH = 13.41  # meters
PICKLEBALL_COURT_HEIGHT = 6.1   # meters
PICKLEBALL_NET_HEIGHT = 0.914   # meters