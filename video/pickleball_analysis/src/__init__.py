"""
Pickleball Video Analysis Pipeline

Một hệ thống phân tích video pickleball hoàn chỉnh từ 4 góc camera
để tạo ra tracking, heatmap, và phân tích chuyển động chi tiết.

Modules:
- court_detection: Phát hiện và chuẩn hóa sân bằng homography
- detection: YOLO object detection cho người chơi và bóng
- tracking: Multi-object tracking với ByteTrack/OC-SORT
- analysis: Phân tích chuyển động, vận tốc, quãng đường
- visualization: Tạo heatmap, charts, và video output
- utils: Các utility functions chung

Author: AI Assistant
Date: October 2, 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = ""
__description__ = "Pickleball Video Analysis Pipeline"

# Import các module chính
from . import court_detection
from . import detection
from . import tracking
from . import analysis
from . import visualization
from . import utils

__all__ = [
    'court_detection',
    'detection', 
    'tracking',
    'analysis',
    'visualization',
    'utils'
]