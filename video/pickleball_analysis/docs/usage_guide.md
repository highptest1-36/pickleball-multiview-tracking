# Usage Guide - Pickleball Video Analysis Pipeline

## 🎯 Tổng quan

Pipeline này cho phép bạn phân tích video pickleball và tạo ra:
- 📍 Player tracking với unique IDs
- 🔥 Heatmaps cho movement patterns  
- 📊 Speed analysis và running pace
- 📈 Trajectory visualization
- 📋 Performance reports chi tiết

## 🚀 Quick Start

### Bước 1: Chuẩn bị video input
```bash
# Copy video files vào data_video/
cp your_videos/*.mp4 data_video/

# Hoặc tạo symbolic link
ln -s /path/to/your/videos data_video
```

### Bước 2: Chạy pipeline cơ bản
```bash
# Chạy full pipeline với video mặc định
python main.py

# Hoặc specify input directory
python main.py --input data_video/ --output results/
```

### Bước 3: Xem kết quả
```bash
# Mở interactive dashboard
open output/charts/interactive_dashboard.html

# Xem tracking data
open output/tracking_data/tracking_results.csv

# Xem reports
open output/reports/analysis_results.json
```

## 📋 Hướng dẫn chi tiết từng bước

### Step 1: Court Calibration

**Court calibration là bước quan trọng nhất** - cần thực hiện trước khi chạy analysis.

#### Calibrate thủ công:
```bash
# Calibrate tất cả 4 cameras
python src/court_detection.py --calibrate

# Hoặc calibrate từng camera riêng
python src/court_detection.py --camera san1
python src/court_detection.py --camera san2
python src/court_detection.py --camera san3
python src/court_detection.py --camera san4
```

#### Hướng dẫn calibration:
1. **Click vào 4 góc sân theo thứ tự:**
   - Top-Left (góc trên trái)
   - Top-Right (góc trên phải)
   - Bottom-Left (góc dưới trái)  
   - Bottom-Right (góc dưới phải)

2. **Phím tắt:**
   - `S`: Save calibration
   - `R`: Reset points đã chọn
   - `Q`: Quit calibration

3. **Tips:**
   - Chọn góc sân chính xác (không phải boundary của video)
   - Đảm bảo 4 điểm tạo thành hình chữ nhật
   - Preview window sẽ hiển thị bird's-eye view

#### Kiểm tra calibration:
```bash
# Xem trạng thái calibration
python src/court_detection.py --validate

# Tạo visualization
python src/court_detection.py --visualize
```

### Step 2: Object Detection

#### Chạy detection cho single video:
```bash
# Detection với video cụ thể
python main.py --mode detection --input data_video/san1.mp4

# Giới hạn số frames để test
python main.py --mode detection --input data_video/san1.mp4 --max-frames 100
```

#### Customization detection:
Chỉnh sửa `config/config.yaml`:
```yaml
detection:
  confidence_threshold: 0.5  # Giảm để detect nhiều hơn
  iou_threshold: 0.45        # NMS threshold
  device: "cuda"             # "cpu" nếu không có GPU
```

### Step 3: Multi-Object Tracking

#### Tracking parameters:
```yaml
tracking:
  max_disappeared: 30    # Frames trước khi xóa track
  max_distance: 100      # Max distance for association
  min_hits: 3           # Min detections để confirm
```

#### Chạy tracking riêng:
```bash
# Với detection data có sẵn
python main.py --mode tracking --input output/detection_results.json
```

### Step 4: Analysis & Visualization

#### Chạy analysis trên tracking data:
```bash
python main.py --mode analysis --input output/tracking_data/tracking_results.csv
```

#### Tạo custom visualizations:
```python
from src.visualization import PickleballVisualizer
from src.utils import load_config

config = load_config()
viz = PickleballVisualizer(config)

# Tạo heatmap cho player cụ thể
positions = [(x, y), ...]  # Your position data
viz.create_heatmap(positions, "Player 1 Heatmap", "player1_heatmap.png")
```

## ⚙️ Configuration Guide

### File config chính: `config/config.yaml`

#### Court Settings:
```yaml
court:
  width_meters: 13.41      # Pickleball standard
  height_meters: 6.1       # Pickleball standard
  net_height_meters: 0.914
```

#### Video Settings:
```yaml
video:
  fps: 30                  # Output FPS
  input_videos:           # Paths to input videos
    - "../data_video/san1.mp4"
    - "../data_video/san2.mp4"
    - "../data_video/san3.mp4"
    - "../data_video/san4.mp4"
```

#### Detection Settings:
```yaml
detection:
  model: "yolov8x.pt"           # Model weights
  confidence_threshold: 0.5      # Min confidence
  iou_threshold: 0.45           # NMS threshold
  device: "cuda"                # "cuda" or "cpu"
```

#### Analysis Settings:
```yaml
analysis:
  smoothing:
    enabled: true
    window_size: 5              # Moving average window
  velocity:
    min_distance_threshold: 0.1  # Min distance for velocity calc
```

#### Visualization Settings:
```yaml
visualization:
  colors:
    player_1: [255, 0, 0]      # Red
    player_2: [0, 255, 0]      # Green
    ball: [255, 0, 255]        # Magenta
  heatmap:
    bins: 50                   # Heatmap resolution
    alpha: 0.7                 # Transparency
```

## 🎮 Command Line Options

### Main Script Options:
```bash
python main.py [OPTIONS]

Options:
  --input, -i PATH          Input video file or directory
  --output, -o PATH         Output directory (default: output)
  --config, -c PATH         Config file (default: config/config.yaml)
  --mode MODE              Pipeline mode: full|detection|tracking|analysis
  --max-frames INT         Max frames to process (for testing)
  --skip-calibration       Skip court calibration check
  --verbose, -v            Verbose logging
```

### Examples:
```bash
# Full pipeline với custom config
python main.py --config my_config.yaml --verbose

# Quick test với 100 frames
python main.py --max-frames 100 --skip-calibration

# Process specific video
python main.py --input data_video/san1.mp4 --output results/san1/

# Detection only mode
python main.py --mode detection --input data_video/ --max-frames 500
```

## 📊 Understanding Output

### Directory Structure:
```
output/
├── tracking_data/
│   ├── tracking_results.csv      # Main tracking data
│   └── detection_stats.json      # Detection statistics
├── charts/
│   ├── heatmap_player_1.png      # Individual heatmaps
│   ├── trajectories.png          # Combined trajectories
│   ├── speed_analysis.png        # Speed charts
│   ├── interactive_dashboard.html # Interactive dashboard
│   └── match_summary_report.png  # Summary report
├── reports/
│   ├── analysis_results.json     # Detailed analysis
│   └── performance_metrics.txt   # Performance stats
└── processed_videos/
    └── annotated_video.mp4       # Video with annotations
```

### Tracking Data Format:
```csv
frame_id,timestamp,object_id,class,center_x,center_y,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2
0,0.000,1,player,640.5,360.2,0.95,600,320,680,420
0,0.000,2,player,800.1,400.8,0.87,760,360,840,460
1,0.033,1,player,642.3,358.9,0.94,602,318,682,418
```

### Analysis Results:
```json
{
  "player_analysis": {
    "player_1": {
      "total_distance_meters": 245.8,
      "avg_speed_kmh": 4.93,
      "max_speed_kmh": 11.66,
      "running_pace_min_per_km": 12.18,
      "movement_zones": {
        "left_court": 65.2,
        "right_court": 34.8
      }
    }
  }
}
```

## 🔧 Advanced Usage

### Batch Processing Multiple Matches:
```bash
#!/bin/bash
# Process multiple match directories
for match_dir in matches/*/; do
    echo "Processing $match_dir"
    python main.py --input "$match_dir" --output "results/$(basename $match_dir)"
done
```

### Custom Analysis Script:
```python
#!/usr/bin/env python3
"""Custom analysis script"""

import pandas as pd
from src.analysis import MovementAnalyzer
from src.utils import load_config

# Load data
config = load_config()
analyzer = MovementAnalyzer(config)

# Load tracking data
df = pd.read_csv('output/tracking_data/tracking_results.csv')

# Custom analysis
results = analyzer.analyze_tracking_data(df)

# Extract specific metrics
for player_id, data in results['player_analysis'].items():
    print(f"{player_id}:")
    print(f"  Distance: {data['total_distance_meters']:.1f}m")
    print(f"  Avg Speed: {data['avg_speed_kmh']:.1f} km/h")
    print(f"  Max Speed: {data['max_speed_kmh']:.1f} km/h")
```

### Real-time Processing:
```python
# Process video stream in real-time
import cv2
from src.detection import PickleballDetector
from src.tracking import PickleballTracker

config = load_config()
detector = PickleballDetector(config)
tracker = PickleballTracker(config)

cap = cv2.VideoCapture(0)  # Webcam
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detection
    detections = detector.detect_frame(frame)
    
    # Tracking
    tracked_objects = tracker.update(detections)
    
    # Visualization
    annotated_frame = tracker.visualize_tracking(frame, tracked_objects)
    
    cv2.imshow('Real-time Tracking', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_id += 1

cap.release()
cv2.destroyAllWindows()
```

## 🚨 Troubleshooting

### Common Issues:

#### 1. "Court not calibrated" Error
```bash
# Solution: Calibrate courts first
python src/court_detection.py --calibrate
```

#### 2. Poor Detection Results
```bash
# Lower confidence threshold
# Edit config.yaml:
detection:
  confidence_threshold: 0.3  # Lower from 0.5
```

#### 3. Tracking ID Switches
```bash
# Adjust tracking parameters
tracking:
  max_distance: 50     # Lower from 100
  min_hits: 5         # Higher from 3
```

#### 4. Out of Memory Error
```bash
# Use CPU mode or reduce batch size
detection:
  device: "cpu"
```

#### 5. Slow Processing
```bash
# Process fewer frames for testing
python main.py --max-frames 200

# Or reduce video resolution in preprocessing
```

### Performance Optimization:

#### GPU Optimization:
```python
# Check GPU memory usage
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# Clear GPU cache
torch.cuda.empty_cache()
```

#### Processing Speed Tips:
- Sử dụng lower resolution videos cho testing
- Process subset của frames với `--max-frames`
- Sử dụng SSD storage cho faster I/O
- Close other GPU applications

## 📝 Best Practices

### 1. Court Calibration:
- Calibrate trong điều kiện ánh sáng tốt
- Chọn frame có thể nhìn rõ 4 góc sân
- Double-check preview bird's-eye view
- Save backup của calibration data

### 2. Video Quality:
- Minimum 720p resolution
- Stable camera mounts
- Good lighting conditions
- Minimal camera shake

### 3. Processing Workflow:
- Test với small subset frames trước
- Monitor memory usage
- Save intermediate results
- Backup important data

### 4. Analysis:
- Review tracking quality trước analysis
- Filter noise data
- Validate metrics có hợp lý
- Cross-check với visual inspection

---

## ❓ FAQ

**Q: Pipeline có thể xử lý video 4K không?**
A: Có, nhưng sẽ chậm hơn. Khuyến nghị resize xuống 1080p để xử lý nhanh hơn.

**Q: Có thể track được bao nhiêu players?**
A: Không giới hạn về mặt lý thuyết, nhưng hiệu suất tốt nhất với 2-4 players.

**Q: Pipeline có hoạt động với sports khác không?**
A: Có thể adapt cho tennis, badminton với chỉnh sửa court dimensions.

**Q: Làm sao để improve tracking accuracy?**
A: Fine-tune YOLO model với custom dataset, adjust tracking parameters, sử dụng multiple cameras.

**Q: Có thể chạy real-time không?**
A: Có với GPU mạnh. Cần optimize pipeline để đạt real-time performance.

---

**Phiên bản**: 1.0.0  
**Cập nhật**: October 2, 2025