# Technical Specifications - Pickleball Video Analysis Pipeline

## 📋 Tổng quan hệ thống

Pipeline này được thiết kế để phân tích video pickleball từ 4 góc camera và tạo ra các insights chi tiết về:
- Tracking chuyển động người chơi và bóng
- Phân tích vận tốc, quãng đường di chuyển
- Heatmap và trajectory visualization
- Báo cáo hiệu suất chi tiết

## 🏗️ Kiến trúc hệ thống

### Pipeline Architecture

```
Input Videos (4 cameras) 
    ↓
Court Detection & Homography
    ↓
YOLO Object Detection
    ↓
Multi-Object Tracking
    ↓
Movement Analysis
    ↓
Visualization & Reports
```

### Module Dependencies

```
main.py
├── src/court_detection.py    (OpenCV, NumPy)
├── src/detection.py          (YOLOv11x, PyTorch)
├── src/tracking.py           (Custom tracking algorithm)
├── src/analysis.py           (Pandas, NumPy, SciPy)
├── src/visualization.py      (Matplotlib, Plotly, Seaborn)
└── src/utils.py             (Core utilities)
```

## 🔧 Chi tiết kỹ thuật từng module

### 1. Court Detection Module (`court_detection.py`)

**Chức năng chính:**
- Phát hiện 4 góc sân pickleball trong video
- Tính toán ma trận homography transformation
- Chuyển đổi camera view → bird's-eye view

**Thuật toán:**
- Manual calibration với mouse interaction
- Perspective transformation sử dụng `cv2.getPerspectiveTransform()`
- Mapping từ pixel space → real-world coordinates (meters)

**Input:** Video frames từ 4 camera
**Output:** Homography matrices, bird's-eye view frames

### 2. Detection Module (`detection.py`)

**Chức năng chính:**
- Detect người chơi và bóng sử dụng YOLOv11x
- Post-processing và filtering detections
- Export detection results

**Thuật toán:**
- YOLOv11x pretrained trên COCO dataset
- Classes: person (0), sports ball (32)
- Confidence thresholding và NMS
- Area-based và position-based filtering

**Input:** Video frames (original hoặc transformed)
**Output:** Detection data với bounding boxes, confidence scores

**Performance:**
- GPU: ~30-60 FPS (depending on hardware)
- CPU: ~5-15 FPS
- Memory: ~2-4GB VRAM (GPU mode)

### 3. Tracking Module (`tracking.py`)

**Chức năng chính:**
- Track multiple objects qua consecutive frames
- Gán unique ID cho mỗi object
- Handle occlusions và re-identification

**Thuật toán:**
- Centroid-based tracking với Kalman filtering
- Hungarian algorithm cho object association
- Track management (birth, death, re-identification)

**Parameters:**
- `max_disappeared`: 30 frames
- `max_distance`: 100 pixels
- `min_hits`: 3 detections để confirm track

**Input:** Frame-by-frame detections
**Output:** Continuous tracks với unique IDs

### 4. Analysis Module (`analysis.py`)

**Chức năng chính:**
- Tính toán movement metrics (speed, acceleration, distance)
- Phân tích court coverage và movement patterns
- Generate performance statistics

**Thuật toán:**
- Finite difference cho velocity calculation
- Moving average smoothing
- Zone-based analysis
- Statistical aggregation

**Metrics được tính:**
- Total distance covered (meters)
- Average/max speed (m/s, km/h)
- Running pace (min/km)
- Direction changes
- Court zone coverage (%)
- Player interaction analysis

### 5. Visualization Module (`visualization.py`)

**Chức năng chính:**
- Tạo heatmaps và trajectory plots
- Speed analysis charts
- Interactive dashboards
- Match summary reports

**Tools:**
- Matplotlib: Static plots và charts
- Plotly: Interactive dashboards
- Seaborn: Statistical visualizations
- OpenCV: Video annotations

## 📊 Data Flow & Formats

### Tracking Data Format (CSV)

```csv
frame_id,timestamp,object_id,class,center_x,center_y,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2
0,0.000,1,player,640.5,360.2,0.95,600,320,680,420
0,0.000,2,player,800.1,400.8,0.87,760,360,840,460
1,0.033,1,player,642.3,358.9,0.94,602,318,682,418
...
```

### Analysis Results Format (JSON)

```json
{
  "player_analysis": {
    "player_1": {
      "duration_seconds": 180.0,
      "total_distance_meters": 245.8,
      "avg_speed_ms": 1.37,
      "max_speed_ms": 3.24,
      "avg_speed_kmh": 4.93,
      "max_speed_kmh": 11.66,
      "running_pace_min_per_km": 12.18,
      "direction_changes": 45,
      "movement_zones": {
        "left_court": 65.2,
        "right_court": 34.8,
        "front_court": 58.1,
        "back_court": 41.9
      }
    }
  },
  "match_statistics": {
    "match_duration": 180.0,
    "total_players": 4,
    "avg_players_on_court": 3.8,
    "ball_in_play_percentage": 78.5
  }
}
```

## ⚡ Performance Specifications

### System Requirements

**Minimum:**
- CPU: Intel i5-8400 hoặc AMD Ryzen 5 2600
- RAM: 8GB
- Storage: 10GB available space
- GPU: Optional (GTX 1060 hoặc tương đương)

**Recommended:**
- CPU: Intel i7-10700K hoặc AMD Ryzen 7 3700X
- RAM: 16GB
- Storage: 20GB SSD space
- GPU: RTX 3070 hoặc tương đương

### Processing Speed

| Stage | GPU Mode | CPU Mode | Bottleneck |
|-------|----------|----------|------------|
| Detection | 30-60 FPS | 5-15 FPS | GPU compute |
| Tracking | 100+ FPS | 50+ FPS | CPU bound |
| Analysis | 500+ FPS | 200+ FPS | I/O bound |
| Visualization | - | Variable | Rendering |

### Memory Usage

- **Detection**: 2-4GB VRAM (GPU), 1-2GB RAM (CPU)
- **Tracking**: 500MB-1GB RAM
- **Analysis**: 200-500MB RAM
- **Peak total**: 4-6GB total system memory

## 🔄 Configuration System

### Config.yaml Structure

```yaml
# Court specifications
court:
  width_meters: 13.41
  height_meters: 6.1
  net_height_meters: 0.914

# Detection settings
detection:
  model: "yolov8x.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: "cuda"  # or "cpu"

# Tracking settings
tracking:
  algorithm: "bytetrack"
  max_disappeared: 30
  max_distance: 100
  min_hits: 3

# Analysis settings
analysis:
  smoothing:
    enabled: true
    window_size: 5
  velocity:
    calculation_method: "finite_difference"
    min_distance_threshold: 0.1
```

## 🧪 Testing & Validation

### Unit Tests

```bash
# Test individual modules
python -m pytest tests/test_detection.py
python -m pytest tests/test_tracking.py
python -m pytest tests/test_analysis.py
```

### Integration Tests

```bash
# Test full pipeline
python tests/test_pipeline.py
```

### Performance Benchmarks

```bash
# Benchmark processing speed
python benchmarks/speed_benchmark.py

# Memory profiling
python -m memory_profiler main.py
```

## 🚨 Error Handling & Logging

### Log Levels

- **DEBUG**: Detailed tracing information
- **INFO**: General information about pipeline progress
- **WARNING**: Potential issues that don't stop execution
- **ERROR**: Errors that stop execution
- **CRITICAL**: Severe errors that may crash the system

### Error Recovery

- **Video file errors**: Skip corrupted frames, continue processing
- **Detection failures**: Return empty detections, log warning
- **Tracking losses**: Mark as lost track, attempt re-identification
- **Memory issues**: Batch processing, garbage collection

## 📈 Scalability Considerations

### Horizontal Scaling

- Process multiple videos in parallel
- Distribute detection across multiple GPUs
- Cloud deployment với containerization

### Optimization Opportunities

1. **Model Optimization**:
   - TensorRT optimization cho YOLO
   - Model quantization (FP16, INT8)
   - Custom model training

2. **Algorithm Improvements**:
   - Advanced tracking algorithms (ByteTrack, OC-SORT)
   - Kalman filter tuning
   - Multi-camera fusion

3. **Infrastructure**:
   - GPU clusters
   - Real-time streaming processing
   - Edge deployment

## 🔒 Security & Privacy

### Data Handling

- Video data processed locally by default
- No cloud upload unless explicitly configured
- Temporary files cleaned up after processing

### Privacy Considerations

- Player anonymization options
- Face blurring capabilities
- GDPR compliance features

---

**Phiên bản tài liệu**: 1.0.0  
**Ngày cập nhật**: October 2, 2025  
**Tác giả**: AI Assistant