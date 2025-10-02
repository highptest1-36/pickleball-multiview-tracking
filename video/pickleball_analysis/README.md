# Pickleball Analysis Project - San4 Video

Phân tích video pickleball san4.mp4 với computer vision, AI tracking và court calibration chính xác.

## 🎯 Script Chính

### `enhanced_tracking_san4.py` - ⭐ PHIÊN BẢN CHÍNH THỨC ⭐
- **Model**: YOLO11x (109MB) - High accuracy detection
- **Tính năng**:
  - ✅ Stable tracking với distance matching (không nhấp nháy)
  - ✅ Zone-locked players: P1,P2 ở Sân 1 | P3,P4 ở Sân 2
  - ✅ Max 2 players mỗi bên sân
  - ✅ Người chơi KHÔNG nhảy sang sân đối diện
  - ✅ Ball tracking với trajectory prediction
  - ✅ Court split theo WIDTH (X-axis): Left/Right sides
  - ✅ Adaptive confidence thresholds (0.20-0.40)
  - ✅ Player trails với fade effects
  - ✅ Real-time visualization (~25 FPS)

- **Court Calibration**:
  - 🟨 Yellow polygon: 8 điểm viền sân (user-selected)
  - 🎾 Net line: 2 điểm đánh dấu lưới
  - 📐 Homography transform: Image → Court coordinates

- **Player Colors**:
  - P1: 🔴 RED (Sân 1)
  - P2: 🟢 GREEN (Sân 1)
  - P3: 🔵 BLUE (Sân 2)
  - P4: 🟡 YELLOW (Sân 2)

- **Tracking Logic**:
  - Mỗi frame: Detect → Classify zone → Distance match → Update
  - Distance threshold: < 1.5m (match với player ID cũ)
  - Nếu không match: Gán vào slot trống (theo confidence)
  - Lost tracking: Deactivate player

## 🛠️ Cài Đặt và Sử Dụng

### Bước 1: Cài đặt dependencies
```bash
pip install ultralytics opencv-python numpy torch scipy
```

### Bước 2: Court Calibration (LÀM 1 LẦN DUY NHẤT)

**2.1. Chọn 8 điểm viền sân (Yellow Polygon)**
```bash
python multi_point_selector.py
```
- Click 8 điểm theo viền sân (theo chiều kim đồng hồ)
- Có thể drag-drop để điều chỉnh
- Nhấn **'s'** để save → tạo file `court_calibration_san4.json`

**2.2. Đánh dấu lưới (Net Line)**
```bash
python net_selector.py
```
- Click 2 điểm trên đường lưới
- Nhấn **'s'** để save → cập nhật `court_calibration_san4.json`

### Bước 3: Chạy Tracking

**Cách 1: Dùng main.py (Khuyến nghị)**
```bash
python main.py
```

**Cách 2: Chạy trực tiếp**
```bash
python enhanced_tracking_san4.py
```

**Cách 3: Từ bất kỳ đâu**
```powershell
Set-Location C:\Users\highp\pickerball\video\pickleball_analysis
python enhanced_tracking_san4.py
```

## ⚙️ Troubleshooting và Optimization

### GPU Acceleration
- Cài CUDA để sử dụng GPU acceleration
- Script sẽ tự động detect và sử dụng GPU nếu có
- Nếu có vấn đề GPU: Thêm `torch.cuda.is_available = lambda: False`

### Memory Optimization  
- `ultra_light_san4.py`: Ít RAM nhất
- `corrected_tracking_san4.py`: Cân bằng RAM/Performance
- `optimized_san4_analysis.py`: Nhiều RAM cho tính năng

### CPU Optimization
```bash
# Set environment variable trước khi chạy
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python script_name.py
```

### Common Issues
- **Video không tìm thấy**: Kiểm tra đường dẫn `data_video/san4.mp4`
- **Court boundary sai**: Chạy lại `recalibrate_court.py`
- **Tracking không chính xác**: Đảm bảo court calibration đúng

## 📊 Tính Năng

### Tất cả phiên bản có:
- ✅ YOLO object detection (person + ball)
- ✅ Court coordinate transformation
- ✅ Real-time tracking visualization
- ✅ Player position smoothing
- ✅ Ball trajectory tracking
- ✅ Court boundary overlay

### Tính năng nâng cao:
- 🎨 Movement heatmap (`optimized_san4_analysis.py`)
- 📈 4-panel dashboard (`final_san4_analysis.py`) 
- 🔄 Advanced trail effects
- 📊 Real-time statistics
- 🎮 Interactive controls

## 🎮 Điều khiển

- **'q'**: Thoát (cho OpenCV versions)
- **Ctrl+C**: Thoát (cho tất cả versions)
- **Mouse**: Click để calibrate court
- **'s'**: Save calibration
- **'r'**: Reset calibration

## 📁 Files Structure

### Core Scripts
- `corrected_tracking_san4.py` - **Main script (khuyến nghị)** ⭐
- `strict_tracking_san4.py` - Strict 4-player tracking
- `recalibrate_court.py` - Court calibration tool **Bắt buộc chạy trước**

### Support Scripts  
- `optimized_san4_analysis.py` - GPU optimized version
- `ultra_light_san4.py` - Lightweight version
- `check_court_boundary.py` - Validation tool

### Data Files
- `court_calibration_san4.json` - Court calibration data (auto-generated)
- `yolov8n.pt` - YOLO model weights (auto-downloaded)

## 🔧 Advanced Configuration

### Court Dimensions
Sân pickleball chuẩn:
- **Width**: 6.1m (20 feet)
- **Length**: 13.41m (44 feet)
- **Net height**: 0.91m (3 feet)

### YOLO Settings
- **Person confidence**: 0.4+
- **Ball confidence**: 0.15+ (lower for better detection)
- **Model**: YOLOv8n (fastest) hoặc YOLOv8x (most accurate)

### Performance Tuning
```python
# Trong script, có thể điều chỉnh:
skip_frames = 2        # Tăng để faster, giảm để more accurate
process_size = 640     # Giảm để faster processing
confidence_threshold = 0.3  # Điều chỉnh detection sensitivity
```

## 📈 Performance Benchmarks

| Script | FPS | Memory | GPU | Accuracy |
|--------|-----|--------|-----|----------|
| `corrected_tracking_san4.py` | 25-30 | Medium | Yes | High |
| `strict_tracking_san4.py` | 20-25 | Medium | Yes | High |  
| `optimized_san4_analysis.py` | 30-35 | High | Yes | High |
| `ultra_light_san4.py` | 40-50 | Low | Yes | Medium |

## 🎬 Demo Output

Kết quả hiển thị:
- **Video gốc** với court boundaries
- **Player tracking** với ID và trails
- **Ball tracking** với trajectory
- **Real-time statistics** panel
- **Court zones** visualization

## 📞 Support

### Debugging Steps
1. Kiểm tra video path: `data_video/san4.mp4`
2. Chạy court calibration: `python recalibrate_court.py`
3. Test với script cơ bản: `python corrected_tracking_san4.py`
4. Kiểm tra dependencies: `pip install -r requirements.txt`

### Known Issues
- Font rendering warnings: Không ảnh hướng tính năng
- CUDA out of memory: Sử dụng CPU hoặc giảm batch size
- Court boundary không khớp: Recalibrate court

---

## 🚀 Quick Start

```bash
# 1. Setup
pip install ultralytics opencv-python numpy matplotlib torch scipy

# 2. Calibrate court (QUAN TRỌNG!)
python recalibrate_court.py

# 3. Run analysis
python corrected_tracking_san4.py
```

**Tác giả:** AI Assistant + User Collaboration  
**Ngày cập nhật:** October 2, 2025  
**Phiên bản:** 2.0.0 - Fixed Court Tracking