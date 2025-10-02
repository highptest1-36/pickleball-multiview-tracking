# Pickleball Analysis Project - San4 Video

Phân tích video pickleball san4.mp4 với computer vision, AI tracking và court calibration chính xác.

## 🎯 Các Phiên Bản Script (Mới Nhất)

### 1. `corrected_tracking_san4.py` - ⭐ KHUYẾN NGHỊ MỚI NHẤT ⭐
- **Tính năng**: Fixed court orientation, proper net direction, accurate 4-player tracking
- **Performance**: Tối ưu và chính xác nhất
- **Sử dụng**: Phiên bản chính thức sau khi fix court calibration
- **Đặc điểm**: 
  - ✅ Net nằm ngang đúng hướng
  - ✅ Near/Far camera zones chính xác
  - ✅ P1,P2 = Near camera | P3,P4 = Far camera
  - ✅ Ball tracking qua lại 2 sân
  - ✅ Court boundary visualization trực tiếp trên video

### 2. `strict_tracking_san4.py` - Phiên bản nghiêm ngặt
- **Tính năng**: STRICT 4-player system, players không được chuyển side
- **Performance**: Stable tracking với fixed IDs
- **Sử dụng**: Khi cần tracking nghiêm ngặt không đổi
- **Đặc điểm**:
  - 🚫 P1,P2 chỉ ở LEFT side
  - 🚫 P3,P4 chỉ ở RIGHT side
  - ✅ Court zones visualization
  - ✅ Player trails với fade effects

### 3. `optimized_san4_analysis.py` - Phiên bản tối ưu GPU
- **Tính năng**: OpenCV visualization, GPU acceleration, 2 cửa sổ riêng biệt
- **Performance**: Nhanh với GPU support
- **Sử dụng**: Khi cần performance cao với GPU
- **Đặc điểm**: 
  - Video gốc + 2D Court riêng biệt
  - CUDA support (YOLOv8n)
  - Skip frames để tăng tốc

### 4. `ultra_light_san4.py` - Phiên bản siêu nhẹ
- **Tính năng**: Minimal features, maximum performance  
- **Performance**: Fastest possible
- **Sử dụng**: Máy yếu hoặc real-time processing
- **Đặc điểm**:
  - 480p processing
  - Skip 3 frames
  - Simple visualization

## 🛠️ Cài Đặt và Setup

### Prerequisites
```bash
pip install ultralytics opencv-python numpy matplotlib torch scipy
```

### Bước 1: Court Calibration (QUAN TRỌNG!)
```bash
# Calibrate court cho san4.mp4 (bắt buộc làm đầu tiên)
python recalibrate_court.py
```
**Hướng dẫn calibration:**
1. Script sẽ mở video san4.mp4
2. Click 4 góc sân theo thứ tự: **Top-Left → Top-Right → Bottom-Right → Bottom-Left**
3. Nhấn **'s'** để save, **'r'** để reset, **'q'** để quit
4. File `court_calibration_san4.json` sẽ được tạo

### Bước 2: Chạy Analysis Script

**Khuyến nghị (Mới nhất)**:
```bash
python corrected_tracking_san4.py
```

**Các phiên bản khác**:
```bash
# Strict tracking (fixed player sides)
python strict_tracking_san4.py

# GPU optimized với 2 cửa sổ
python optimized_san4_analysis.py

# Ultra light version
python ultra_light_san4.py
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