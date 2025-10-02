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

## �️ Cài Đặt

```bash
pip install ultralytics opencv-python numpy matplotlib torch
```

## 🚀 Sử dụng

1. **Calibrate court** (chỉ làm 1 lần):
```bash
python calibrate_san4.py
```

2. **Chọn phiên bản phù hợp**:

**Máy mạnh + cần đẹp**:
```bash
python final_san4_analysis.py
```

**Khuyến nghị (cân bằng)**:
```bash
python optimized_san4_analysis.py
```

**Máy yếu + cần nhanh**:
```bash
python ultra_light_san4.py
```

## ⚙️ Tối Ưu Performance

### GPU Acceleration
- Cài CUDA để sử dụng GPU
- Script sẽ tự động detect và sử dụng

### Memory Optimization  
- `ultra_light_san4.py`: Dùng ít RAM nhất
- `optimized_san4_analysis.py`: Cân bằng RAM/Performance
- `final_san4_analysis.py`: Dùng nhiều RAM nhất

### CPU Optimization
```bash
# Set environment variable trước khi chạy
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python script_name.py
```

## 📊 Tính Năng

### Tất cả phiên bản có:
- ✅ Tracking người chơi trong sân
- ✅ Tracking bóng trong sân  
- ✅ Tối đa 2 người/bên sân
- ✅ Position smoothing
- ✅ 2D court visualization
- ✅ Real-time statistics

### Chỉ phiên bản cao cấp:
- 🎨 Movement heatmap (`final_san4_analysis.py`)
- 📈 4-panel dashboard (`final_san4_analysis.py`) 
- 🔄 Advanced trail effects (`optimized_san4_analysis.py`, `final_san4_analysis.py`)

## 🎮 Điều khiển

- **'q'**: Thoát (cho OpenCV versions)
- **Ctrl+C**: Thoát (cho tất cả versions)
- **Mouse**: Đóng cửa sổ để thoát

## 📁 Files

- `calibrate_san4.py` - Công cụ calibrate sân cho san4.mp4
- `court_calibration_san4.json` - Dữ liệu calibration (tự động tạo)
- `final_san4_analysis.py` - Phiên bản đầy đủ tính năng
- `optimized_san4_analysis.py` - Phiên bản tối ưu khuyến nghị ⭐
- `ultra_light_san4.py` - Phiên bản siêu nhẹ

## 🔧 Troubleshooting

### Lỗi CUDA/GPU:
```bash
# Chỉ định CPU
import torch
torch.cuda.is_available = lambda: False
```

### Lỗi OpenMP:
```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### Video không tìm thấy:
- Kiểm tra đường dẫn trong script
- Đảm bảo `san4.mp4` ở đúng folder `data_video`

---

## 🎬 Demo Screenshots

Hình ảnh đính kèm cho thấy:
- **Bên trái**: Video gốc san4.mp4 với frame tracking
- **Bên phải**: 2D Court view với players và ball tracking
- **Thống kê**: Real-time game statistics và player positioning