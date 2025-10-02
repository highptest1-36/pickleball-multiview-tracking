# Pickleball Analysis Project - San4 Video

Phân tích video pickleball san4.mp4 với computer vision và AI tracking.

## 🎯 Các Phiên Bản Script

### 1. `final_san4_analysis.py` - Phiên bản đầy đủ tính năng
- **Tính năng**: Matplotlib visualization, heatmap, thống kê chi tiết
- **Performance**: Chậm nhưng đẹp
- **Sử dụng**: Khi cần visualization đẹp và thống kê chi tiết

### 2. `optimized_san4_analysis.py` - Phiên bản tối ưu ⭐ KHUYẾN NGHỊ
- **Tính năng**: OpenCV visualization, GPU acceleration, 2 cửa sổ riêng biệt
- **Performance**: Nhanh và mượt
- **Sử dụng**: Khuyến nghị cho phân tích thường xuyên
- **Đặc điểm**: 
  - Video gốc + 2D Court riêng biệt
  - GPU support (CUDA)
  - YOLOv8n (nhẹ hơn 6x)
  - Skip frames để tăng tốc

### 3. `ultra_light_san4.py` - Phiên bản siêu nhẹ
- **Tính năng**: Tối thiểu tính năng, tối đa performance  
- **Performance**: Nhanh nhất
- **Sử dụng**: Khi máy yếu hoặc cần real-time processing
- **Đặc điểm**:
  - Xử lý frame 480p
  - Skip 3 frames
  - Visualization đơn giản
  - Memory usage thấp nhất

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