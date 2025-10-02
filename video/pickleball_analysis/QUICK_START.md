# 🚀 Quick Start Guide - Pickleball Analysis

## ⚡ Fast Setup (10 phút)

```bash
# 1. Install dependencies
pip install ultralytics opencv-python numpy torch scipy

# 2. Navigate to project folder
cd C:\Users\highp\pickerball\video\pickleball_analysis

# 3. Calibrate court (LÀM 1 LẦN DUY NHẤT!)

# 3a. Chọn 8 điểm viền sân
python multi_point_selector.py
# - Click 8 điểm theo viền sân
# - Drag để điều chỉnh
# - Press 's' to save

# 3b. Đánh dấu lưới
python net_selector.py
# - Click 2 điểm trên đường lưới
# - Press 's' to save

# 4. Run tracking
python main.py
# HOẶC: python enhanced_tracking_san4.py
# Press 'q' to quit
```

## 📋 Files Quan Trọng

| File | Mục đích | Ghi chú |
|------|----------|---------|
| `multi_point_selector.py` | Chọn 8 điểm viền sân | Chạy 1 lần khi setup |
| `net_selector.py` | Đánh dấu lưới | Chạy 1 lần khi setup |
| `enhanced_tracking_san4.py` | **MAIN TRACKING SCRIPT** | ⭐ File chính |
| `main.py` | Entry point wrapper | Gọi enhanced_tracking_san4.py |
| `court_calibration_san4.json` | Calibration data | Auto-generated |

## 🎯 Tracking Rules

- **Sân 1 (Left/Near)**: P1 🔴 RED, P2 🟢 GREEN
- **Sân 2 (Right/Far)**: P3 🔵 BLUE, P4 🟡 YELLOW
- **Max 2 players** mỗi bên sân
- **Players stay on their side** - không nhảy sang đối diện
- **Stable tracking** - không nhấp nháy, không "nối nối"

## 🔧 Common Issues & Fixes

### ❌ "court_calibration_san4.json not found"
```bash
# Chạy calibration tools
python multi_point_selector.py  # Chọn 8 điểm viền sân
python net_selector.py          # Chọn 2 điểm lưới
```

### ❌ "Video not found"
```bash
# Check video exists
ls C:\Users\highp\pickerball\video\data_video\san4.mp4
```

### ❌ "yolo11x.pt not found"
```bash
# Model sẽ tự động download lần đầu chạy
# Hoặc download thủ công từ: https://github.com/ultralytics/assets/releases
```

### ❌ "Players flickering/nhấp nháy"
```bash
# Enhanced tracking đã fix vấn đề này!
# Sử dụng distance matching thay vì clear mỗi frame
```

### ❌ "CUDA out of memory"
```bash
# Sử dụng CPU mode
$env:CUDA_VISIBLE_DEVICES=""
python enhanced_tracking_san4.py
```

## 🎯 Expected Results

- **🟨 Yellow polygon** = Court boundary (8 points)
- **⚪ White line** = Net (horizontal)
- **🔴 Red box (P1)** = Player 1 (Sân 1)
- **🟢 Green box (P2)** = Player 2 (Sân 1)
- **🔵 Blue box (P3)** = Player 3 (Sân 2)
- **🟡 Yellow box (P4)** = Player 4 (Sân 2)
- **🎾 Ball tracking** = Cyan box (hiện khi có detect)
- **Trails** = Movement history (fade effect)

## 📊 Performance

- **Model**: YOLO11x (109MB) - High accuracy
- **FPS**: ~25 FPS (CUDA) / ~10 FPS (CPU)
- **Resolution**: Full HD (1920x1080)
- **Tracking**: Distance matching (< 1.5m threshold)
- **Memory**: ~2GB GPU / ~4GB RAM

## 🎮 Controls

- **'q'** = Quit video playback
- **Ctrl+C** = Force quit
- **'s'** = Save (trong calibration tools)
- **'r'** = Reset (trong calibration tools)
- **Mouse** = Click/drag điểm (trong calibration tools)

## 🚀 Recommended Command

```powershell
# Từ bất kỳ đâu:
Set-Location C:\Users\highp\pickerball\video\pickleball_analysis
python enhanced_tracking_san4.py

# HOẶC dùng main.py:
python main.py
```

---

**Need help?** Check `README.md` for full documentation.