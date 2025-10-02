# 📊 Project Summary - Pickleball Analysis

**Dự án**: Phân tích video pickleball với AI tracking  
**Video**: san4.mp4 (9048 frames @ 30 FPS)  
**Ngày cập nhật**: October 2025  

---

## 🎯 File Chính

### **enhanced_tracking_san4.py** ⭐
**File tracking chính thức** - Stable, accurate, zone-locked tracking

**Đặc điểm**:
- ✅ YOLO11x model (109MB) - High accuracy
- ✅ Stable tracking với distance matching (< 1.5m)
- ✅ Zone-locked: P1,P2 ở Sân 1 | P3,P4 ở Sân 2
- ✅ Players KHÔNG nhảy sang sân đối diện
- ✅ Max 2 players mỗi bên
- ✅ KHÔNG nhấp nháy (no flickering)
- ✅ Ball tracking với trajectory prediction
- ✅ Real-time ~25 FPS (CUDA)

**Cách chạy**:
```powershell
Set-Location C:\Users\highp\pickerball\video\pickleball_analysis
python enhanced_tracking_san4.py
```

HOẶC:
```powershell
python main.py
```

---

## 🛠️ Calibration Tools (Chạy 1 lần khi setup)

### 1. **multi_point_selector.py**
Chọn 8 điểm viền sân để tạo yellow polygon

**Sử dụng**:
```bash
python multi_point_selector.py
```
- Click 8 điểm theo viền sân (theo chiều kim đồng hồ)
- Drag để điều chỉnh vị trí điểm
- Press 's' để save → tạo `court_calibration_san4.json`

### 2. **net_selector.py**
Đánh dấu 2 điểm trên đường lưới

**Sử dụng**:
```bash
python net_selector.py
```
- Click 2 điểm trên đường lưới
- Press 's' để save → cập nhật `court_calibration_san4.json`

---

## 📁 Generated Files

### **court_calibration_san4.json**
Chứa thông tin calibration:
- `yellow_polygon`: 8 điểm viền sân (user-selected)
- `image_points`: 4 góc sân (fitted rectangle)
- `net_line`: 2 điểm đường lưới
- `homography`: Matrix transform image ↔ court coordinates
- `court_width`: 6.1m
- `court_length`: 13.41m
- `court_split`: "by_width" (X-axis)

---

## 🎨 Visualization

### Court Layout
```
┌─────────────────────┬─────────────────────┐
│                     │                     │
│      Sân 1          │      Sân 2          │
│   (Left/Near)       │   (Right/Far)       │
│                     │                     │
│   P1 🔴 RED         │   P3 🔵 BLUE        │
│   P2 🟢 GREEN       │   P4 🟡 YELLOW      │
│                     │                     │
│   0 - 3.05m         │   3.05m - 6.10m     │
│                     │                     │
└─────────────────────┴─────────────────────┘
         ⚪ Net Line (White)
```

### Màu sắc
- **P1**: 🔴 RED (0, 0, 255)
- **P2**: 🟢 GREEN (0, 255, 0)
- **P3**: 🔵 BLUE (255, 0, 0)
- **P4**: 🟡 YELLOW (0, 255, 255)
- **Ball**: 🎾 Cyan (0, 255, 255)
- **Court boundary**: 🟨 Yellow polygon
- **Net**: ⚪ White line
- **Trails**: Fade effect (0.8 alpha)

---

## 🔧 Tracking Logic

### Zone Assignment
```python
# Split court theo WIDTH (X-axis)
if x <= 3.05m:
    zone = 'near'  # Sân 1 (Left)
    available_ids = [1, 2]  # P1, P2
else:
    zone = 'far'   # Sân 2 (Right)
    available_ids = [3, 4]  # P3, P4
```

### Matching Algorithm
```python
1. Detect players trong frame
2. Transform to court coordinates
3. Classify vào zone (near/far)
4. Sort by confidence (cao xuống thấp)
5. Take top 2 per zone

For each detection:
    - Tìm player ID cũ trong zone (distance < 1.5m)
    - Nếu tìm thấy → Update position (giữ ID)
    - Nếu không → Assign vào slot trống
    - Nếu quá 2 người → Deactivate người confidence thấp nhất
```

### Key Features
- **Distance threshold**: < 1.5m (match với player cũ)
- **Adaptive confidence**: 0.40 (near) → 0.20 (far)
- **Zone lock**: Players không bao giờ nhảy sang sân đối diện
- **Stable ID**: Giữ ID nếu movement < 1.5m/frame
- **No flickering**: Không clear mỗi frame, chỉ update position

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model | YOLO11x (109MB) |
| FPS (CUDA) | ~25 |
| FPS (CPU) | ~10 |
| Resolution | 1920x1080 (Full HD) |
| GPU Memory | ~2GB |
| RAM | ~4GB |
| Detection classes | Person (class 0), Sports ball (class 32) |
| Confidence threshold | 0.20 - 0.40 (adaptive) |
| IOU threshold | 0.5 |

---

## 📋 Dependencies

```txt
ultralytics>=8.0.0      # YOLO11x model
opencv-python>=4.8.0     # Video processing
numpy>=1.24.0            # Array operations
torch>=2.0.0             # Deep learning backend
scipy>=1.11.0            # Distance calculations (cdist)
```

**Install**:
```bash
pip install ultralytics opencv-python numpy torch scipy
```

---

## 🚀 Quick Start

```powershell
# 1. Setup (chạy 1 lần)
cd C:\Users\highp\pickerball\video\pickleball_analysis
python multi_point_selector.py  # Chọn 8 điểm viền sân
python net_selector.py           # Chọn 2 điểm lưới

# 2. Run tracking
python main.py
# HOẶC:
python enhanced_tracking_san4.py

# 3. Quit
Press 'q' trong video window
```

---

## 🗑️ Cleaned Up Files

Các file tracking cũ đã được xóa:
- ❌ `stable_tracking_san4.py`
- ❌ `strict_tracking_san4.py`
- ❌ `stable_reid_tracking_san4.py`
- ❌ `fixed_tracking_san4.py`
- ❌ `corrected_tracking_san4.py`
- ❌ `advanced_tracking_san4.py`
- ❌ `optimized_san4_analysis.py`
- ❌ `ultra_light_san4.py`
- ❌ `final_san4_analysis.py`
- ❌ `calibrate_san4.py`
- ❌ `test_san4_calibration.py`
- ❌ `run_san4_analysis.py`
- ❌ `san4_2d_viewer.py`
- ❌ Các file demo/test/analyze cũ

**Lý do xóa**: Duplicate logic, outdated, replaced by `enhanced_tracking_san4.py`

---

## 📝 Notes

- ⚠️ **QUAN TRỌNG**: Phải chạy calibration tools (multi_point_selector.py + net_selector.py) trước lần đầu
- ⚠️ File `court_calibration_san4.json` bắt buộc phải tồn tại
- ⚠️ Video `san4.mp4` phải ở `C:\Users\highp\pickerball\video\data_video\`
- 💡 Model `yolo11x.pt` sẽ tự động download lần đầu chạy
- 💡 Sử dụng CUDA nếu có GPU để tăng tốc (25 FPS vs 10 FPS)
- 💡 Press 'q' để quit video playback

---

## 🎯 Future Improvements

- [ ] Export tracking data to CSV/JSON
- [ ] Heatmap generation
- [ ] Shot detection (smash, lob, drop shot)
- [ ] Rally analysis
- [ ] Speed/distance metrics
- [ ] Multi-camera support
- [ ] Dashboard visualization

---

**Status**: ✅ Production Ready  
**Last Update**: October 2025  
**Maintained by**: Enhanced Tracking System V3
