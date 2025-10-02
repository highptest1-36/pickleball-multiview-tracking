# CHANGELOG - Pickleball Analysis Project

## Version 2.0.0 - October 2, 2025 🎯

### 🔥 Major Updates - FIXED COURT TRACKING

#### ✅ New Features
- **`corrected_tracking_san4.py`** - Main script với court orientation đã fix
- **`recalibrate_court.py`** - Interactive court calibration tool
- **Net orientation fixed** - Net nằm ngang theo chiều rộng sân (đúng)
- **Proper camera zones** - Near camera vs Far camera zones
- **Enhanced ball tracking** - Ball có thể di chuyển qua lại 2 sân
- **Court boundary visualization** - Hiển thị trực tiếp trên video

#### 🛠️ Fixed Issues
- ❌ **FIXED**: Net direction (was vertical, now horizontal)
- ❌ **FIXED**: Player zone assignment (Near/Far camera instead of Left/Right)
- ❌ **FIXED**: Court boundary detection accuracy
- ❌ **FIXED**: Ball tracking across court sides
- ❌ **FIXED**: Player assignment logic for far camera players

#### 🚀 Performance Improvements  
- GPU acceleration với CUDA support
- YOLOv8n model cho tốc độ cao
- Optimized frame processing với skip frames
- Memory usage optimization

#### 📊 New Scripts Added
1. **`corrected_tracking_san4.py`** ⭐ - Main recommended script
2. **`strict_tracking_san4.py`** - Strict 4-player tracking
3. **`recalibrate_court.py`** - Court calibration tool
4. **`check_court_boundary.py`** - Validation tool
5. **`validate_court_boundary.py`** - Advanced validation

---

## Version 1.5.0 - October 1, 2025

### ✅ Added
- **`optimized_san4_analysis.py`** - GPU optimized version
- **`ultra_light_san4.py`** - Lightweight version  
- **`final_san4_analysis.py`** - Full-featured version
- Multiple analysis options cho different performance needs

### 🛠️ Improvements
- OpenCV visualization instead of matplotlib
- 2-window layout (Original + 2D Court)
- Real-time statistics panel
- Player trail effects với fade
- Ball trajectory visualization

---

## Version 1.0.0 - September 30, 2025

### 🎉 Initial Release
- **`calibrate_san4.py`** - Basic court calibration
- **`final_san4_analysis.py`** - Initial analysis script
- Basic YOLO detection cho players và ball
- Matplotlib-based visualization
- Court coordinate transformation
- Simple player tracking

---

## 🔄 Migration Guide

### From Version 1.x to 2.0.0

**IMPORTANT**: Court calibration cần làm lại!

```bash
# 1. Recalibrate court (REQUIRED)
python recalibrate_court.py

# 2. Switch to new main script
python corrected_tracking_san4.py  # Thay vì final_san4_analysis.py
```

### Key Changes
- **Court orientation**: Fixed từ vertical net → horizontal net
- **Player zones**: Changed từ Left/Right → Near/Far camera
- **Script naming**: `corrected_tracking_san4.py` là main script mới

---

## 🐛 Known Issues

### Fixed in v2.0.0
- ✅ Court boundary không khớp với thực tế
- ✅ Net direction sai hướng  
- ✅ Player tracking ở sân xa camera
- ✅ Ball không track được qua 2 sân

### Still Working On
- Font rendering warnings (non-critical)
- Occasional tracking ID switches
- Performance optimization cho real-time

---

## 📋 Roadmap

### Version 2.1.0 (Planned)
- [ ] Auto-court detection (không cần manual calibration)
- [ ] Shot detection và analysis
- [ ] Game scoring system
- [ ] Export tracking data to CSV
- [ ] Web-based visualization dashboard

### Version 2.2.0 (Future)
- [ ] Multi-camera support
- [ ] 3D court reconstruction  
- [ ] Player pose estimation
- [ ] Advanced game analytics
- [ ] Real-time streaming support

---

**Maintainer**: AI Assistant + User Collaboration  
**Last Updated**: October 2, 2025