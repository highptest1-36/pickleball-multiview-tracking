# 🚀 Quick Start Guide - Pickleball Analysis v2.0.0

## ⚡ Fast Setup (5 minutes)

```bash
# 1. Install dependencies
pip install ultralytics opencv-python numpy matplotlib torch scipy

# 2. Navigate to project folder
cd pickleball_analysis

# 3. Calibrate court (REQUIRED FIRST!)
python recalibrate_court.py
# Click 4 corners: Top-Left → Top-Right → Bottom-Right → Bottom-Left
# Press 's' to save

# 4. Run analysis
python corrected_tracking_san4.py
# Press 'q' to quit
```

## 📋 Script Cheat Sheet

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `recalibrate_court.py` | **REQUIRED FIRST** - Set court boundaries | Always run này trước |
| `corrected_tracking_san4.py` | **MAIN SCRIPT** - Best tracking | ⭐ Khuyến nghị |
| `strict_tracking_san4.py` | Fixed 4-player zones | Khi cần strict zones |
| `optimized_san4_analysis.py` | GPU optimized | Máy mạnh có GPU |
| `ultra_light_san4.py` | Fastest performance | Máy yếu |

## 🔧 Common Issues & Fixes

### ❌ "Video not found"
```bash
# Check video path
ls data_video/san4.mp4  # Should exist
```

### ❌ "Court boundary wrong"
```bash
# Recalibrate court
python recalibrate_court.py
```

### ❌ "CUDA out of memory"
```bash
# Use CPU mode
$env:CUDA_VISIBLE_DEVICES=""
python corrected_tracking_san4.py
```

### ❌ "No players detected"
```bash
# Check court calibration file exists
ls court_calibration_san4.json
```

## 🎯 Expected Results

- **Green boundary** = Court edges
- **White line** = Net (horizontal)
- **Colored boxes** = Players (P1-P4)
- **Yellow box** = Ball
- **Trails** = Movement history

## 📊 Performance Tips

- **Fastest**: `ultra_light_san4.py`
- **Most accurate**: `corrected_tracking_san4.py`  
- **GPU boost**: Set `$env:CUDA_VISIBLE_DEVICES="0"`
- **Memory save**: Close other apps

## 🎮 Controls

- **'q'** = Quit
- **Ctrl+C** = Force quit
- **Mouse click** = Calibrate points
- **'s'** = Save calibration
- **'r'** = Reset calibration

---

**Need help?** Check `README.md` for full documentation.