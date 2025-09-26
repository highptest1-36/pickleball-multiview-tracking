# Pickleball Multi-View Tracking & Analysis System 🏓

Advanced computer vision system for pickleball analysis using multi-camera setup with 3D tracking, video stitching, and synchronized playback.

## 🚀 Features

### 📹 Multi-View Video Processing
- **4-Camera System**: Simultaneous processing from 4 different angles
- **360° View Stitching**: Seamless video combining from multiple perspectives  
- **Frame Synchronization**: Smart alignment using timestamp patterns
- **Real-time Processing**: Up to 35 FPS processing speed

### 🎯 Object Detection & Tracking
- **YOLO11 Integration**: Advanced object detection for players and ball
- **Player Tracking**: Maximum 4 players per court (2 per side)
- **Movement Trails**: Historical position tracking with visualization
- **Ball Trajectory**: Real-time ball tracking and trajectory prediction

### 🎨 Visualization Systems
- **God View**: Top-down 3D court visualization
- **Side View**: Profile perspective with height tracking
- **Split-Screen**: Multi-view + 3D visualization combined
- **Picture-in-Picture**: Main camera with smaller auxiliary views

### 🔄 Synchronization Technology
- **Reference Camera**: Camera 1 as master timing anchor
- **Frame Offset Detection**: Automatic sync using corner timestamps
- **Smart Correlation**: Pattern matching for perfect alignment
- **Real-time Adjustment**: Dynamic frame positioning

## 📁 Project Structure

```
pickerball/
├── 📹 Video Processing
│   ├── video_360_stitcher.py          # Multi-view video stitching
│   ├── enhanced_realtime.py           # Real-time visualization
│   └── realtime_visualization.py      # Basic real-time system
│
├── 🎯 3D Tracking
│   ├── pickleball_3d_system.py        # Core 3D tracking system
│   ├── advanced_3d_tracking.py        # Advanced player movement
│   └── view_3d_results.py            # Analysis and visualization
│
├── 📊 Analysis
│   ├── pickleball_3d_analysis.ipynb   # Jupyter notebook analysis
│   └── Untitled14.ipynb              # Additional experiments
│
├── 🎥 Video Data
│   ├── e4e66c2058ff-0.0.0.0-3000-1-0-mzle9eCKS2oQvLJa7rOE000000/
│   ├── e4e66c2058ff-0.0.0.0-3000-2-0-vvkoKtKIUN7KS72O4bfR000000/
│   ├── e4e66c2058ff-0.0.0.0-3000-3-0-a4TtYafdNkjZQjVO5hll000000/
│   └── e4e66c2058ff-0.0.0.0-3000-4-0-ZhV2hb2DFg8xhbXYcpWn000000/
│
└── 🏓 YOLO Models
    └── yolov5/ (YOLOv5 framework)
```

## 🛠️ Installation

### Prerequisites
```bash
conda create -n pickleball python=3.9
conda activate pickleball
```

### Dependencies
```bash
pip install ultralytics opencv-python matplotlib numpy
pip install torch torchvision torchaudio
```

### YOLO Model
The system uses YOLO11n for object detection:
- **Classes**: Person (0), Sports Ball (37)
- **Confidence**: 0.25-0.4 threshold
- **Input Size**: 320x240 for performance

## 🎮 Usage

### 1. Synchronized Multi-View Stitching
```python
python video_360_stitcher.py
```
**Output**: `synced_multiview.mp4`
- 4-camera synchronized playback
- Reference camera as timing anchor
- Frame offset detection and correction

### 2. Advanced 3D Tracking
```python  
python advanced_3d_tracking.py
```
**Output**: `advanced_3d_tracking.mp4`
- Real-time player movement tracking
- 3D court visualization with trails
- Maximum 4 players limitation

### 3. Enhanced Real-time Visualization
```python
python enhanced_realtime.py
```
**Output**: `enhanced_realtime_pickleball.mp4`
- Split-screen: Multi-view + 3D God View
- 4.8 FPS processing with optimizations
- Ball trajectory and player mapping

### 4. Basic 3D System
```python
python pickleball_3d_system.py
```
**Output**: Multiple analysis images
- God View visualization generation
- 3D position calculation and mapping
- Statistical analysis output

## 📊 Performance Metrics

| System | Processing Speed | Output Quality | Features |
|--------|------------------|----------------|----------|
| Video Stitching | 35 FPS | 960x405 | Sync, PiP, Labels |
| 3D Tracking | 4.4 FPS | 1280x720 | Movement, Trails, Court |
| Enhanced RT | 4.8 FPS | 1120x360 | Split-screen, God View |
| Basic 3D | 3.8 FPS | Variable | Analysis, Statistics |

## 🎯 Key Features Explained

### Frame Synchronization
```python
# Automatic offset detection
frame_offsets = [0, 10, -10, 10]  # Camera 1 = reference
# Camera 2: +10 frames (0.33s delay)
# Camera 3: -10 frames (0.33s ahead)  
# Camera 4: +10 frames (0.33s delay)
```

### 3D Court Mapping
```python
# Pickleball court dimensions
court_length = 6.1  # meters
court_width = 4.27   # meters  
net_height = 0.91    # meters
player_height = 1.7  # meters (average)
```

### Player Tracking Limits
- **Maximum Players**: 4 per court
- **Court Sides**: 2 players maximum per side
- **Tracking History**: 50 frames per player
- **Position Matching**: 0.5m threshold

## 🔧 Configuration

### Camera Setup
```python
camera_positions = [
    [0, 0, 2.5],    # Camera 1: Front-left corner
    [6, 0, 2.5],    # Camera 2: Front-right corner
    [6, 4, 2.5],    # Camera 3: Back-right corner  
    [0, 4, 2.5]     # Camera 4: Back-left corner
]
```

### Video Processing Settings
```python
detection_size = (320, 240)  # Optimized for speed
display_size = (480, 270)    # 16:9 aspect ratio
confidence_threshold = 0.3    # YOLO detection confidence
max_players = 4              # Court player limit
```

## 📈 Output Files

### Generated Videos
- `synced_multiview.mp4` - Synchronized 4-camera view
- `advanced_3d_tracking.mp4` - 3D player tracking
- `enhanced_realtime_pickleball.mp4` - Split-screen visualization
- `realtime_pickleball.mp4` - Basic real-time output

### Analysis Images
- `god_view_frame_*.png` - Top-down court visualizations
- `side_view_frame_*.png` - Profile perspective views
- `tracking_analysis_*.png` - Movement analysis charts

## 🚨 System Requirements

### Hardware
- **GPU**: CUDA-compatible (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB+ for video processing
- **CPU**: Multi-core processor for parallel processing

### Software
- **Python**: 3.8+
- **OpenCV**: 4.7.0+
- **PyTorch**: Latest stable
- **CUDA**: 11.0+ (for GPU acceleration)

## 🎯 Use Cases

### Sports Analysis
- **Player Movement**: Track player positioning and court coverage
- **Game Statistics**: Analyze rally lengths and ball trajectories  
- **Performance Metrics**: Movement patterns and reaction times
- **Training Feedback**: Visual analysis for improvement

### Broadcasting
- **Multi-Angle Coverage**: Synchronized camera switching
- **Highlight Generation**: Automatic event detection
- **Replay Systems**: Frame-perfect synchronization
- **Live Analysis**: Real-time statistics overlay

### Research Applications
- **Computer Vision**: Multi-view geometry and tracking
- **Sports Science**: Biomechanics and movement analysis
- **Machine Learning**: Training data for sports AI
- **3D Reconstruction**: Court and player modeling

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **YOLO**: Ultralytics for object detection framework
- **OpenCV**: Computer vision and video processing
- **PyTorch**: Deep learning and GPU acceleration
- **Matplotlib**: Visualization and plotting

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out through GitHub.

---

**Built with ❤️ for pickleball analysis and computer vision research**

🏓 *"Advanced multi-view tracking for the modern game"*