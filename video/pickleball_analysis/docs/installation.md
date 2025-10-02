# Installation Guide - Pickleball Video Analysis Pipeline

## 📋 Yêu cầu hệ thống

### Hệ điều hành được hỗ trợ
- Windows 10/11 (64-bit)
- macOS 10.15+ (Intel hoặc Apple Silicon)
- Ubuntu 18.04+ / Debian 10+
- Other Linux distributions (với Python 3.8+)

### Phần cứng tối thiểu
- **CPU**: Intel i5-8400 hoặc AMD Ryzen 5 2600
- **RAM**: 8GB (16GB khuyến nghị)
- **Storage**: 10GB dung lượng trống (20GB khuyến nghị)
- **GPU**: Optional - NVIDIA GTX 1060+ với CUDA support

### Phần mềm yêu cầu
- Python 3.8 - 3.11
- pip package manager
- Git (để clone repository)

## 🚀 Cài đặt nhanh (Quick Start)

### Bước 1: Clone repository
```bash
# Clone project (nếu có repository)
git clone <repository-url>
cd pickleball_analysis

# Hoặc tạo project từ files có sẵn
mkdir pickleball_analysis
cd pickleball_analysis
# Copy tất cả files vào thư mục này
```

### Bước 2: Tạo Python virtual environment
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Cài đặt requirements
pip install -r requirements.txt
```

### Bước 4: Verify installation
```bash
# Test import các modules chính
python -c "import cv2; import torch; import ultralytics; print('Installation successful!')"
```

## 🔧 Cài đặt chi tiết

### Option 1: CPU-only Installation

Nếu không có GPU hoặc muốn sử dụng CPU only:

```bash
# Cài đặt PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các dependencies khác
pip install opencv-python numpy pandas matplotlib seaborn plotly
pip install ultralytics supervision loguru tqdm moviepy
pip install scikit-image scipy filterpy lap PyYAML
```

### Option 2: GPU Installation (NVIDIA CUDA)

Nếu có GPU NVIDIA với CUDA support:

#### Kiểm tra CUDA version:
```bash
# Kiểm tra CUDA version
nvidia-smi

# Hoặc
nvcc --version
```

#### Cài đặt PyTorch với CUDA:
```bash
# CUDA 11.8 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các dependencies khác
pip install opencv-python numpy pandas matplotlib seaborn plotly
pip install ultralytics supervision loguru tqdm moviepy
pip install scikit-image scipy filterpy lap PyYAML
```

#### Verify GPU installation:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Option 3: Conda Installation

Nếu sử dụng Anaconda/Miniconda:

```bash
# Tạo conda environment
conda create -n pickleball python=3.10
conda activate pickleball

# Cài đặt các packages cơ bản
conda install opencv numpy pandas matplotlib seaborn plotly scipy

# Cài đặt PyTorch (CPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Hoặc PyTorch (GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Cài đặt các packages còn lại qua pip
pip install ultralytics supervision loguru tqdm moviepy
pip install scikit-image filterpy lap PyYAML
```

## 🎯 Cấu hình sau cài đặt

### Kiểm tra cài đặt
```bash
# Chạy script kiểm tra
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import cv2
    print(f'OpenCV version: {cv2.__version__}')
except:
    print('OpenCV not found')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except:
    print('PyTorch not found')

try:
    from ultralytics import YOLO
    print('YOLOv8 available')
except:
    print('Ultralytics not found')
"
```

### Download YOLOv8 weights
```bash
# YOLOv8 weights sẽ được tự động download khi chạy lần đầu
# Hoặc download thủ công:
python -c "from ultralytics import YOLO; model = YOLO('yolov8x.pt')"
```

### Cấu hình video input paths
```bash
# Tạo symbolic link hoặc copy video files
# Windows
mklink /D data_video "C:\path\to\your\videos"

# macOS/Linux  
ln -s /path/to/your/videos data_video
```

## 🛠️ Troubleshooting

### Lỗi thường gặp và cách khắc phục

#### 1. Import Error: "cv2" could not be resolved
```bash
# Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

#### 2. CUDA out of memory
```bash
# Giảm batch size trong config.yaml
# Hoặc chuyển sang CPU mode
```

#### 3. YOLOv8 download fails
```bash
# Download manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
# Đặt file vào project directory
```

#### 4. Permission denied (Linux/macOS)
```bash
# Fix permissions
chmod +x main.py
sudo chown -R $USER:$USER pickleball_analysis/
```

#### 5. FFmpeg not found
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows
# Download từ https://ffmpeg.org/download.html
# Thêm vào PATH environment variable
```

#### 6. Virtual environment issues
```bash
# Recreate virtual environment
deactivate
rm -rf venv  # hoặc rmdir /s venv trên Windows
python -m venv venv
source venv/bin/activate  # hoặc venv\Scripts\activate
pip install -r requirements.txt
```

## 🧪 Test installation

### Basic functionality test
```bash
# Test detection module
python src/detection.py

# Test court detection
python src/court_detection.py --validate

# Test full pipeline với sample video
python main.py --input sample_video.mp4 --max-frames 100
```

### Performance benchmark
```bash
# GPU performance test
python -c "
import torch
import time

if torch.cuda.is_available():
    # Warm up
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    end = time.time()
    
    print(f'GPU performance: {(end-start)*1000:.2f}ms for 100 matrix multiplications')
else:
    print('CUDA not available')
"
```

## 🐳 Docker Installation (Optional)

### Build Docker image
```dockerfile
# Dockerfile example
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

```bash
# Build and run
docker build -t pickleball-analysis .
docker run -v /path/to/videos:/app/data_video pickleball-analysis
```

## 🌐 Cloud Deployment

### Google Colab
```python
# Install in Colab
!pip install ultralytics supervision loguru
!pip install opencv-python matplotlib seaborn plotly

# Upload files và chạy
from google.colab import files
files.upload()  # Upload project files
!python main.py --input sample_video.mp4
```

### AWS EC2 / GCP Compute Engine
```bash
# Instance requirements
# - GPU instance (g4dn.xlarge hoặc n1-standard-4-gpu)
# - Ubuntu 20.04 LTS
# - 50GB storage

# Setup script
sudo apt update
sudo apt install -y python3-pip git ffmpeg
git clone <repository>
cd pickleball_analysis
pip3 install -r requirements.txt
```

## 📊 Resource Usage

### Disk Space
- Base installation: ~2GB
- YOLOv8 models: ~150MB
- Sample outputs: ~500MB
- Total recommended: 10GB

### Memory Usage
- CPU mode: 2-4GB RAM
- GPU mode: 4-6GB RAM + 2-4GB VRAM

### Processing Time (1080p video, 1000 frames)
- CPU (i7-10700K): ~10-15 minutes
- GPU (RTX 3070): ~3-5 minutes

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] GPU setup completed (if applicable)
- [ ] YOLOv8 weights downloaded
- [ ] Video input paths configured
- [ ] Basic functionality test passed
- [ ] Court calibration completed

## 🆘 Getting Help

Nếu gặp vấn đề:
1. Kiểm tra logs trong thư mục `logs/`
2. Chạy test commands ở trên
3. Kiểm tra requirements versions
4. Tham khảo [technical_specs.md](technical_specs.md)
5. Tạo issue với đầy đủ thông tin về environment

---

**Phiên bản**: 1.0.0  
**Cập nhật**: October 2, 2025  
**Hỗ trợ**: Python 3.8-3.11, CUDA 11.8+