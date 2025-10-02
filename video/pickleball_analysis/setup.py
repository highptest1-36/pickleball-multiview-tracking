"""
Setup script for Pickleball Video Analysis Pipeline

Cài đặt và setup tự động cho pipeline.
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path

def check_python_version():
    """Kiểm tra Python version."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_git():
    """Kiểm tra Git installation."""
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
        print("✅ Git is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Git not found (optional)")
        return False

def install_requirements():
    """Cài đặt Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found")
        return False
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True)
        
        # Install requirements
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], 
                      check=True)
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Tạo các thư mục cần thiết."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "output",
        "output/tracking_data",
        "output/charts", 
        "output/reports",
        "output/processed_videos",
        "logs",
        "data_video"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   📂 {directory}")
    
    print("✅ Directories created")
    return True

def download_yolo_weights():
    """Download YOLOv8 weights."""
    print("\n🤖 Downloading YOLO weights...")
    
    try:
        # Import here to avoid issues if ultralytics not installed yet
        from ultralytics import YOLO
        
        # This will auto-download the weights
        model = YOLO('yolov8x.pt')
        print("✅ YOLOv8 weights downloaded")
        return True
    except Exception as e:
        print(f"⚠️ Could not download YOLO weights: {e}")
        print("   Weights will be downloaded automatically on first run")
        return True  # Not critical

def test_imports():
    """Test critical imports."""
    print("\n🧪 Testing imports...")
    
    critical_imports = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML'),
        ('loguru', 'Loguru')
    ]
    
    optional_imports = [
        ('torch', 'PyTorch'),
        ('ultralytics', 'Ultralytics'),
        ('plotly', 'Plotly'),
        ('seaborn', 'Seaborn')
    ]
    
    success = True
    
    # Test critical imports
    for module_name, display_name in critical_imports:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ❌ {display_name} - REQUIRED")
            success = False
    
    # Test optional imports
    for module_name, display_name in optional_imports:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            print(f"   ⚠️ {display_name} - OPTIONAL")
    
    return success

def test_gpu():
    """Test GPU availability."""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   🎯 Primary GPU: {gpu_name}")
            return True
        else:
            print("   ⚠️ CUDA not available - will use CPU")
            return False
    except ImportError:
        print("   ⚠️ PyTorch not installed - cannot check GPU")
        return False

def create_sample_config():
    """Tạo sample configuration files."""
    print("\n⚙️ Creating sample configuration...")
    
    # Check if config already exists
    config_file = "config/config.yaml"
    if os.path.exists(config_file):
        print(f"   ℹ️ {config_file} already exists")
        return True
    
    # Create basic config
    sample_config = """# Pickleball Analysis Configuration (Auto-generated)

# Court specifications
court:
  width_meters: 13.41
  height_meters: 6.1
  net_height_meters: 0.914

# Video settings
video:
  fps: 30
  input_videos:
    - "data_video/san1.mp4"
    - "data_video/san2.mp4" 
    - "data_video/san3.mp4"
    - "data_video/san4.mp4"

# Detection settings
detection:
  model: "yolov8x.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: "auto"  # Will auto-detect GPU/CPU

# Tracking settings
tracking:
  max_disappeared: 30
  max_distance: 100
  min_hits: 3

# Analysis settings
analysis:
  smoothing:
    enabled: true
    window_size: 5
  velocity:
    min_distance_threshold: 0.1

# Visualization settings
visualization:
  colors:
    player_1: [255, 0, 0]
    player_2: [0, 255, 0]
    player_3: [0, 0, 255]
    player_4: [255, 255, 0]
    ball: [255, 0, 255]
  heatmap:
    bins: 50
    alpha: 0.7
    colormap: "hot"
  charts:
    dpi: 300
    figsize: [12, 8]

# Output settings
output:
  base_dir: "output"
  subdirs:
    videos: "processed_videos"
    data: "tracking_data"
    heatmaps: "heatmaps" 
    charts: "charts"
    reports: "reports"

# Logging settings
logging:
  level: "INFO"
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
  file_rotation: "1 week"
  file_retention: "1 month"
"""
    
    os.makedirs("config", exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(sample_config)
    
    print(f"   ✅ Created {config_file}")
    return True

def create_run_scripts():
    """Tạo convenience scripts."""
    print("\n📜 Creating run scripts...")
    
    # Windows batch script
    windows_script = """@echo off
echo Starting Pickleball Analysis Pipeline...
python main.py %*
pause
"""
    
    with open("run_pipeline.bat", 'w') as f:
        f.write(windows_script)
    
    # Linux/Mac shell script  
    unix_script = """#!/bin/bash
echo "Starting Pickleball Analysis Pipeline..."
python main.py "$@"
"""
    
    with open("run_pipeline.sh", 'w') as f:
        f.write(unix_script)
    
    # Make shell script executable
    try:
        os.chmod("run_pipeline.sh", 0o755)
    except:
        pass  # Windows doesn't support chmod
    
    print("   ✅ Created run_pipeline.bat (Windows)")
    print("   ✅ Created run_pipeline.sh (Linux/Mac)")
    return True

def run_demo():
    """Chạy demo để test setup."""
    print("\n🎬 Running demo test...")
    
    try:
        subprocess.run([sys.executable, 'demo.py', '--quick'], 
                      check=True, timeout=60)
        print("✅ Demo test completed successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Demo test failed - but setup may still be OK")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️ Demo test timed out")
        return False
    except FileNotFoundError:
        print("⚠️ demo.py not found - skipping test")
        return True

def print_next_steps():
    """In hướng dẫn next steps."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. 🎥 Prepare your videos:")
    print("   - Copy video files to data_video/ directory")
    print("   - Supported formats: .mp4, .avi, .mov, .mkv")
    
    print("\n2. 🎯 Calibrate courts:")
    print("   python src/court_detection.py --calibrate")
    
    print("\n3. 🚀 Run analysis:")
    print("   python main.py --input data_video/")
    print("   # Or use convenience scripts:")
    print("   ./run_pipeline.sh --input data_video/  (Linux/Mac)")
    print("   run_pipeline.bat --input data_video\\   (Windows)")
    
    print("\n4. 📊 View results:")
    print("   - Open output/charts/interactive_dashboard.html")
    print("   - Check output/tracking_data/tracking_results.csv")
    print("   - Review output/reports/analysis_results.json")
    
    print("\n🔧 Configuration:")
    print("   - Edit config/config.yaml to customize settings")
    print("   - See docs/usage_guide.md for detailed instructions")
    
    print("\n🆘 If you need help:")
    print("   - Check logs/ directory for error messages")
    print("   - Run: python demo.py --test-only config")
    print("   - Review docs/installation.md and docs/usage_guide.md")
    
    print("\n" + "="*60)

def main():
    """Main setup function."""
    print("🏓 Pickleball Video Analysis Pipeline Setup")
    print("="*50)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    check_git()
    
    # Core setup steps
    steps = [
        ("Installing dependencies", install_requirements),
        ("Setting up directories", setup_directories),
        ("Creating configuration", create_sample_config),
        ("Testing imports", test_imports),
        ("Creating run scripts", create_run_scripts),
        ("Downloading YOLO weights", download_yolo_weights),
    ]
    
    success = True
    for step_name, step_func in steps:
        print(f"\n⏳ {step_name}...")
        if not step_func():
            print(f"❌ {step_name} failed")
            success = False
            break
    
    if success:
        # Optional steps
        test_gpu()
        run_demo()
        print_next_steps()
    else:
        print("\n❌ Setup failed. Please check error messages above.")
        print("💡 Try running individual steps manually or check docs/installation.md")
    
    return success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Pickleball Analysis Pipeline")
    parser.add_argument('--skip-demo', action='store_true', 
                       help='Skip demo test')
    parser.add_argument('--force', action='store_true',
                       help='Force reinstall dependencies')
    
    args = parser.parse_args()
    
    if args.force:
        print("🔄 Force mode: Will reinstall all dependencies")
    
    success = main()
    sys.exit(0 if success else 1)