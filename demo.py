#!/usr/bin/env python3
"""
Quick demo script for Pickleball Multi-View Tracking System
Run this to test the basic functionality
"""

import os
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'cv2', 'numpy', 'matplotlib', 'ultralytics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'ultralytics':
                from ultralytics import YOLO
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def check_video_files():
    """Check if video files exist"""
    print("\n📹 Checking video files...")
    
    video_dirs = [
        "e4e66c2058ff-0.0.0.0-3000-1-0-mzle9eCKS2oQvLJa7rOE000000",
        "e4e66c2058ff-0.0.0.0-3000-2-0-vvkoKtKIUN7KS72O4bfR000000",
        "e4e66c2058ff-0.0.0.0-3000-3-0-a4TtYafdNkjZQjVO5hll000000",
        "e4e66c2058ff-0.0.0.0-3000-4-0-ZhV2hb2DFg8xhbXYcpWn000000"
    ]
    
    found_videos = 0
    for i, video_dir in enumerate(video_dirs):
        if os.path.exists(video_dir):
            # Look for mp4 files in directory
            mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            if mp4_files:
                print(f"✅ Camera {i+1}: {mp4_files[0]}")
                found_videos += 1
            else:
                print(f"❌ Camera {i+1}: No MP4 files found")
        else:
            print(f"❌ Camera {i+1}: Directory not found")
    
    if found_videos == 4:
        print("✅ All 4 camera videos found!")
        return True
    else:
        print(f"⚠️ Only {found_videos}/4 videos found")
        return False

def run_demo():
    """Run a quick demo"""
    print("\n🚀 Running Quick Demo...")
    
    # Try to import and run basic functionality
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        
        print("✅ Loading YOLO model...")
        model = YOLO('yolo11n.pt')  # This will download if not exists
        
        print("✅ Testing video capture...")
        # Test if we can create video capture objects
        test_video = "e4e66c2058ff-0.0.0.0-3000-1-0-mzle9eCKS2oQvLJa7rOE000000/h20250926093017-20250926093526m.mp4"
        if os.path.exists(test_video):
            cap = cv2.VideoCapture(test_video)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Successfully read frame: {frame.shape}")
                    cap.release()
                else:
                    print("❌ Could not read frame")
            else:
                print("❌ Could not open video")
        
        print("\n🎉 Demo completed successfully!")
        print("\nAvailable scripts:")
        print("  • python video_360_stitcher.py - Multi-view video stitching")
        print("  • python advanced_3d_tracking.py - 3D player tracking") 
        print("  • python enhanced_realtime.py - Real-time visualization")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    print("🏓 PICKLEBALL MULTI-VIEW TRACKING SYSTEM")
    print("=" * 50)
    
    # Check system
    deps_ok = check_dependencies()
    videos_ok = check_video_files()
    
    if deps_ok and videos_ok:
        run_demo()
    else:
        print("\n⚠️ System not ready. Please fix the issues above.")
        
    print("\n📚 For detailed usage, see README.md")
    print("🔗 GitHub: https://github.com/your-username/pickleball-tracking")

if __name__ == "__main__":
    main()