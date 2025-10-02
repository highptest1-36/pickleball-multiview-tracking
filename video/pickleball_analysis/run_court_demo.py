"""
Demo script để chạy 2D Court Viewer với controls đầy đủ

Chạy: python run_court_demo.py
"""

import os
import sys
from simple_2d_viewer import run_simple_demo

def main():
    """Main demo function."""
    # Paths (absolute)
    video_path = r"C:\Users\highp\pickerball\video\data_video\san1.mp4"
    tracking_path = r"C:\Users\highp\pickerball\video\pickleball_analysis\real_demo_output\tracking_data\real_tracking.csv"
    
    # Check files exist
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
        
    if not os.path.exists(tracking_path):
        print(f"❌ Tracking data not found: {tracking_path}")
        return
    
    print("🏓 Starting Pickleball 2D Court Demo")
    print("=" * 50)
    print("Features:")
    print("✅ Court calibration với homography matrix")
    print("✅ Only tracking players trong khu vực sân")
    print("✅ Chia sân thành 2 zones (left/right)")
    print("✅ Tối đa 2 players mỗi zone")
    print("✅ Real-time heatmaps")
    print("✅ Player trails và ball tracking")
    print()
    print("Controls trong demo:")
    print("  SPACE: Pause/Resume")
    print("  'h': Toggle heatmap")
    print("  '1-6': Show heatmap for specific player")
    print("  '0': Show heatmap for all players")
    print("  'q': Quit")
    print("=" * 50)
    
    # Run demo với longer duration
    run_simple_demo(video_path, tracking_path, max_frames=500)

if __name__ == "__main__":
    main()