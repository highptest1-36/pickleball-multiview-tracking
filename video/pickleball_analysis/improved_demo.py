"""
Demo Script - IMPROVED 2D Court Viewer

Features:
- Simple rectangular court design (chỉ hình chữ nhật + net)
- Accurate coordinate transformation
- Better player/ball visualization
- Side A/B management (max 2 players each)
"""

import sys
import os
from simple_2d_viewer import run_simple_demo

def main():
    """Demo với improved features."""
    print("🏓 IMPROVED 2D COURT VIEWER DEMO")
    print("=" * 60)
    print("🔥 NEW FEATURES:")
    print("✅ Simplified Court Design - Only rectangle + center net")
    print("✅ Accurate Coordinate Transformation")  
    print("✅ Better Player/Ball Visualization")
    print("✅ Side A/B Management (max 2 players each)")
    print("✅ Real-time Movement Tracking")
    print("✅ Ball Speed Indicators")
    print("=" * 60)
    
    # Paths
    video_path = r"C:\Users\highp\pickerball\video\data_video\san1.mp4"
    tracking_path = r"C:\Users\highp\pickerball\video\pickleball_analysis\real_demo_output\tracking_data\real_tracking.csv"
    
    # Check files
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
        
    if not os.path.exists(tracking_path):
        print(f"❌ Tracking data not found: {tracking_path}")
        return
    
    print("🎮 CONTROLS:")
    print("  SPACE: Pause/Resume")
    print("  'h': Toggle heatmap")
    print("  '1-6': Show player-specific heatmap")
    print("  '0': Show all players heatmap")
    print("  'q': Quit")
    print("=" * 60)
    print("🚀 Starting demo... Press any key in video window to start!")
    
    # Run improved demo
    run_simple_demo(video_path, tracking_path, max_frames=300)
    
    print("=" * 60)
    print("✅ Demo completed!")
    print("📊 Features demonstrated:")
    print("  - Simple court design (rectangle + net)")
    print("  - Accurate player position mapping")
    print("  - Side A/B player management")
    print("  - Ball tracking with speed")
    print("  - Real-time heatmaps")
    print("=" * 60)

if __name__ == "__main__":
    main()