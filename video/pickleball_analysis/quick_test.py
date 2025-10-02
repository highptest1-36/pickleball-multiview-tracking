"""
Quick Test Script để demo 2D Court Viewer

Tạo một test nhanh với info chi tiết về kết quả.
"""

import cv2
import pandas as pd
import json
import os
from simple_2d_viewer import Simple2DCourtViewer

def quick_test():
    """Quick test của 2D Court Viewer."""
    print("🧪 QUICK TEST - 2D Court Viewer")
    print("=" * 50)
    
    # Check files
    video_path = r"C:\Users\highp\pickerball\video\data_video\san1.mp4"
    tracking_path = r"real_demo_output\tracking_data\real_tracking.csv"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
        
    if not os.path.exists(tracking_path):
        print(f"❌ Tracking data not found: {tracking_path}")
        return
    
    # Load data
    tracking_df = pd.read_csv(tracking_path)
    print(f"📊 Tracking Data: {len(tracking_df)} records")
    
    # Check unique players
    unique_players = tracking_df[tracking_df['class'].str.contains('player|person', na=False)]['object_id'].unique()
    unique_balls = tracking_df[tracking_df['class'] == 'ball']['object_id'].unique()
    
    print(f"👥 Unique Players: {list(unique_players)}")
    print(f"⚽ Unique Balls: {list(unique_balls)}")
    
    # Frame range
    frame_range = f"{tracking_df['frame_id'].min()} - {tracking_df['frame_id'].max()}"
    print(f"🎬 Frame Range: {frame_range}")
    
    # Initialize viewer
    viewer = Simple2DCourtViewer("config/court_points.json")
    
    # Test with first 10 frames
    test_frames = 10
    player_counts = {"left": 0, "right": 0}
    
    print(f"\n🔍 Testing first {test_frames} frames...")
    
    for frame_id in range(test_frames):
        frame_detections = tracking_df[tracking_df['frame_id'] == frame_id]
        
        if len(frame_detections) > 0:
            print(f"\nFrame {frame_id}: {len(frame_detections)} detections")
            
            for _, detection in frame_detections.iterrows():
                obj_id = int(detection['object_id'])
                center_x = detection['center_x']
                center_y = detection['center_y']
                timestamp = detection['timestamp']
                class_name = detection['class']
                
                if 'player' in class_name or class_name == 'person':
                    # Test point in court
                    in_court = viewer.is_point_in_court(center_x, center_y)
                    
                    if in_court:
                        # Transform point
                        if viewer.homography_matrix is not None:
                            court_x, court_y = viewer.transform_point_calibrated(center_x, center_y)
                        else:
                            court_x, court_y = viewer.transform_point_simple(center_x, center_y)
                        
                        # Get side
                        side = viewer.get_court_side(court_x, court_y)
                        
                        print(f"  Player {obj_id}: ({center_x:.0f}, {center_y:.0f}) -> Court ({court_x}, {court_y}) [{side}]")
                        
                        # Add to viewer
                        viewer.add_player_position(obj_id, center_x, center_y, timestamp)
                    else:
                        print(f"  Player {obj_id}: OUTSIDE court bounds")
                        
                elif class_name == 'ball':
                    in_court = viewer.is_point_in_court(center_x, center_y)
                    if in_court:
                        viewer.add_ball_position(center_x, center_y, timestamp)
                        print(f"  Ball {obj_id}: ({center_x:.0f}, {center_y:.0f}) -> Added to court")
    
    # Final stats
    print(f"\n📈 FINAL STATS:")
    print(f"Left Side Players: {len(viewer.left_side_players)} - {list(viewer.left_side_players)}")
    print(f"Right Side Players: {len(viewer.right_side_players)} - {list(viewer.right_side_players)}")
    print(f"Total Tracked Players: {len(viewer.player_positions)}")
    print(f"Ball Positions: {len(viewer.ball_positions)}")
    
    # Test render a frame
    print(f"\n🖼️ Testing frame render...")
    frame_info = {'frame': 10, 'time': 0.33, 'detections': 5}
    
    try:
        court_image = viewer.render_frame(frame_info, show_heatmap=False)
        print(f"✅ Successfully rendered court view: {court_image.shape}")
        
        # Save test image
        cv2.imwrite("test_court_output.png", court_image)
        print(f"💾 Saved test image: test_court_output.png")
        
    except Exception as e:
        print(f"❌ Render error: {e}")
    
    print(f"\n✅ Quick test completed!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    quick_test()