"""
Demo Script cho Pickleball Analysis Pipeline

Script này tạo data demo và test các chức năng chính của pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Any
import json
import time

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.utils import load_config, setup_logging, create_output_dirs
    from src.visualization import PickleballVisualizer
    from src.analysis import MovementAnalyzer
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

def create_demo_video(output_path: str = "demo_video.mp4", 
                     duration_seconds: int = 10,
                     fps: int = 30) -> str:
    """
    Tạo demo video với objects di chuyển.
    
    Args:
        output_path: Đường dẫn video output
        duration_seconds: Độ dài video (giây)
        fps: Frames per second
        
    Returns:
        Đường dẫn video đã tạo
    """
    print(f"🎬 Tạo demo video: {output_path}")
    
    width, height = 1280, 720
    total_frames = duration_seconds * fps
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create background (green court-like)
    court_color = (34, 139, 34)  # Forest green
    
    # Player positions (simulate 2 players moving)
    player1_start = (200, 300)
    player2_start = (800, 400)
    
    for frame_idx in range(total_frames):
        # Create frame
        frame = np.full((height, width, 3), court_color, dtype=np.uint8)
        
        # Draw court lines (simplified)
        cv2.line(frame, (0, height//2), (width, height//2), (255, 255, 255), 3)  # Net
        cv2.rectangle(frame, (50, 50), (width-50, height-50), (255, 255, 255), 2)  # Boundary
        
        # Animate players
        t = frame_idx / total_frames
        
        # Player 1 moves in sinusoidal pattern
        p1_x = int(player1_start[0] + 300 * t)
        p1_y = int(player1_start[1] + 100 * np.sin(t * 4 * np.pi))
        
        # Player 2 moves in different pattern
        p2_x = int(player2_start[0] - 200 * t)
        p2_y = int(player2_start[1] + 50 * np.cos(t * 6 * np.pi))
        
        # Draw players (circles)
        cv2.circle(frame, (p1_x, p1_y), 20, (0, 0, 255), -1)  # Red player
        cv2.circle(frame, (p2_x, p2_y), 20, (0, 255, 255), -1)  # Yellow player
        
        # Draw ball (small circle moving fast)
        ball_x = int(400 + 300 * np.sin(t * 8 * np.pi))
        ball_y = int(360 + 150 * np.cos(t * 12 * np.pi))
        cv2.circle(frame, (ball_x, ball_y), 5, (255, 255, 255), -1)  # White ball
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Demo video created: {output_path}")
    return output_path

def create_demo_tracking_data(output_path: str = "demo_tracking.csv") -> pd.DataFrame:
    """
    Tạo demo tracking data.
    
    Args:
        output_path: Đường dẫn CSV output
        
    Returns:
        DataFrame với tracking data
    """
    print(f"📊 Tạo demo tracking data: {output_path}")
    
    # Simulate 10 seconds at 30 FPS = 300 frames
    frames = 300
    fps = 30
    
    data = []
    
    # Player 1 data
    for frame_id in range(frames):
        timestamp = frame_id / fps
        
        # Simulate movement
        t = timestamp / 10.0  # Normalize to 0-1
        
        # Player 1 - moves across court
        p1_x = 50 + 1100 * t + 100 * np.sin(t * 4 * np.pi)
        p1_y = 300 + 100 * np.sin(t * 2 * np.pi)
        
        data.append({
            'frame_id': frame_id,
            'timestamp': timestamp,
            'object_id': 1,
            'class': 'player',
            'center_x': p1_x,
            'center_y': p1_y,
            'confidence': 0.95,
            'bbox_x1': p1_x - 20,
            'bbox_y1': p1_y - 30,
            'bbox_x2': p1_x + 20,
            'bbox_y2': p1_y + 30
        })
        
        # Player 2 - different movement pattern
        p2_x = 800 - 300 * t + 50 * np.cos(t * 6 * np.pi)
        p2_y = 400 + 80 * np.cos(t * 3 * np.pi)
        
        data.append({
            'frame_id': frame_id,
            'timestamp': timestamp,
            'object_id': 2,
            'class': 'player',
            'center_x': p2_x,
            'center_y': p2_y,
            'confidence': 0.88,
            'bbox_x1': p2_x - 20,
            'bbox_y1': p2_y - 30,
            'bbox_x2': p2_x + 20,
            'bbox_y2': p2_y + 30
        })
        
        # Ball data (not in every frame)
        if frame_id % 3 == 0:  # Ball visible every 3rd frame
            ball_x = 400 + 300 * np.sin(t * 8 * np.pi)
            ball_y = 360 + 150 * np.cos(t * 12 * np.pi)
            
            data.append({
                'frame_id': frame_id,
                'timestamp': timestamp,
                'object_id': 3,
                'class': 'ball',
                'center_x': ball_x,
                'center_y': ball_y,
                'confidence': 0.75,
                'bbox_x1': ball_x - 5,
                'bbox_y1': ball_y - 5,
                'bbox_x2': ball_x + 5,
                'bbox_y2': ball_y + 5
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Demo tracking data created: {len(df)} records")
    return df

def test_analysis_module(tracking_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test analysis module với demo data.
    
    Args:
        tracking_df: Demo tracking DataFrame
        
    Returns:
        Analysis results
    """
    print("🔬 Testing analysis module...")
    
    try:
        config = load_config()
        analyzer = MovementAnalyzer(config)
        
        # Run analysis
        start_time = time.time()
        results = analyzer.analyze_tracking_data(tracking_df)
        analysis_time = time.time() - start_time
        
        print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
        
        # Print summary
        player_analysis = results.get('player_analysis', {})
        for player_id, data in player_analysis.items():
            print(f"📈 {player_id}:")
            print(f"   Distance: {data.get('total_distance_meters', 0):.1f}m")
            print(f"   Avg Speed: {data.get('avg_speed_kmh', 0):.1f} km/h")
            print(f"   Max Speed: {data.get('max_speed_kmh', 0):.1f} km/h")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return {}

def test_visualization_module(analysis_results: Dict[str, Any]) -> List[str]:
    """
    Test visualization module.
    
    Args:
        analysis_results: Results từ analysis
        
    Returns:
        List đường dẫn files đã tạo
    """
    print("🎨 Testing visualization module...")
    
    try:
        config = load_config()
        visualizer = PickleballVisualizer(config)
        
        # Create demo output directory
        demo_output_dir = "demo_output"
        os.makedirs(demo_output_dir, exist_ok=True)
        
        # Generate visualizations
        start_time = time.time()
        output_files = visualizer.save_all_visualizations(analysis_results, demo_output_dir)
        viz_time = time.time() - start_time
        
        print(f"✅ Visualizations created in {viz_time:.2f} seconds")
        
        # List created files
        for file_type, file_path in output_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"   📁 {file_type}: {file_path} ({file_size:.1f} KB)")
            else:
                print(f"   ❌ {file_type}: {file_path} (not found)")
        
        return list(output_files.values())
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return []

def test_configuration() -> bool:
    """
    Test configuration loading.
    
    Returns:
        True nếu thành công
    """
    print("⚙️ Testing configuration...")
    
    try:
        config = load_config()
        
        # Check required sections
        required_sections = ['court', 'detection', 'tracking', 'analysis', 'visualization']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
            print(f"   ✅ {section}: OK")
        
        # Test output directory creation
        output_dirs = create_output_dirs("demo_output", config)
        print(f"   ✅ Output directories: {len(output_dirs)} created")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_full_demo():
    """Chạy full demo pipeline."""
    print("🚀 Starting Pickleball Analysis Demo")
    print("=" * 50)
    
    # Test 1: Configuration
    if not test_configuration():
        print("❌ Demo failed at configuration stage")
        return
    
    # Test 2: Create demo data
    demo_video = create_demo_video("demo_output/demo_video.mp4")
    demo_tracking_df = create_demo_tracking_data("demo_output/demo_tracking.csv")
    
    # Test 3: Analysis
    analysis_results = test_analysis_module(demo_tracking_df)
    if not analysis_results:
        print("❌ Demo failed at analysis stage")
        return
    
    # Test 4: Visualization
    viz_files = test_visualization_module(analysis_results)
    if not viz_files:
        print("❌ Demo failed at visualization stage")
        return
    
    # Test 5: Export results
    print("💾 Exporting demo results...")
    
    # Save analysis results
    analysis_output = "demo_output/demo_analysis_results.json"
    with open(analysis_output, 'w', encoding='utf-8') as f:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        json.dump(convert_types(analysis_results), f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis results saved: {analysis_output}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("📁 Demo outputs created in 'demo_output/' directory:")
    print("   🎬 demo_video.mp4 - Sample video")
    print("   📊 demo_tracking.csv - Tracking data")
    print("   📈 demo_analysis_results.json - Analysis results")
    print("   🎨 Visualization files (PNG, HTML)")
    print("\n💡 Next steps:")
    print("   1. Review demo outputs")
    print("   2. Try with real video: python main.py --input your_video.mp4")
    print("   3. Calibrate court: python src/court_detection.py --calibrate")

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pickleball Analysis Demo")
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick demo (skip video creation)')
    parser.add_argument('--test-only', choices=['config', 'analysis', 'viz'],
                       help='Run specific test only')
    
    args = parser.parse_args()
    
    if args.test_only:
        if args.test_only == 'config':
            test_configuration()
        elif args.test_only == 'analysis':
            df = create_demo_tracking_data()
            test_analysis_module(df)
        elif args.test_only == 'viz':
            df = create_demo_tracking_data()
            results = test_analysis_module(df)
            test_visualization_module(results)
    else:
        run_full_demo()

if __name__ == "__main__":
    main()