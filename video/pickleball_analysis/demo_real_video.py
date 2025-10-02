"""
Demo Real Video Analysis - Phân tích video thật với hiển thị song song

Script này sẽ:
1. Phân tích video san1.mp4 
2. Hiển thị video gốc và kết quả phân tích side-by-side
3. Tạo output video với overlay tracking/analysis
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.utils import load_config, setup_logging, create_output_dirs
    from src.court_detection import CourtDetector
    from src.detection import PickleballDetector
    from src.tracking import SimpleTracker
    from src.analysis import MovementAnalyzer
    from src.visualization import PickleballVisualizer
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class RealVideoDemo:
    def __init__(self, video_path: str, output_dir: str = "real_demo_output"):
        """
        Initialize demo với video thật.
        
        Args:
            video_path: Đường dẫn video input
            output_dir: Thư mục output
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = load_config()
        
        # Setup logging
        setup_logging(self.config)
        
        # Create output directories
        create_output_dirs(output_dir, self.config)
        
        # Initialize components
        self.court_detector = CourtDetector()
        self.detector = PickleballDetector(self.config)
        self.tracker = SimpleTracker(self.config)
        self.analyzer = MovementAnalyzer(self.config)
        self.visualizer = PickleballVisualizer(self.config)
        
        print(f"🎬 Initialized demo for video: {video_path}")

    def analyze_video(self, max_frames: int = 300) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Phân tích video và trả về tracking data + analysis results.
        
        Args:
            max_frames: Số frame tối đa để xử lý (để demo nhanh)
            
        Returns:
            (tracking_df, analysis_results)
        """
        print(f"🔍 Analyzing video: {self.video_path}")
        
        # Check if video exists
        if not os.path.exists(self.video_path):
            print(f"❌ Video không tồn tại: {self.video_path}")
            return pd.DataFrame(), {}
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {self.video_path}")
            return pd.DataFrame(), {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {total_frames} frames, {fps:.2f} FPS")
        print(f"🎯 Processing max {max_frames} frames for demo")
        
        # Court calibration (manual - cần chạy trước)
        court_points_file = "config/court_points.json"
        if not os.path.exists(court_points_file):
            print("⚠️  Court chưa được calibrate. Sử dụng default court points.")
            # Set default court points directly
            self.court_detector.court_points = np.array([
                [100, 100],   # Top-left
                [1180, 100],  # Top-right  
                [1180, 620],  # Bottom-right
                [100, 620]    # Bottom-left
            ], dtype=np.float32)
        else:
            # Load court points from file - get san1 camera data
            import json
            with open(court_points_file, 'r') as f:
                court_data = json.load(f)
                san1_corners = court_data['cameras']['san1']['court_corners']
                # Check if calibrated
                if court_data['cameras']['san1']['calibration_status'] == 'not_calibrated':
                    print("⚠️  san1 camera chưa được calibrate. Sử dụng default court points.")
                    self.court_detector.court_points = np.array([
                        [200, 150],   # Top-left
                        [1080, 150],  # Top-right  
                        [1080, 570],  # Bottom-right
                        [200, 570]    # Bottom-left
                    ], dtype=np.float32)
                else:
                    # Use calibrated points
                    self.court_detector.court_points = np.array([
                        san1_corners['top_left'],
                        san1_corners['top_right'],
                        san1_corners['bottom_right'],
                        san1_corners['bottom_left']
                    ], dtype=np.float32)
        
        # Process frames
        all_detections = []
        frame_count = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            
            timestamp = frame_count / fps
            
            # YOLO detection
            detections = self.detector.detect_frame(frame)
            
            # Update tracking
            tracked_objects = self.tracker.update(detections)
            
            # Convert to tracking records
            for obj_id, obj_data in tracked_objects.items():
                all_detections.append({
                    'frame_id': frame_count,
                    'timestamp': timestamp,
                    'object_id': obj_id,
                    'class': obj_data.get('class', 'person'),
                    'center_x': obj_data['centroid'][0],
                    'center_y': obj_data['centroid'][1],
                    'confidence': obj_data.get('confidence', 0.8),
                    'bbox_x1': obj_data['centroid'][0] - 25,
                    'bbox_y1': obj_data['centroid'][1] - 35, 
                    'bbox_x2': obj_data['centroid'][0] + 25,
                    'bbox_y2': obj_data['centroid'][1] + 35
                })
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                print(f"⏳ Processed {frame_count}/{max_frames} frames ({fps_current:.1f} FPS)")
        
        cap.release()
        
        processing_time = time.time() - start_time
        print(f"✅ Video analysis completed in {processing_time:.2f}s")
        print(f"📊 Found {len(all_detections)} detections")
        
        # Create DataFrame
        tracking_df = pd.DataFrame(all_detections)
        
        # Save tracking data
        tracking_file = os.path.join(self.output_dir, "tracking_data", "real_tracking.csv")
        tracking_df.to_csv(tracking_file, index=False)
        print(f"💾 Tracking data saved: {tracking_file}")
        
        # Run analysis
        if not tracking_df.empty:
            print("🔬 Running movement analysis...")
            analysis_results = self.analyzer.analyze_tracking_data(tracking_df)
            
            # Save analysis results
            analysis_file = os.path.join(self.output_dir, "reports", "real_analysis_results.json")
            import json
            with open(analysis_file, 'w', encoding='utf-8') as f:
                # Convert numpy types for JSON
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
            
            print(f"💾 Analysis results saved: {analysis_file}")
            return tracking_df, analysis_results
        else:
            print("⚠️  No tracking data found")
            return tracking_df, {}

    def create_side_by_side_video(self, tracking_df: pd.DataFrame, 
                                 analysis_results: Dict[str, Any],
                                 max_frames: int = 300) -> str:
        """
        Tạo video side-by-side: gốc vs phân tích.
        
        Args:
            tracking_df: Tracking data
            analysis_results: Analysis results
            max_frames: Số frame tối đa
            
        Returns:
            Đường dẫn video output
        """
        print("🎬 Creating side-by-side comparison video...")
        
        # Input video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return ""
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video (side-by-side)
        output_width = frame_width * 2
        output_height = frame_height
        
        output_path = os.path.join(self.output_dir, "processed_videos", "side_by_side_comparison.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break
            
            # Original frame (left side)
            original_frame = frame.copy()
            
            # Analysis frame (right side)
            analysis_frame = frame.copy()
            
            # Get tracking data for current frame
            frame_detections = tracking_df[tracking_df['frame_id'] == frame_count]
            
            # Draw tracking on analysis frame
            for _, detection in frame_detections.iterrows():
                center_x = int(detection['center_x'])
                center_y = int(detection['center_y'])
                object_id = detection['object_id']
                class_name = detection['class']
                confidence = detection['confidence']
                
                # Choose color based on class
                if class_name == 'player':
                    color = (0, 255, 0)  # Green for players
                elif class_name == 'ball':
                    color = (0, 0, 255)  # Red for ball
                else:
                    color = (255, 255, 0)  # Cyan for others
                
                # Draw bounding box
                x1, y1 = int(detection['bbox_x1']), int(detection['bbox_y1'])
                x2, y2 = int(detection['bbox_x2']), int(detection['bbox_y2'])
                cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw center point
                cv2.circle(analysis_frame, (center_x, center_y), 5, color, -1)
                
                # Draw label
                label = f"{class_name}_{object_id}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(analysis_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(analysis_frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw court points if available
            if hasattr(self.court_detector, 'court_points') and self.court_detector.court_points is not None:
                court_points = self.court_detector.court_points.astype(int)
                cv2.polylines(analysis_frame, [court_points], True, (255, 0, 255), 2)
                
                # Label court corners
                corner_labels = ['TL', 'TR', 'BR', 'BL']
                for i, (point, label) in enumerate(zip(court_points, corner_labels)):
                    cv2.circle(analysis_frame, tuple(point), 8, (255, 0, 255), -1)
                    cv2.putText(analysis_frame, label, (point[0] + 10, point[1]),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Add frame info
            cv2.putText(original_frame, "ORIGINAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(analysis_frame, "ANALYSIS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Frame counter
            cv2.putText(original_frame, f"Frame: {frame_count}", (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(analysis_frame, f"Detections: {len(frame_detections)}", 
                       (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combine frames side by side
            combined_frame = np.hstack([original_frame, analysis_frame])
            
            # Write frame
            out.write(combined_frame)
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"⏳ Created {frame_count}/{max_frames} comparison frames")
        
        cap.release()
        out.release()
        
        print(f"✅ Side-by-side video created: {output_path}")
        return output_path

    def create_live_preview(self, tracking_df: pd.DataFrame, max_frames: int = 300):
        """
        Hiển thị live preview side-by-side.
        
        Args:
            tracking_df: Tracking data
            max_frames: Số frame tối đa
        """
        print("🔴 Starting live preview (Press 'q' to quit, SPACE to pause)")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        
        frame_count = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break
            
            # Original frame (left)
            original_frame = frame.copy()
            
            # Analysis frame (right)
            analysis_frame = frame.copy()
            
            # Get tracking data for current frame
            frame_detections = tracking_df[tracking_df['frame_id'] == frame_count]
            
            # Draw tracking on analysis frame
            for _, detection in frame_detections.iterrows():
                center_x = int(detection['center_x'])
                center_y = int(detection['center_y'])
                object_id = detection['object_id']
                class_name = detection['class']
                
                # Color based on class
                color = (0, 255, 0) if class_name == 'player' else (0, 0, 255)
                
                # Draw detection
                x1, y1 = int(detection['bbox_x1']), int(detection['bbox_y1'])
                x2, y2 = int(detection['bbox_x2']), int(detection['bbox_y2'])
                cv2.rectangle(analysis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(analysis_frame, (center_x, center_y), 5, color, -1)
                
                # Label
                label = f"{class_name}_{object_id}"
                cv2.putText(analysis_frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add titles
            cv2.putText(original_frame, "ORIGINAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(analysis_frame, "TRACKING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Resize for display
            display_width = 640
            display_height = 360
            original_resized = cv2.resize(original_frame, (display_width, display_height))
            analysis_resized = cv2.resize(analysis_frame, (display_width, display_height))
            
            # Show side by side
            cv2.imshow('Original Video', original_resized)
            cv2.imshow('Tracking Analysis', analysis_resized)
            
            if not paused:
                frame_count += 1
            
            # Handle keys
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause/resume
                paused = not paused
                status = "PAUSED" if paused else "PLAYING"
                print(f"📺 {status} at frame {frame_count}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function để chạy real video demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real Video Analysis Demo")
    parser.add_argument('--video', type=str, 
                       default=r"C:\Users\highp\pickerball\video\data_video\san1.mp4",
                       help='Đường dẫn video input')
    parser.add_argument('--max-frames', type=int, default=300,
                       help='Số frame tối đa để xử lý (default: 300)')
    parser.add_argument('--live', action='store_true',
                       help='Hiển thị live preview')
    parser.add_argument('--video-only', action='store_true',
                       help='Chỉ tạo video output, không live preview')
    
    args = parser.parse_args()
    
    print("🏓 PICKLEBALL REAL VIDEO ANALYSIS DEMO")
    print("=" * 50)
    print(f"📹 Video: {args.video}")
    print(f"🎯 Max frames: {args.max_frames}")
    
    # Initialize demo
    demo = RealVideoDemo(args.video)
    
    # Step 1: Analyze video
    tracking_df, analysis_results = demo.analyze_video(args.max_frames)
    
    if tracking_df.empty:
        print("❌ No tracking data found. Exiting.")
        return
    
    # Step 2: Create visualizations
    print("🎨 Creating visualizations...")
    viz_files = demo.visualizer.save_all_visualizations(analysis_results, demo.output_dir)
    
    # Step 3: Create side-by-side video
    if not args.live:
        side_by_side_video = demo.create_side_by_side_video(tracking_df, analysis_results, args.max_frames)
        
        if side_by_side_video and os.path.exists(side_by_side_video):
            file_size = os.path.getsize(side_by_side_video) / (1024 * 1024)  # MB
            print(f"✅ Side-by-side video created: {side_by_side_video} ({file_size:.1f} MB)")
        
    # Step 4: Live preview (if requested)
    if args.live and not args.video_only:
        demo.create_live_preview(tracking_df, args.max_frames)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 REAL VIDEO DEMO COMPLETED!")
    print("=" * 50)
    
    # Analysis summary
    if analysis_results:
        player_analysis = analysis_results.get('player_analysis', {})
        print(f"📈 Analysis Summary:")
        for player_id, data in player_analysis.items():
            distance = data.get('total_distance_meters', 0)
            avg_speed = data.get('avg_speed_kmh', 0)
            max_speed = data.get('max_speed_kmh', 0)
            print(f"   🏃 {player_id}: {distance:.1f}m, avg {avg_speed:.1f} km/h, max {max_speed:.1f} km/h")
    
    print(f"\n📁 All outputs saved in: {demo.output_dir}/")
    print("💡 Next: Review the visualizations and side-by-side video!")

if __name__ == "__main__":
    main()