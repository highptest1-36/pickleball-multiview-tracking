"""
Court Calibration Tool - Hiệu chỉnh sân pickleball cho perspective transformation

Tool này cho phép:
1. Calibrate 4 góc sân từ video
2. Tạo homography transformation từ camera view sang bird's-eye view
3. Test transformation với video
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Optional
import argparse

class CourtCalibrator:
    def __init__(self, video_path: str):
        """
        Initialize court calibrator.
        
        Args:
            video_path: Đường dẫn video để calibrate
        """
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Court points (sẽ được click bởi user)
        self.court_points = []
        self.frame = None
        
        # Target court dimensions (bird's-eye view)
        self.target_width = 1341  # 13.41m * 100 pixels/m
        self.target_height = 610  # 6.1m * 100 pixels/m
        
        # Target court corners (bird's-eye view)
        self.target_corners = np.array([
            [50, 50],                                    # Top-left
            [self.target_width - 50, 50],               # Top-right
            [self.target_width - 50, self.target_height - 50],  # Bottom-right
            [50, self.target_height - 50]               # Bottom-left
        ], dtype=np.float32)
        
        # Window name
        self.window_name = f"Court Calibration - {self.video_name}"
        
        print(f"🎯 Court Calibrator initialized for: {video_path}")
        print(f"📐 Target court size: {self.target_width}x{self.target_height} pixels")

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback để click chọn 4 góc sân.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.court_points) < 4:
                self.court_points.append([x, y])
                print(f"📍 Point {len(self.court_points)}: ({x}, {y})")
                
                # Draw point on frame
                cv2.circle(self.frame, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(self.frame, f"P{len(self.court_points)}", (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if len(self.court_points) == 4:
                    # Draw court polygon
                    court_pts = np.array(self.court_points, dtype=np.int32)
                    cv2.polylines(self.frame, [court_pts], True, (255, 0, 0), 3)
                    print("✅ All 4 points selected! Press 's' to save, 'r' to reset")
                
                cv2.imshow(self.window_name, self.frame)

    def calibrate_court(self) -> bool:
        """
        Interactive court calibration.
        
        Returns:
            True nếu calibration thành công
        """
        print("🎬 Opening video for calibration...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return False
        
        # Get first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Cannot read first frame")
            return False
        
        self.frame = frame.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n🖱️  CALIBRATION INSTRUCTIONS:")
        print("1. Click 4 corners của sân theo thứ tự:")
        print("   - Top-left (góc trái trên)")
        print("   - Top-right (góc phải trên)")  
        print("   - Bottom-right (góc phải dưới)")
        print("   - Bottom-left (góc trái dưới)")
        print("2. Press 's' để save calibration")
        print("3. Press 'r' để reset và chọn lại")
        print("4. Press 'q' để quit")
        
        cv2.imshow(self.window_name, self.frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("🚪 Calibration cancelled")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('r'):
                print("🔄 Reset calibration")
                self.court_points = []
                self.frame = frame.copy()
                cv2.imshow(self.window_name, self.frame)
            
            elif key == ord('s') and len(self.court_points) == 4:
                print("💾 Saving calibration...")
                success = self.save_calibration()
                if success:
                    print("✅ Calibration saved successfully!")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("❌ Failed to save calibration")
        
        cv2.destroyAllWindows()
        return False

    def save_calibration(self) -> bool:
        """
        Save calibration data to JSON file.
        
        Returns:
            True nếu save thành công
        """
        try:
            # Load existing config
            config_file = "config/court_points.json"
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Update specific camera data
            camera_key = self.video_name  # e.g., "san1"
            
            config['cameras'][camera_key] = {
                "description": f"Camera {camera_key} (calibrated)",
                "court_corners": {
                    "top_left": self.court_points[0],
                    "top_right": self.court_points[1],
                    "bottom_right": self.court_points[2], 
                    "bottom_left": self.court_points[3]
                },
                "calibration_status": "calibrated",
                "notes": f"Calibrated on {np.datetime64('today')}"
            }
            
            # Calculate and test homography
            source_points = np.array(self.court_points, dtype=np.float32)
            homography_matrix, _ = cv2.findHomography(source_points, self.target_corners)
            
            # Save homography matrix
            config['cameras'][camera_key]['homography_matrix'] = homography_matrix.tolist()
            
            # Save back to file
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Calibration saved to: {config_file}")
            print(f"🎯 Camera: {camera_key}")
            print(f"📐 Court points: {self.court_points}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False

    def test_calibration(self, max_frames: int = 100) -> bool:
        """
        Test calibration bằng cách transform một số frames.
        
        Args:
            max_frames: Số frames để test
            
        Returns:
            True nếu test thành công
        """
        try:
            # Load calibration data
            config_file = "config/court_points.json"
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            camera_key = self.video_name
            if camera_key not in config['cameras']:
                print(f"❌ No calibration found for {camera_key}")
                return False
            
            camera_config = config['cameras'][camera_key]
            if camera_config['calibration_status'] != 'calibrated':
                print(f"❌ Camera {camera_key} not calibrated")
                return False
            
            # Get homography matrix
            homography_matrix = np.array(camera_config['homography_matrix'], dtype=np.float32)
            
            print(f"🧪 Testing calibration for {camera_key}...")
            print("Press 'q' to quit, SPACE to pause/resume")
            
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return False
            
            frame_count = 0
            paused = False
            
            while frame_count < max_frames:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                # Apply homography transformation
                transformed = cv2.warpPerspective(frame, homography_matrix, 
                                                (self.target_width, self.target_height))
                
                # Draw court lines on transformed view
                court_frame = self.draw_court_lines(transformed)
                
                # Resize for display
                display_original = cv2.resize(frame, (600, 400))
                display_transformed = cv2.resize(court_frame, (600, 400))
                
                # Add labels
                cv2.putText(display_original, "ORIGINAL VIEW", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_transformed, "BIRD'S-EYE VIEW", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show side by side
                combined = np.hstack([display_original, display_transformed])
                cv2.imshow("Calibration Test", combined)
                
                if not paused:
                    frame_count += 1
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    status = "PAUSED" if paused else "PLAYING"
                    print(f"📺 {status} at frame {frame_count}")
            
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"✅ Calibration test completed ({frame_count} frames)")
            return True
            
        except Exception as e:
            print(f"❌ Error testing calibration: {e}")
            return False

    def draw_court_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Vẽ lines sân pickleball trên bird's-eye view.
        
        Args:
            frame: Frame đã được transform
            
        Returns:
            Frame với court lines
        """
        court_frame = frame.copy()
        
        # Court dimensions in pixels (13.41m x 6.1m at 100 pixels/m)
        court_w = self.target_width - 100  # Trừ padding
        court_h = self.target_height - 100
        
        start_x = 50
        start_y = 50
        
        # Outer boundary
        cv2.rectangle(court_frame, (start_x, start_y), 
                     (start_x + court_w, start_y + court_h), (255, 255, 255), 2)
        
        # Center line (net)
        center_y = start_y + court_h // 2
        cv2.line(court_frame, (start_x, center_y), 
                (start_x + court_w, center_y), (255, 255, 255), 3)
        
        # Service areas
        service_line_distance = int(2.13 * 100)  # 2.13m from net
        
        # Service lines
        cv2.line(court_frame, (start_x, start_y + service_line_distance),
                (start_x + court_w, start_y + service_line_distance), (255, 255, 255), 2)
        cv2.line(court_frame, (start_x, start_y + court_h - service_line_distance),
                (start_x + court_w, start_y + court_h - service_line_distance), (255, 255, 255), 2)
        
        # Center service line
        center_x = start_x + court_w // 2
        cv2.line(court_frame, (center_x, start_y), (center_x, start_y + service_line_distance), (255, 255, 255), 2)
        cv2.line(court_frame, (center_x, start_y + court_h - service_line_distance), 
                (center_x, start_y + court_h), (255, 255, 255), 2)
        
        # Non-volley zone (kitchen) - 2.13m from net
        kitchen_color = (0, 255, 255)  # Yellow
        cv2.rectangle(court_frame, (start_x, center_y - service_line_distance//2), 
                     (start_x + court_w, center_y + service_line_distance//2), kitchen_color, 2)
        
        return court_frame

def main():
    """Main calibration function."""
    parser = argparse.ArgumentParser(description="Court Calibration Tool")
    parser.add_argument('--video', type=str, required=True,
                       help='Đường dẫn video để calibrate')
    parser.add_argument('--test', action='store_true',
                       help='Test existing calibration')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Max frames for testing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    calibrator = CourtCalibrator(args.video)
    
    if args.test:
        # Test existing calibration
        success = calibrator.test_calibration(args.max_frames)
        if success:
            print("✅ Calibration test passed!")
        else:
            print("❌ Calibration test failed!")
    else:
        # Perform new calibration
        success = calibrator.calibrate_court()
        if success:
            print("✅ Court calibration completed!")
            
            # Offer to test
            test_input = input("🧪 Test calibration now? (y/n): ")
            if test_input.lower() == 'y':
                calibrator.test_calibration()
        else:
            print("❌ Court calibration failed!")

if __name__ == "__main__":
    main()