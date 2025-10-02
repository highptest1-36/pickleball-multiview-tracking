"""
Simple Court Calibration Tool - Không bị che bởi overlay

Tool đơn giản để chọn điểm trên sân mà không bị che bởi bảng thông tin.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict, Any
import argparse

class SimpleCourtCalibrator:
    def __init__(self, video_path: str):
        """
        Initialize simple court calibrator.
        
        Args:
            video_path: Đường dẫn video để calibrate
        """
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Court points
        self.court_points = []
        self.frame = None
        self.original_frame = None
        
        # Point labels
        self.point_labels = [
            "Top-Left Corner",
            "Top-Right Corner", 
            "Bottom-Right Corner",
            "Bottom-Left Corner",
            "Net Left",
            "Net Right", 
            "Net Center",
            "Service Line Top-Left",
            "Service Line Top-Right",
            "Service Line Bottom-Left", 
            "Service Line Bottom-Right"
        ]
        
        self.current_point_index = 0
        
        # Target court dimensions
        self.target_width = 1341
        self.target_height = 610
        
        # Target points (bird's-eye view)
        self.target_points = [
            [50, 50],                           # Top-left corner
            [1291, 50],                         # Top-right corner
            [1291, 560],                        # Bottom-right corner
            [50, 560],                          # Bottom-left corner
            [50, 305],                          # Net left
            [1291, 305],                        # Net right
            [670, 305],                         # Net center
            [50, 263],                          # Service top-left
            [1291, 263],                        # Service top-right
            [50, 347],                          # Service bottom-left
            [1291, 347]                         # Service bottom-right
        ]
        
        self.window_name = f"Court Calibration - {self.video_name}"
        
        print(f"🎯 Simple Court Calibrator - {self.video_name}")

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback để click chọn điểm."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point_index < len(self.point_labels):
                self.court_points.append([x, y])
                
                print(f"📍 {self.point_labels[self.current_point_index]}: ({x}, {y})")
                
                # Draw point
                cv2.circle(self.frame, (x, y), 6, (0, 255, 0), -1)
                cv2.circle(self.frame, (x, y), 8, (0, 255, 0), 2)
                cv2.putText(self.frame, str(self.current_point_index + 1), 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                self.current_point_index += 1
                
                # Update display
                self.update_display()
                
                if self.current_point_index >= len(self.point_labels):
                    print("✅ All points selected! Press 's' to save, 'r' to reset")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to undo last point
            if self.court_points:
                self.court_points.pop()
                self.current_point_index -= 1
                
                # Redraw frame
                self.frame = self.original_frame.copy()
                self.draw_existing_points()
                self.update_display()
                
                print(f"🗑️  Removed last point. Current: {self.current_point_index}")

    def draw_existing_points(self):
        """Vẽ lại tất cả điểm đã chọn."""
        for i, point in enumerate(self.court_points):
            cv2.circle(self.frame, tuple(point), 6, (0, 255, 0), -1)
            cv2.circle(self.frame, tuple(point), 8, (0, 255, 0), 2)
            cv2.putText(self.frame, str(i + 1), 
                       (point[0] + 10, point[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def update_display(self):
        """Update display với thông tin minimal ở góc."""
        display_frame = self.frame.copy()
        
        # Small info box ở góc phải dưới - KHÔNG che video
        h, w = display_frame.shape[:2]
        box_w = 300
        box_h = 80
        box_x = w - box_w - 10
        box_y = h - box_h - 10
        
        # Semi-transparent background
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(display_frame, 0.8, overlay, 0.2, 0)
        
        # Border
        cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
        
        # Text
        if self.current_point_index < len(self.point_labels):
            current_label = self.point_labels[self.current_point_index]
            cv2.putText(display_frame, f"Click: {current_label}", 
                       (box_x + 10, box_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(display_frame, "All points selected!", 
                       (box_x + 10, box_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(display_frame, f"Points: {len(self.court_points)}/{len(self.point_labels)}", 
                   (box_x + 10, box_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(display_frame, "Right-click: Undo | 's': Save | 'r': Reset", 
                   (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow(self.window_name, display_frame)

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
        
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n🖱️  CALIBRATION INSTRUCTIONS:")
        print("══════════════════════════════════════")
        print("1. Click theo thứ tự các điểm:")
        for i, label in enumerate(self.point_labels):
            print(f"   {i+1}. {label}")
        print("2. Right-click để undo điểm cuối")
        print("3. 's' để save, 'r' để reset, 'q' để quit")
        print("══════════════════════════════════════")
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("🚪 Calibration cancelled")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('r'):
                print("🔄 Reset all points")
                self.court_points = []
                self.current_point_index = 0
                self.frame = self.original_frame.copy()
                self.update_display()
            
            elif key == ord('s'):
                if len(self.court_points) >= 4:
                    print("💾 Saving calibration...")
                    success = self.save_calibration()
                    if success:
                        print("✅ Calibration saved successfully!")
                        cv2.destroyAllWindows()
                        return True
                    else:
                        print("❌ Failed to save calibration")
                else:
                    print(f"⚠️  Need at least 4 points (current: {len(self.court_points)})")
        
        cv2.destroyAllWindows()
        return False

    def calculate_homography(self) -> Optional[np.ndarray]:
        """
        Tính homography matrix từ các điểm đã chọn.
        
        Returns:
            Homography matrix hoặc None
        """
        if len(self.court_points) < 4:
            return None
        
        # Use only selected points that have corresponding targets
        num_points = min(len(self.court_points), len(self.target_points))
        
        source_pts = np.array(self.court_points[:num_points], dtype=np.float32)
        target_pts = np.array(self.target_points[:num_points], dtype=np.float32)
        
        if num_points == 4:
            homography = cv2.getPerspectiveTransform(source_pts, target_pts)
        else:
            homography, _ = cv2.findHomography(source_pts, target_pts, cv2.RANSAC)
        
        return homography

    def save_calibration(self) -> bool:
        """
        Save calibration data.
        
        Returns:
            True nếu save thành công
        """
        try:
            # Calculate homography
            homography = self.calculate_homography()
            if homography is None:
                print("❌ Cannot calculate homography")
                return False
            
            # Load existing config
            config_file = "config/court_points.json"
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Map first 4 points to corners
            camera_key = self.video_name
            court_corners = {
                "top_left": self.court_points[0] if len(self.court_points) > 0 else [0, 0],
                "top_right": self.court_points[1] if len(self.court_points) > 1 else [0, 0],
                "bottom_right": self.court_points[2] if len(self.court_points) > 2 else [0, 0],
                "bottom_left": self.court_points[3] if len(self.court_points) > 3 else [0, 0]
            }
            
            # Create detailed point mapping
            point_mapping = {}
            for i, point in enumerate(self.court_points):
                if i < len(self.point_labels):
                    point_mapping[self.point_labels[i]] = point
            
            config['cameras'][camera_key] = {
                "description": f"Camera {camera_key} (simple calibration)",
                "court_corners": court_corners,
                "calibration_status": "calibrated", 
                "calibration_method": "simple_multi_point",
                "points_used": len(self.court_points),
                "detailed_points": point_mapping,
                "homography_matrix": homography.tolist(),
                "notes": f"Simple calibration with {len(self.court_points)} points"
            }
            
            # Save
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Calibration saved: {config_file}")
            print(f"🎯 Camera: {camera_key}")
            print(f"📊 Points: {len(self.court_points)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving: {e}")
            return False

    def test_calibration(self, max_frames: int = 50):
        """Test calibration với video."""
        homography = self.calculate_homography()
        if homography is None:
            print("❌ No homography to test")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        
        print("🧪 Testing calibration... Press 'q' to quit")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply transformation
            transformed = cv2.warpPerspective(frame, homography,
                                            (self.target_width, self.target_height))
            
            # Resize for display
            orig_small = cv2.resize(frame, (600, 400))
            trans_small = cv2.resize(transformed, (600, 400))
            
            # Labels
            cv2.putText(orig_small, "ORIGINAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(trans_small, "TRANSFORMED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show side by side
            combined = np.hstack([orig_small, trans_small])
            cv2.imshow("Calibration Test", combined)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple Court Calibration Tool")
    parser.add_argument('--video', type=str, required=True,
                       help='Đường dẫn video để calibrate')
    parser.add_argument('--test', action='store_true',
                       help='Test existing calibration')
    
    args = parser.parse_args()
    
    calibrator = SimpleCourtCalibrator(args.video)
    
    if args.test:
        calibrator.test_calibration()
    else:
        success = calibrator.calibrate_court()
        if success:
            test_choice = input("🧪 Test calibration? (y/n): ")
            if test_choice.lower() == 'y':
                calibrator.test_calibration()

if __name__ == "__main__":
    main()