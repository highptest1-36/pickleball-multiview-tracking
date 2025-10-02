import cv2
import numpy as np
import json
from pathlib import Path

class SmartCourtDetector:
    """
    Smart court detector: Try auto first, fallback to interactive manual
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Court dimensions
        self.court_width = 6.1
        self.court_length = 13.41
        
        # Points for manual selection
        self.points = []
        self.current_frame = None
        
        print("🏟️ Smart Court Detector")
        print(f"📹 Video: {Path(video_path).name}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"✅ Point {len(self.points)}: ({x}, {y})")
            
            # Draw point
            cv2.circle(self.current_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(self.current_frame, str(len(self.points)), 
                       (x+12, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw lines between points
            if len(self.points) > 1:
                cv2.line(self.current_frame, self.points[-2], self.points[-1], 
                        (0, 255, 0), 2)
            
            # Complete rectangle
            if len(self.points) == 4:
                cv2.line(self.current_frame, self.points[3], self.points[0], 
                        (0, 255, 0), 2)
                pts = np.array(self.points, dtype=np.int32)
                cv2.polylines(self.current_frame, [pts], True, (0, 255, 255), 3)
            
            cv2.imshow('Court Selection', self.current_frame)
    
    def try_simple_auto_detection(self, frame):
        """
        Simple auto-detection: Find largest rectangle in frame
        """
        print("\n🤖 Trying simple auto-detection...")
        
        # Convert to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("   No contours found")
            return None
        
        # Find contours that look like rectangles
        h, w = frame.shape[:2]
        min_area = (w * h) * 0.05  # At least 5% of frame
        max_area = (w * h) * 0.9   # At most 90% of frame
        
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Approximate to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Look for 4-sided polygons
                if len(approx) == 4:
                    candidates.append((area, approx))
        
        if not candidates:
            print("   No rectangular contours found")
            return None
        
        # Get largest rectangle
        candidates.sort(key=lambda x: x[0], reverse=True)
        largest_rect = candidates[0][1]
        
        # Convert to list of tuples
        corners = [tuple(pt[0]) for pt in largest_rect]
        
        print(f"✅ Found rectangle candidate (area: {candidates[0][0]:.0f})")
        
        return corners
    
    def manual_selection(self, frame_number=100):
        """
        Manual court selection by clicking 4 corners
        """
        print("\n👆 Manual Court Selection")
        print("="*60)
        print("Click 4 corners of the court in order:")
        print("  1. Top-Left")
        print("  2. Top-Right")
        print("  3. Bottom-Right")
        print("  4. Bottom-Left")
        print("Press 'r' to reset, 'q' to quit")
        print("="*60)
        
        # Get frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            return None
        
        self.current_frame = frame.copy()
        self.points = []
        
        # Create window
        cv2.namedWindow('Court Selection')
        cv2.setMouseCallback('Court Selection', self.mouse_callback)
        cv2.imshow('Court Selection', self.current_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset
                self.points = []
                self.current_frame = frame.copy()
                cv2.imshow('Court Selection', self.current_frame)
                print("🔄 Reset - Click 4 corners again")
            elif len(self.points) == 4 and key == 13:  # Enter key
                break
        
        cv2.destroyAllWindows()
        
        if len(self.points) == 4:
            return self.points
        
        return None
    
    def save_calibration(self, corners, output_path='court_calibration_san4.json'):
        """Save calibration"""
        court_points = np.array([
            [0, 0],
            [self.court_width, 0],
            [self.court_width, self.court_length],
            [0, self.court_length]
        ], dtype=np.float32)
        
        image_points = np.array(corners, dtype=np.float32)
        homography, _ = cv2.findHomography(image_points, court_points)
        
        calibration = {
            'homography': homography.tolist(),
            'court_width': self.court_width,
            'court_length': self.court_length,
            'image_points': image_points.tolist(),
            'court_points': court_points.tolist(),
            'method': 'smart_detection',
            'video_path': str(self.video_path)
        }
        
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✅ Calibration saved to: {output_path}")
        
        # Save preview
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = self.cap.read()
        if ret:
            vis_frame = frame.copy()
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 4)
            
            for i, corner in enumerate(corners):
                cv2.circle(vis_frame, tuple(corner), 10, (0, 0, 255), -1)
                cv2.putText(vis_frame, str(i+1), 
                           (corner[0]+15, corner[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imwrite('auto_detection_preview.jpg', vis_frame)
            print(f"✅ Preview saved to: auto_detection_preview.jpg")
        
        return calibration
    
    def run(self):
        """Main execution"""
        print("\n" + "="*60)
        print("🎯 SMART COURT DETECTION")
        print("="*60)
        
        # Try auto detection first
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = self.cap.read()
        
        if not ret:
            print("❌ Failed to read video")
            return False
        
        corners = self.try_simple_auto_detection(frame)
        
        if corners is not None:
            # Show auto-detected result
            vis = frame.copy()
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 4)
            
            for i, corner in enumerate(corners):
                cv2.circle(vis, tuple(corner), 10, (0, 0, 255), -1)
            
            cv2.namedWindow('Auto Detection Result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Auto Detection Result', 1200, 700)
            cv2.imshow('Auto Detection Result', vis)
            
            print("\n✅ Auto-detection successful!")
            print("   Press 'y' to accept, 'n' for manual selection")
            
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key != ord('y'):
                corners = None
        
        # Fallback to manual if auto failed or rejected
        if corners is None:
            print("\n⚠️ Using manual selection...")
            corners = self.manual_selection(frame_number=100)
        
        if corners is None:
            print("\n❌ No corners selected")
            return False
        
        # Ensure corners are in correct format
        corners = [tuple(int(c) for c in corner) for corner in corners]
        
        # Save calibration
        self.save_calibration(corners)
        
        print("\n" + "="*60)
        print("✅ CALIBRATION COMPLETE!")
        print("="*60)
        print("\n💡 You can now run tracking scripts:")
        print("   python enhanced_tracking_san4.py")
        print("   python advanced_tracking_san4.py")
        print("   python stable_reid_tracking_san4.py")
        
        return True
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    
    detector = SmartCourtDetector(video_path)
    success = detector.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
