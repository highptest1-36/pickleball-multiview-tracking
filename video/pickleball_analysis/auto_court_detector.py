import cv2
import numpy as np
import json
from pathlib import Path

class AutoCourtDetector:
    """
    Automatic court detection using edge detection and Hough line transform
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Court dimensions (standard pickleball)
        self.court_width = 6.1  # meters
        self.court_length = 13.41  # meters
        
        # Detection parameters
        self.line_threshold = 100
        self.min_line_length = 50
        self.max_line_gap = 10
        
        print("🏟️ Auto Court Detector Initialized")
        print(f"📹 Video: {Path(video_path).name}")
    
    def detect_green_lines(self, frame):
        """
        Detect green court lines using color filtering
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for green color (court lines are usually white/light on green background)
        # We'll detect bright colors (white/yellow lines)
        lower_bright = np.array([0, 0, 180])
        upper_bright = np.array([180, 70, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bright, upper_bright)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def detect_edges(self, frame):
        """
        Detect edges using Canny edge detection
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        return edges
    
    def find_court_lines(self, frame):
        """
        Find court lines using Hough line transform
        """
        # Method 1: Color-based detection
        mask = self.detect_green_lines(frame)
        
        # Method 2: Edge detection
        edges = self.detect_edges(frame)
        
        # Combine both methods
        combined = cv2.bitwise_or(edges, mask)
        
        # Hough line transform
        lines = cv2.HoughLinesP(
            combined,
            rho=1,
            theta=np.pi/180,
            threshold=self.line_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        return lines, combined
    
    def classify_lines(self, lines):
        """
        Classify lines into horizontal and vertical
        """
        if lines is None:
            return [], []
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Classify by angle
            if angle < 30 or angle > 150:  # Horizontal
                horizontal_lines.append(line[0])
            elif 60 < angle < 120:  # Vertical
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines
    
    def merge_similar_lines(self, lines, threshold=20):
        """
        Merge similar parallel lines
        """
        if not lines:
            return []
        
        merged = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            similar_lines = [line1]
            
            for j, line2 in enumerate(lines[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if lines are similar (parallel and close)
                # Simple check: compare average positions
                avg1 = (line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2
                avg2 = (line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2
                
                distance = np.sqrt((avg1[0] - avg2[0])**2 + (avg1[1] - avg2[1])**2)
                
                if distance < threshold:
                    similar_lines.append(line2)
                    used.add(j)
            
            # Average the similar lines
            if similar_lines:
                avg_line = np.mean(similar_lines, axis=0).astype(int)
                merged.append(avg_line)
        
        return merged
    
    def find_court_corners(self, horizontal_lines, vertical_lines):
        """
        Find court corners from line intersections
        """
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None
        
        # Find top and bottom horizontal lines
        h_lines_sorted = sorted(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
        top_line = h_lines_sorted[0]
        bottom_line = h_lines_sorted[-1]
        
        # Find left and right vertical lines
        v_lines_sorted = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
        left_line = v_lines_sorted[0]
        right_line = v_lines_sorted[-1]
        
        # Calculate intersections
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            return (int(x), int(y))
        
        # Find 4 corners
        top_left = line_intersection(top_line, left_line)
        top_right = line_intersection(top_line, right_line)
        bottom_right = line_intersection(bottom_line, right_line)
        bottom_left = line_intersection(bottom_line, left_line)
        
        if all([top_left, top_right, bottom_right, bottom_left]):
            return [top_left, top_right, bottom_right, bottom_left]
        
        return None
    
    def detect_court_interactive(self, frame_number=100, auto_accept=False):
        """
        Detect court with user validation
        """
        # Seek to specific frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            return None
        
        original = frame.copy()
        
        print(f"\n🔍 Analyzing frame {frame_number}...")
        
        # Find court lines
        lines, combined_edges = self.find_court_lines(frame)
        
        if lines is None or len(lines) == 0:
            print("❌ No lines detected")
            return None
        
        print(f"✅ Found {len(lines)} lines")
        
        # Classify lines
        h_lines, v_lines = self.classify_lines(lines)
        
        print(f"   Horizontal: {len(h_lines)}, Vertical: {len(v_lines)}")
        
        if len(h_lines) < 2 or len(v_lines) < 2:
            print("⚠️ Not enough lines to form court rectangle")
            return None
        
        # Merge similar lines
        h_lines = self.merge_similar_lines(h_lines, threshold=30)
        v_lines = self.merge_similar_lines(v_lines, threshold=30)
        
        print(f"   After merging - Horizontal: {len(h_lines)}, Vertical: {len(v_lines)}")
        
        # Find corners
        corners = self.find_court_corners(h_lines, v_lines)
        
        if corners is None:
            print("❌ Could not find court corners")
            return None
        
        print(f"✅ Found 4 corners")
        
        # Visualization
        vis_frame = original.copy()
        
        # Draw lines
        for line in h_lines:
            cv2.line(vis_frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
        
        for line in v_lines:
            cv2.line(vis_frame, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
        
        # Draw corners
        for i, corner in enumerate(corners):
            cv2.circle(vis_frame, corner, 10, (0, 0, 255), -1)
            cv2.putText(vis_frame, str(i+1), (corner[0]+15, corner[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw court polygon
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(vis_frame, [pts], True, (255, 255, 0), 3)
        
        # Save detection preview
        preview_path = 'auto_detection_preview.jpg'
        cv2.imwrite(preview_path, vis_frame)
        print(f"✅ Preview saved to: {preview_path}")
        
        # Auto-accept mode (for CI/CD or batch processing)
        if auto_accept:
            print("✅ Auto-accepting detection...")
            return corners
        
        # Show results
        cv2.namedWindow('Auto Court Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Auto Court Detection', 1200, 700)
        
        # Side by side comparison
        combined_display = np.hstack([
            cv2.resize(original, (600, 350)),
            cv2.resize(vis_frame, (600, 350))
        ])
        
        cv2.putText(combined_display, "Original", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_display, "Auto Detection", (620, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Auto Court Detection', combined_display)
        
        print("\n✅ Auto detection complete!")
        print("   Press 'y' to accept, 'n' to reject, 'r' to retry with different frame")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('y'):
                cv2.destroyAllWindows()
                return corners
            elif key == ord('n'):
                cv2.destroyAllWindows()
                return None
            elif key == ord('r'):
                cv2.destroyAllWindows()
                new_frame = int(input("Enter new frame number: "))
                return self.detect_court_interactive(new_frame)
    
    def save_calibration(self, corners, output_path='court_calibration_san4.json'):
        """
        Save court calibration to JSON
        """
        # Define court coordinates (standard pickleball court)
        court_points = np.array([
            [0, 0],  # Top-left
            [self.court_width, 0],  # Top-right
            [self.court_width, self.court_length],  # Bottom-right
            [0, self.court_length]  # Bottom-left
        ], dtype=np.float32)
        
        # Calculate homography
        image_points = np.array(corners, dtype=np.float32)
        homography, _ = cv2.findHomography(image_points, court_points)
        
        # Create calibration data
        calibration = {
            'homography': homography.tolist(),
            'court_width': self.court_width,
            'court_length': self.court_length,
            'image_points': image_points.tolist(),
            'court_points': court_points.tolist(),
            'method': 'auto_detection',
            'video_path': str(self.video_path)
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✅ Calibration saved to: {output_path}")
        
        return calibration
    
    def run(self, auto_accept=True):
        """
        Main execution
        """
        print("\n" + "="*60)
        print("🤖 AUTOMATIC COURT DETECTION")
        print("="*60)
        
        # Try to detect court
        corners = self.detect_court_interactive(auto_accept=auto_accept)
        
        if corners is None:
            print("\n❌ Auto detection failed. Please use manual calibration.")
            print("   Run: python recalibrate_court.py")
            return False
        
        # Save calibration
        self.save_calibration(corners)
        
        print("\n" + "="*60)
        print("✅ AUTO CALIBRATION COMPLETE!")
        print("="*60)
        print("\n💡 You can now run any tracking script:")
        print("   python enhanced_tracking_san4.py")
        print("   python advanced_tracking_san4.py")
        print("   python stable_reid_tracking_san4.py")
        
        return True
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main entry point"""
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    
    detector = AutoCourtDetector(video_path)
    success = detector.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
