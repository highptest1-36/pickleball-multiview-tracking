import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

class AutoCourtDetector:
    """
    Automatic court detection using improved edge detection and geometric constraints
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Court dimensions (standard pickleball)
        self.court_width = 6.1  # meters
        self.court_length = 13.41  # meters
        
        # Detection parameters - IMPROVED
        self.line_threshold = 80  # Adjusted for balance
        self.min_line_length = 80  # Reasonable length
        self.max_line_gap = 20
        
        print("🏟️ Improved Auto Court Detector Initialized")
        print(f"📹 Video: {Path(video_path).name}")
    
    def get_roi_mask(self, frame):
        """
        Create ROI mask to focus on center area where court likely is
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Focus on center 80% of frame (court usually not at edges)
        x_margin = int(w * 0.1)
        y_margin = int(h * 0.1)
        
        cv2.rectangle(mask, 
                     (x_margin, y_margin), 
                     (w - x_margin, h - y_margin), 
                     255, -1)
        
        return mask
    
    def detect_green_court_area(self, frame):
        """
        Detect green court area (grass/synthetic surface)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for green court surface
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green areas
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Clean up
        kernel = np.ones((15, 15), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest green contour (likely the court)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create mask from largest contour
            court_mask = np.zeros_like(green_mask)
            cv2.drawContours(court_mask, [largest_contour], -1, 255, -1)
            
            return court_mask
        
        return green_mask
    
    def detect_white_lines(self, frame, court_mask=None):
        """
        Detect white court lines with better filtering
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask if provided
        if court_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=court_mask)
        
        # Enhance white lines
        # Use adaptive threshold to handle lighting variations
        adaptive = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            blockSize=15, 
            C=-2
        )
        
        # Also use simple threshold for bright areas
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Combine both methods
        combined = cv2.bitwise_or(adaptive, bright)
        
        # Morphological operations to enhance lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_v)
        
        lines = cv2.bitwise_or(horizontal, vertical)
        
        return lines
    
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
        Find court lines with improved filtering
        """
        # Step 1: Get ROI mask
        roi_mask = self.get_roi_mask(frame)
        
        # Step 2: Detect green court area
        court_mask = self.detect_green_court_area(frame)
        
        # Combine ROI and court mask
        combined_mask = cv2.bitwise_and(roi_mask, court_mask)
        
        # Step 3: Detect white lines
        line_mask = self.detect_white_lines(frame, combined_mask)
        
        # Step 4: Apply Canny for edges
        edges = cv2.Canny(line_mask, 50, 150, apertureSize=3)
        
        # Step 5: Hough line transform with STRICTER parameters
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.line_threshold,  # 150
            minLineLength=self.min_line_length,  # 100
            maxLineGap=self.max_line_gap  # 20
        )
        
        return lines, edges, combined_mask
    
    def filter_lines_by_length(self, lines, min_length=150):
        """
        Filter lines by minimum length
        """
        if lines is None:
            return []
        
        filtered = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length >= min_length:
                filtered.append(line[0])
        
        return filtered
    
    def classify_lines(self, lines):
        """
        Classify lines into horizontal and vertical with STRICT angle threshold
        """
        if lines is None or len(lines) == 0:
            return [], []
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0] if isinstance(line[0], np.ndarray) else line
            
            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # STRICTER classification (only near horizontal/vertical)
            if angle < 15 or angle > 165:  # Horizontal (within 15° of horizontal)
                horizontal_lines.append([x1, y1, x2, y2])
            elif 75 < angle < 105:  # Vertical (within 15° of vertical)
                vertical_lines.append([x1, y1, x2, y2])
        
        return horizontal_lines, vertical_lines
    
    def cluster_parallel_lines(self, lines, threshold=50):
        """
        Cluster parallel lines and return representative lines using RANSAC-like approach
        """
        if not lines:
            return []
        
        # For horizontal lines, cluster by Y position
        # For vertical lines, cluster by X position
        
        # Determine if lines are horizontal or vertical
        first_line = lines[0]
        angle = np.abs(np.arctan2(first_line[3] - first_line[1], 
                                  first_line[2] - first_line[0]) * 180 / np.pi)
        
        is_horizontal = angle < 45 or angle > 135
        
        # Get positions
        if is_horizontal:
            positions = [(l[1] + l[3]) / 2 for l in lines]  # Y position
        else:
            positions = [(l[0] + l[2]) / 2 for l in lines]  # X position
        
        # Cluster positions
        positions = np.array(positions)
        clusters = []
        used = set()
        
        for i, pos in enumerate(positions):
            if i in used:
                continue
            
            # Find all lines within threshold
            cluster_lines = []
            for j, pos2 in enumerate(positions):
                if abs(pos - pos2) < threshold:
                    cluster_lines.append(lines[j])
                    used.add(j)
            
            if cluster_lines:
                # Get the longest line from cluster
                longest = max(cluster_lines, 
                            key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2))
                clusters.append(longest)
        
        return clusters
    
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
    
    def find_court_corners(self, horizontal_lines, vertical_lines, frame_shape):
        """
        Find court corners with geometric validation
        """
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None
        
        # Find top and bottom horizontal lines (should be far apart)
        h_lines_sorted = sorted(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
        
        # Select lines that span significant portion of frame
        h, w = frame_shape[:2]
        min_h_length = w * 0.3  # At least 30% of frame width
        
        valid_h_lines = [l for l in h_lines_sorted 
                        if np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) > min_h_length]
        
        if len(valid_h_lines) < 2:
            return None
        
        top_line = valid_h_lines[0]
        bottom_line = valid_h_lines[-1]
        
        # Check if lines are far enough apart (at least 20% of frame height)
        y_top = (top_line[1] + top_line[3]) / 2
        y_bottom = (bottom_line[1] + bottom_line[3]) / 2
        
        if abs(y_bottom - y_top) < h * 0.2:
            return None
        
        # Find left and right vertical lines (should be far apart)
        v_lines_sorted = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
        
        min_v_length = h * 0.2  # At least 20% of frame height
        
        valid_v_lines = [l for l in v_lines_sorted 
                        if np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2) > min_v_length]
        
        if len(valid_v_lines) < 2:
            return None
        
        left_line = valid_v_lines[0]
        right_line = valid_v_lines[-1]
        
        # Check if lines are far enough apart (at least 15% of frame width)
        x_left = (left_line[0] + left_line[2]) / 2
        x_right = (right_line[0] + right_line[2]) / 2
        
        if abs(x_right - x_left) < w * 0.15:
            return None
        
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
        
        if not all([top_left, top_right, bottom_right, bottom_left]):
            return None
        
        # Validate corners are within frame
        corners = [top_left, top_right, bottom_right, bottom_left]
        for corner in corners:
            if corner[0] < 0 or corner[0] >= w or corner[1] < 0 or corner[1] >= h:
                return None
        
        # Validate court shape (should be roughly rectangular)
        # Check aspect ratio is reasonable (pickleball court is ~2.2:1)
        width1 = np.linalg.norm(np.array(top_right) - np.array(top_left))
        width2 = np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))
        height1 = np.linalg.norm(np.array(bottom_left) - np.array(top_left))
        height2 = np.linalg.norm(np.array(bottom_right) - np.array(top_right))
        
        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2
        
        if avg_width < 1 or avg_height < 1:
            return None
        
        aspect_ratio = avg_height / avg_width
        
        # Pickleball court is 13.41m x 6.1m = 2.2:1
        # With perspective, accept 1.0 to 4.0
        if aspect_ratio < 1.0 or aspect_ratio > 4.0:
            print(f"⚠️ Invalid aspect ratio: {aspect_ratio:.2f} (expected 1.0-4.0)")
            return None
        
        print(f"✅ Valid court shape - Aspect ratio: {aspect_ratio:.2f}")
        
        return corners
    
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
        
        # Find court lines with improved method
        lines, edges, court_mask = self.find_court_lines(frame)
        
        if lines is None or len(lines) == 0:
            print("❌ No lines detected")
            print("💡 Try: Different frame, better lighting, or manual calibration")
            return None
        
        print(f"✅ Found {len(lines)} raw lines")
        
        # Filter by length
        lines = self.filter_lines_by_length(lines, min_length=100)
        print(f"   After length filter: {len(lines)} lines")
        
        if len(lines) < 4:
            print("❌ Not enough long lines detected")
            return None
        
        # Classify lines
        h_lines, v_lines = self.classify_lines([[l] for l in lines])
        
        print(f"   Horizontal: {len(h_lines)}, Vertical: {len(v_lines)}")
        
        if len(h_lines) < 2 or len(v_lines) < 2:
            print("⚠️ Not enough horizontal/vertical lines")
            return None
        
        # Cluster parallel lines
        h_lines = self.cluster_parallel_lines(h_lines, threshold=50)
        v_lines = self.cluster_parallel_lines(v_lines, threshold=50)
        
        print(f"   After clustering - Horizontal: {len(h_lines)}, Vertical: {len(v_lines)}")
        
        # Find corners with geometric validation
        corners = self.find_court_corners(h_lines, v_lines, frame.shape)
        
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
    
    def run(self, auto_accept=True, frame_number=100):
        """
        Main execution
        """
        print("\n" + "="*60)
        print("🤖 AUTOMATIC COURT DETECTION")
        print("="*60)
        print(f"📍 Trying frame {frame_number}...")
        
        # Try multiple frames if first fails
        frames_to_try = [frame_number, 50, 150, 200, 300]
        
        corners = None
        for frame_num in frames_to_try:
            print(f"\n🎯 Attempting frame {frame_num}...")
            corners = self.detect_court_interactive(frame_num, auto_accept=auto_accept)
            if corners is not None:
                break
            print(f"   Frame {frame_num} failed, trying next...")
        
        if corners is None:
            print("\n❌ Auto detection failed on all frames. Please use manual calibration.")
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
