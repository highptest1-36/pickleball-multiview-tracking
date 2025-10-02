import cv2
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import RANSACRegressor

class MultiPointCourtSelector:
    """
    Court selector: Click multiple points along court boundary
    System will automatically fit best rectangle
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Court dimensions
        self.court_width = 6.1
        self.court_length = 13.41
        
        # Points for selection
        self.points = []
        self.current_frame = None
        self.display_frame = None
        
        print("🏟️ Multi-Point Court Selector")
        print(f"📹 Video: {Path(video_path).name}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"✅ Point {len(self.points)}: ({x}, {y})")
            
            # Redraw
            self.update_display()
    
    def update_display(self):
        """Update display with current points and fitted rectangle"""
        self.display_frame = self.current_frame.copy()
        
        # Draw all points
        for i, pt in enumerate(self.points):
            cv2.circle(self.display_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(self.display_frame, str(i+1), 
                       (pt[0]+10, pt[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw lines between consecutive points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(self.display_frame, self.points[i], self.points[i+1], 
                        (255, 255, 0), 1)
        
        # Try to fit rectangle if we have enough points
        if len(self.points) >= 4:
            corners = self.fit_rectangle()
            if corners is not None:
                # Draw fitted rectangle
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(self.display_frame, [pts], True, (0, 255, 255), 3)
                
                # Draw corner numbers
                labels = ['TL', 'TR', 'BR', 'BL']
                for i, (corner, label) in enumerate(zip(corners, labels)):
                    cv2.circle(self.display_frame, tuple(corner), 8, (0, 0, 255), -1)
                    cv2.putText(self.display_frame, label, 
                               (corner[0]+12, corner[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show info
                info_text = f"Points: {len(self.points)} | Rectangle fitted!"
                cv2.putText(self.display_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            info_text = f"Points: {len(self.points)}/4+ | Keep clicking..."
            cv2.putText(self.display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow('Multi-Point Court Selection', self.display_frame)
    
    def fit_rectangle(self):
        """
        Fit best rectangle from multiple points
        Uses RANSAC for robust fitting
        """
        if len(self.points) < 4:
            return None
        
        points = np.array(self.points)
        
        # Method 1: Use minimum area rectangle
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Sort corners: TL, TR, BR, BL
        # Find center
        center = np.mean(box, axis=0)
        
        # Sort by angle from center
        def angle_from_center(pt):
            return np.arctan2(pt[1] - center[1], pt[0] - center[0])
        
        sorted_corners = sorted(box, key=angle_from_center)
        
        # Determine which is top-left based on position
        # Top-left should have smallest y and smallest x
        y_coords = [pt[1] for pt in sorted_corners]
        top_two_indices = np.argsort(y_coords)[:2]
        top_two = [sorted_corners[i] for i in top_two_indices]
        
        # Among top two, left one is top-left
        if top_two[0][0] < top_two[1][0]:
            tl_idx = top_two_indices[0]
        else:
            tl_idx = top_two_indices[1]
        
        # Reorder starting from top-left, going clockwise
        corners = []
        for i in range(4):
            idx = (tl_idx + i) % 4
            corners.append(tuple(sorted_corners[idx]))
        
        # Validate rectangle shape
        if not self.validate_rectangle(corners):
            # Try alternative method: fit lines to points
            return self.fit_rectangle_from_lines()
        
        return corners
    
    def fit_rectangle_from_lines(self):
        """
        Alternative method: Cluster points into 4 sides, fit lines, find intersections
        """
        if len(self.points) < 8:
            return None
        
        points = np.array(self.points)
        
        # Use k-means to cluster points into 4 groups (4 sides)
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=4, random_state=0, n_init=10).fit(points)
            labels = kmeans.labels_
            
            # Fit line to each cluster
            lines = []
            for i in range(4):
                cluster_points = points[labels == i]
                
                if len(cluster_points) < 2:
                    continue
                
                # Fit line using least squares
                [vx, vy, x, y] = cv2.fitLine(cluster_points, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Convert to line endpoints (extend far)
                scale = 2000
                x1 = int(x - vx * scale)
                y1 = int(y - vy * scale)
                x2 = int(x + vx * scale)
                y2 = int(y + vy * scale)
                
                lines.append([x1, y1, x2, y2])
            
            if len(lines) < 4:
                return None
            
            # Find intersections between lines
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
            
            # Find all intersections
            intersections = []
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    inter = line_intersection(lines[i], lines[j])
                    if inter is not None:
                        intersections.append(inter)
            
            if len(intersections) < 4:
                return None
            
            # Select 4 corners that form largest area
            from itertools import combinations
            
            max_area = 0
            best_corners = None
            
            for combo in combinations(intersections, 4):
                area = cv2.contourArea(np.array(combo))
                if area > max_area:
                    max_area = area
                    best_corners = combo
            
            if best_corners is None:
                return None
            
            # Sort corners properly
            best_corners = list(best_corners)
            center = np.mean(best_corners, axis=0)
            
            def angle_from_center(pt):
                return np.arctan2(pt[1] - center[1], pt[0] - center[0])
            
            sorted_corners = sorted(best_corners, key=angle_from_center)
            
            # Find top-left
            y_coords = [pt[1] for pt in sorted_corners]
            top_two_indices = np.argsort(y_coords)[:2]
            top_two = [sorted_corners[i] for i in top_two_indices]
            
            if top_two[0][0] < top_two[1][0]:
                tl_idx = top_two_indices[0]
            else:
                tl_idx = top_two_indices[1]
            
            corners = []
            for i in range(4):
                idx = (tl_idx + i) % 4
                corners.append(tuple(sorted_corners[idx]))
            
            return corners
            
        except Exception as e:
            print(f"⚠️ Line fitting error: {e}")
            return None
    
    def validate_rectangle(self, corners):
        """Validate if corners form a reasonable rectangle"""
        if len(corners) != 4:
            return False
        
        # Check if all corners are within frame
        # (We'll check this later with actual frame dimensions)
        
        # Check if area is reasonable
        area = cv2.contourArea(np.array(corners))
        if area < 1000:  # Too small
            return False
        
        return True
    
    def manual_selection(self, frame_number=100):
        """
        Multi-point manual court selection
        """
        print("\n👆 Multi-Point Court Selection")
        print("="*60)
        print("Click 4-20 points along the court boundary")
        print("(More points = better fit)")
        print("")
        print("Suggested: Click along each side of the court:")
        print("  - Top edge: 2-3 points")
        print("  - Right edge: 2-3 points")
        print("  - Bottom edge: 2-3 points")
        print("  - Left edge: 2-3 points")
        print("")
        print("Controls:")
        print("  Left Click  - Add point")
        print("  'r'         - Reset points")
        print("  'u'         - Undo last point")
        print("  Enter       - Accept fitted rectangle")
        print("  'q'         - Quit")
        print("="*60)
        
        # Get frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            return None
        
        self.current_frame = frame.copy()
        self.display_frame = frame.copy()
        self.points = []
        
        # Create window
        cv2.namedWindow('Multi-Point Court Selection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Multi-Point Court Selection', 1400, 900)
        cv2.setMouseCallback('Multi-Point Court Selection', self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
                
            elif key == ord('r'):
                # Reset
                self.points = []
                self.update_display()
                print("🔄 Reset - Click points again")
                
            elif key == ord('u'):
                # Undo last point
                if self.points:
                    removed = self.points.pop()
                    self.update_display()
                    print(f"↩️  Undo - Removed point: {removed}")
                    
            elif key == 13 or key == 10:  # Enter
                if len(self.points) >= 4:
                    corners = self.fit_rectangle()
                    if corners is not None:
                        cv2.destroyAllWindows()
                        return corners
                    else:
                        print("⚠️ Could not fit rectangle. Add more points or adjust.")
                else:
                    print(f"⚠️ Need at least 4 points (currently {len(self.points)})")
        
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
            'method': 'multi_point_selection',
            'num_points_used': len(self.points),
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
            
            # Draw all clicked points
            for i, pt in enumerate(self.points):
                cv2.circle(vis_frame, pt, 4, (255, 255, 0), -1)
            
            # Draw fitted rectangle
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 4)
            
            # Draw corners
            labels = ['TL', 'TR', 'BR', 'BL']
            for i, (corner, label) in enumerate(zip(corners, labels)):
                cv2.circle(vis_frame, tuple(corner), 10, (0, 0, 255), -1)
                cv2.putText(vis_frame, label, 
                           (corner[0]+15, corner[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add info text
            cv2.putText(vis_frame, f"Multi-Point Calibration ({len(self.points)} points)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite('auto_detection_preview.jpg', vis_frame)
            print(f"✅ Preview saved to: auto_detection_preview.jpg")
        
        return calibration
    
    def run(self):
        """Main execution"""
        print("\n" + "="*60)
        print("🎯 MULTI-POINT COURT CALIBRATION")
        print("="*60)
        
        # Manual multi-point selection
        corners = self.manual_selection(frame_number=100)
        
        if corners is None:
            print("\n❌ Calibration cancelled")
            return False
        
        print(f"\n✅ Rectangle fitted from {len(self.points)} points!")
        print(f"   Corners: {corners}")
        
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
    
    detector = MultiPointCourtSelector(video_path)
    success = detector.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
