import cv2
import numpy as np
import json
from pathlib import Path

class Interactive4PointCalibrator:
    """
    Interactive court calibrator with drag-and-drop adjustment
    1. Click 4 corners
    2. Drag any corner to fine-tune position
    3. Visual real-time preview
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Court dimensions
        self.court_width = 6.1
        self.court_length = 13.41
        
        # Points
        self.points = []
        self.current_frame = None
        self.display_frame = None
        
        # Drag state
        self.dragging = False
        self.drag_point_idx = None
        self.hover_point_idx = None
        
        # Visual settings
        self.point_radius = 10
        self.hover_radius = 15
        self.drag_threshold = 20  # pixels
        
        print("🏟️ Interactive 4-Point Court Calibrator")
        print(f"📹 Video: {Path(video_path).name}")
    
    def get_point_near_mouse(self, x, y):
        """Find if mouse is near any point"""
        for i, pt in enumerate(self.points):
            dist = np.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
            if dist < self.drag_threshold:
                return i
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events: click, drag, hover"""
        
        # HOVER detection
        if event == cv2.EVENT_MOUSEMOVE and not self.dragging:
            old_hover = self.hover_point_idx
            self.hover_point_idx = self.get_point_near_mouse(x, y)
            
            # Redraw only if hover changed
            if old_hover != self.hover_point_idx:
                self.update_display()
        
        # LEFT BUTTON DOWN
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near existing point (start drag)
            point_idx = self.get_point_near_mouse(x, y)
            
            if point_idx is not None:
                # Start dragging existing point
                self.dragging = True
                self.drag_point_idx = point_idx
                print(f"🖐️  Dragging point {point_idx + 1}...")
            
            elif len(self.points) < 4:
                # Add new point
                self.points.append([x, y])
                print(f"✅ Point {len(self.points)}: ({x}, {y})")
                self.update_display()
        
        # MOUSE MOVE (dragging)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Update point position
            if self.drag_point_idx is not None:
                self.points[self.drag_point_idx] = [x, y]
                self.update_display()
        
        # LEFT BUTTON UP (stop dragging)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            if self.drag_point_idx is not None:
                self.points[self.drag_point_idx] = [x, y]
                print(f"✅ Point {self.drag_point_idx + 1} adjusted to: ({x}, {y})")
            
            self.dragging = False
            self.drag_point_idx = None
            self.update_display()
    
    def update_display(self):
        """Update display with current state"""
        self.display_frame = self.current_frame.copy()
        
        # Draw court rectangle if we have 4 points
        if len(self.points) == 4:
            # Draw filled semi-transparent overlay
            overlay = self.display_frame.copy()
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0, 50))
            cv2.addWeighted(overlay, 0.2, self.display_frame, 0.8, 0, self.display_frame)
            
            # Draw rectangle outline
            cv2.polylines(self.display_frame, [pts], True, (0, 255, 255), 3)
            
            # Draw lines between points
            for i in range(4):
                pt1 = tuple(self.points[i])
                pt2 = tuple(self.points[(i + 1) % 4])
                cv2.line(self.display_frame, pt1, pt2, (0, 255, 0), 2)
        
        elif len(self.points) > 1:
            # Draw lines between consecutive points
            for i in range(len(self.points) - 1):
                pt1 = tuple(self.points[i])
                pt2 = tuple(self.points[i + 1])
                cv2.line(self.display_frame, pt1, pt2, (255, 255, 0), 2)
        
        # Draw all points
        labels = ['TL (Top-Left)', 'TR (Top-Right)', 'BR (Bottom-Right)', 'BL (Bottom-Left)']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, pt in enumerate(self.points):
            # Determine color and size
            if self.dragging and i == self.drag_point_idx:
                # Being dragged - large red circle
                color = (0, 0, 255)
                radius = self.hover_radius + 3
                thickness = -1
            elif self.hover_point_idx == i:
                # Hovered - larger circle with outline
                color = (255, 255, 255)
                radius = self.hover_radius
                thickness = 3
            else:
                # Normal point
                color = colors[i] if i < len(colors) else (255, 255, 255)
                radius = self.point_radius
                thickness = -1
            
            # Draw point
            cv2.circle(self.display_frame, tuple(pt), radius, color, thickness)
            
            # Draw inner circle for hollow points
            if self.hover_point_idx == i and not self.dragging:
                cv2.circle(self.display_frame, tuple(pt), radius - 3, colors[i], -1)
            
            # Draw point number/label
            label = labels[i] if i < len(labels) else f"P{i+1}"
            label_short = label.split()[0]  # Just "TL", "TR", etc.
            
            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness_text = 2
            (text_width, text_height), _ = cv2.getTextSize(label_short, font, font_scale, thickness_text)
            
            label_x = pt[0] + 20
            label_y = pt[1] - 10
            
            # Background rectangle
            cv2.rectangle(self.display_frame,
                         (label_x - 5, label_y - text_height - 5),
                         (label_x + text_width + 5, label_y + 5),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(self.display_frame, label_short,
                       (label_x, label_y),
                       font, font_scale, colors[i], thickness_text)
        
        # Status bar
        self.draw_status_bar()
        
        cv2.imshow('Interactive Court Calibration', self.display_frame)
    
    def draw_status_bar(self):
        """Draw status information"""
        # Top bar
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display_frame, 0.3, 0, self.display_frame)
        
        # Status text
        if len(self.points) < 4:
            status = f"Click point {len(self.points) + 1}/4: "
            labels = ['Top-Left corner', 'Top-Right corner', 'Bottom-Right corner', 'Bottom-Left corner']
            status += labels[len(self.points)]
            color = (0, 255, 255)
        elif self.dragging:
            status = f"Dragging point {self.drag_point_idx + 1} - Release to confirm"
            color = (0, 0, 255)
        else:
            status = "All points set! Drag any point to adjust or press ENTER to confirm"
            color = (0, 255, 0)
        
        cv2.putText(self.display_frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Controls help
        help_text = "Controls: Left Click = Add/Drag | R = Reset | U = Undo | ENTER = Confirm | Q = Quit"
        cv2.putText(self.display_frame, help_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def interactive_calibration(self, frame_number=100):
        """
        Interactive calibration with drag-and-drop
        """
        print("\n" + "="*70)
        print("🎯 INTERACTIVE 4-POINT CALIBRATION")
        print("="*70)
        print("")
        print("📍 Step 1: Click 4 corners in order:")
        print("   1️⃣  Top-Left corner")
        print("   2️⃣  Top-Right corner")
        print("   3️⃣  Bottom-Right corner")
        print("   4️⃣  Bottom-Left corner")
        print("")
        print("🖐️  Step 2: Fine-tune by dragging any point")
        print("   - Hover over a point to highlight it")
        print("   - Click and drag to adjust position")
        print("   - Release to confirm new position")
        print("")
        print("⌨️  Keyboard Controls:")
        print("   Left Click  - Add point (if < 4) or Drag point")
        print("   R           - Reset all points")
        print("   U           - Undo last point")
        print("   ENTER       - Accept and save")
        print("   Q           - Quit without saving")
        print("="*70)
        
        # Get frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            return None
        
        self.current_frame = frame.copy()
        self.display_frame = frame.copy()
        self.points = []
        self.dragging = False
        self.drag_point_idx = None
        self.hover_point_idx = None
        
        # Create window
        window_name = 'Interactive Court Calibration'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\n❌ Calibration cancelled")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r'):
                # Reset all points
                self.points = []
                self.dragging = False
                self.drag_point_idx = None
                self.hover_point_idx = None
                self.update_display()
                print("🔄 Reset - Click 4 corners again")
            
            elif key == ord('u'):
                # Undo last point
                if self.points and not self.dragging:
                    removed = self.points.pop()
                    self.update_display()
                    print(f"↩️  Undo - Removed point: {removed}")
            
            elif key == 13 or key == 10:  # ENTER
                if len(self.points) == 4 and not self.dragging:
                    # Show final confirmation
                    confirm_frame = self.display_frame.copy()
                    
                    # Big confirmation text
                    text = "Saving calibration..."
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2
                    thickness = 3
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    x = (confirm_frame.shape[1] - text_width) // 2
                    y = (confirm_frame.shape[0] + text_height) // 2
                    
                    # Background
                    cv2.rectangle(confirm_frame,
                                 (x - 20, y - text_height - 20),
                                 (x + text_width + 20, y + 20),
                                 (0, 255, 0), -1)
                    
                    cv2.putText(confirm_frame, text, (x, y),
                               font, font_scale, (0, 0, 0), thickness)
                    
                    cv2.imshow(window_name, confirm_frame)
                    cv2.waitKey(500)
                    
                    cv2.destroyAllWindows()
                    
                    # Convert to tuples
                    corners = [tuple(pt) for pt in self.points]
                    return corners
                else:
                    if len(self.points) < 4:
                        print(f"⚠️  Need 4 points (currently {len(self.points)})")
                    elif self.dragging:
                        print("⚠️  Release mouse button first")
        
        return None
    
    def save_calibration(self, corners, output_path='court_calibration_san4.json'):
        """Save calibration with preview"""
        # Define court coordinates
        court_points = np.array([
            [0, 0],  # TL
            [self.court_width, 0],  # TR
            [self.court_width, self.court_length],  # BR
            [0, self.court_length]  # BL
        ], dtype=np.float32)
        
        image_points = np.array(corners, dtype=np.float32)
        homography, _ = cv2.findHomography(image_points, court_points)
        
        calibration = {
            'homography': homography.tolist(),
            'court_width': self.court_width,
            'court_length': self.court_length,
            'image_points': image_points.tolist(),
            'court_points': court_points.tolist(),
            'method': 'interactive_4point_dragdrop',
            'video_path': str(self.video_path)
        }
        
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✅ Calibration saved to: {output_path}")
        
        # Create detailed preview
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = self.cap.read()
        
        if ret:
            vis_frame = frame.copy()
            
            # Draw court boundary
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 4)
            
            # Draw corners with labels
            labels = ['TL', 'TR', 'BR', 'BL']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            
            for i, (corner, label, color) in enumerate(zip(corners, labels, colors)):
                cv2.circle(vis_frame, corner, 12, color, -1)
                cv2.circle(vis_frame, corner, 12, (255, 255, 255), 2)
                
                cv2.putText(vis_frame, label,
                           (corner[0] + 20, corner[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw net line
            net_start_court = np.array([[0, self.court_length/2]], dtype=np.float32).reshape(-1, 1, 2)
            net_end_court = np.array([[self.court_width, self.court_length/2]], dtype=np.float32).reshape(-1, 1, 2)
            
            net_start_img = cv2.perspectiveTransform(net_start_court, np.linalg.inv(homography))[0][0]
            net_end_img = cv2.perspectiveTransform(net_end_court, np.linalg.inv(homography))[0][0]
            
            cv2.line(vis_frame,
                    tuple(net_start_img.astype(int)),
                    tuple(net_end_img.astype(int)),
                    (255, 255, 255), 4)
            
            # Add info text
            cv2.putText(vis_frame, "Court Calibration - Interactive 4-Point",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(vis_frame, f"Court: {self.court_width}m x {self.court_length}m",
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imwrite('auto_detection_preview.jpg', vis_frame)
            print(f"✅ Preview saved to: auto_detection_preview.jpg")
        
        return calibration
    
    def run(self):
        """Main execution"""
        print("\n" + "="*70)
        print("🎯 INTERACTIVE COURT CALIBRATION")
        print("="*70)
        
        # Interactive calibration
        corners = self.interactive_calibration(frame_number=100)
        
        if corners is None:
            print("\n❌ Calibration cancelled")
            return False
        
        print(f"\n✅ 4 corners confirmed!")
        for i, corner in enumerate(corners):
            labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
            print(f"   {labels[i]:15s}: {corner}")
        
        # Save calibration
        self.save_calibration(corners)
        
        print("\n" + "="*70)
        print("✅ CALIBRATION COMPLETE!")
        print("="*70)
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
    
    calibrator = Interactive4PointCalibrator(video_path)
    success = calibrator.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
