import cv2
import numpy as np
import json
from pathlib import Path

class NetLineSelector:
    """
    Tool to select net line by clicking 2 points
    Net line divides the court into 2 halves
    """
    def __init__(self, video_path, calibration_path='court_calibration_san4.json'):
        self.video_path = video_path
        self.calibration_path = calibration_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Load existing calibration
        with open(calibration_path, 'r') as f:
            self.calibration = json.load(f)
        
        # Net line points (2 points to define the line)
        self.net_points = []
        
        # Court polygon for visualization
        self.court_polygon = None
        if 'yellow_polygon' in self.calibration:
            self.court_polygon = np.array(self.calibration['yellow_polygon'], dtype=np.int32)
        
        # Drag state
        self.dragging_point = None
        self.hover_point = None
        
        self.current_frame = None
        self.display_frame = None
        
        print("🎾 Net Line Selector")
        print(f"📹 Video: {Path(video_path).name}")
        print(f"🏟️ Court polygon loaded: {len(self.court_polygon) if self.court_polygon is not None else 0} points")
    
    def get_point_near_mouse(self, x, y):
        """Find if mouse is near any point"""
        for i, point in enumerate(self.net_points):
            dist = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if dist < 15:
                return i
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            point_idx = self.get_point_near_mouse(x, y)
            
            if point_idx is not None:
                # Start dragging point
                self.dragging_point = point_idx
            elif len(self.net_points) < 2:
                # Add new point
                self.net_points.append([x, y])
                print(f"✅ Net point {len(self.net_points)}: ({x}, {y})")
                self.update_display()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point is not None:
                # Drag point
                self.net_points[self.dragging_point] = [x, y]
                self.update_display()
            else:
                # Hover detection
                old_hover = self.hover_point
                self.hover_point = self.get_point_near_mouse(x, y)
                if old_hover != self.hover_point:
                    self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_point is not None:
                self.net_points[self.dragging_point] = [x, y]
                print(f"📍 Net point {self.dragging_point + 1} moved to: ({x}, {y})")
                self.dragging_point = None
                self.update_display()
    
    def update_display(self):
        """Update display"""
        self.display_frame = self.current_frame.copy()
        
        # Draw court polygon (yellow) if available
        if self.court_polygon is not None:
            cv2.polylines(self.display_frame, [self.court_polygon], True, (0, 255, 255), 2)
        
        # Draw net line if we have 2 points
        if len(self.net_points) == 2:
            pt1 = tuple(self.net_points[0])
            pt2 = tuple(self.net_points[1])
            
            # Draw thick white line for net
            cv2.line(self.display_frame, pt1, pt2, (255, 255, 255), 6)
            
            # Draw dotted line extension to show full division
            self.draw_dotted_line(self.display_frame, pt1, pt2, (255, 0, 255), 2, 20)
        elif len(self.net_points) == 1:
            # Just show first point
            pt1 = tuple(self.net_points[0])
            cv2.circle(self.display_frame, pt1, 15, (255, 255, 255), 3)
        
        # Draw net points
        for i, point in enumerate(self.net_points):
            if self.dragging_point == i:
                # Being dragged
                cv2.circle(self.display_frame, tuple(point), 12, (0, 0, 255), -1)
                cv2.circle(self.display_frame, tuple(point), 12, (255, 255, 255), 3)
            elif self.hover_point == i:
                # Hovered
                cv2.circle(self.display_frame, tuple(point), 10, (255, 255, 255), 3)
                cv2.circle(self.display_frame, tuple(point), 7, (255, 255, 255), -1)
            else:
                # Normal
                cv2.circle(self.display_frame, tuple(point), 8, (255, 255, 255), -1)
                cv2.circle(self.display_frame, tuple(point), 8, (0, 0, 0), 2)
            
            # Point label
            label = "LEFT" if i == 0 else "RIGHT"
            cv2.putText(self.display_frame, label,
                       (point[0] + 12, point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status bar
        self.draw_status_bar()
        
        cv2.imshow('Net Line Selector', self.display_frame)
    
    def draw_dotted_line(self, img, pt1, pt2, color, thickness, gap):
        """Draw dotted line"""
        dist = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
        pts = []
        for i in np.arange(0, dist, gap):
            r = i / dist
            x = int((pt1[0]*(1-r) + pt2[0]*r) + 0.5)
            y = int((pt1[1]*(1-r) + pt2[1]*r) + 0.5)
            pts.append((x, y))
        
        for i in range(0, len(pts), 2):
            if i+1 < len(pts):
                cv2.line(img, pts[i], pts[i+1], color, thickness)
    
    def draw_status_bar(self):
        """Draw status bar"""
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_frame.shape[1], 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, self.display_frame, 0.25, 0, self.display_frame)
        
        if len(self.net_points) < 2:
            status = f"Click {2 - len(self.net_points)} more point(s) to define NET LINE"
            color = (0, 255, 255)
        else:
            status = "✅ Net line defined! Press ENTER to save"
            color = (0, 255, 0)
        
        controls = "Click = Add point | Drag = Move | U = Undo | R = Reset | ENTER = Save | Q = Quit"
        
        cv2.putText(self.display_frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(self.display_frame, controls, (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def interactive_selection(self, frame_number=100):
        """Interactive net line selection"""
        print("\n" + "="*70)
        print("🎾 NET LINE SELECTION")
        print("="*70)
        print("")
        print("📐 INSTRUCTIONS:")
        print("")
        print("1. Click 2 points to define the net line")
        print("   - LEFT point (left edge of net)")
        print("   - RIGHT point (right edge of net)")
        print("   - This line divides the court into 2 halves")
        print("")
        print("2. Drag points to adjust position")
        print("   - White line = net position")
        print("   - Yellow polygon = court boundary (from previous step)")
        print("")
        print("3. Press ENTER when satisfied to save")
        print("")
        print("⌨️  Controls:")
        print("  Click = Add point (max 2)")
        print("  Drag = Move point")
        print("  U = Undo last point")
        print("  R = Reset all")
        print("  ENTER = Save net line")
        print("  Q = Quit without saving")
        print("="*70)
        
        # Get frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            print("❌ Failed to read frame")
            return None
        
        self.current_frame = frame.copy()
        self.display_frame = frame.copy()
        
        # Reset state
        self.net_points = []
        self.dragging_point = None
        self.hover_point = None
        
        # Create window
        window_name = 'Net Line Selector'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n❌ Selection cancelled")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r') or key == ord('R'):
                # Reset
                self.net_points = []
                self.update_display()
                print("🔄 Reset all points")
            
            elif key == ord('u') or key == ord('U'):
                # Undo last point
                if self.net_points:
                    removed = self.net_points.pop()
                    self.update_display()
                    print(f"↩️  Undo point: {removed}")
            
            elif key == 13 or key == 10:  # ENTER
                if len(self.net_points) == 2:
                    cv2.destroyAllWindows()
                    return self.net_points
                else:
                    print(f"⚠️  Need 2 points (currently {len(self.net_points)})")
        
        return None
    
    def save_net_line(self, net_points):
        """Save net line to calibration"""
        # Update calibration with net line
        self.calibration['net_line'] = [list(map(int, p)) for p in net_points]
        self.calibration['net_line_defined'] = True
        
        with open(self.calibration_path, 'w') as f:
            json.dump(self.calibration, f, indent=2)
        
        print(f"\n✅ Net line saved to: {self.calibration_path}")
        print(f"   Left: {net_points[0]}")
        print(f"   Right: {net_points[1]}")
        
        # Preview
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = self.cap.read()
        
        if ret:
            vis_frame = frame.copy()
            
            # Draw court polygon (yellow)
            if self.court_polygon is not None:
                cv2.polylines(vis_frame, [self.court_polygon], True, (0, 255, 255), 3)
            
            # Draw net line (white thick line)
            pt1 = tuple(net_points[0])
            pt2 = tuple(net_points[1])
            cv2.line(vis_frame, pt1, pt2, (255, 255, 255), 8)
            
            # Draw net points
            cv2.circle(vis_frame, pt1, 12, (255, 255, 255), -1)
            cv2.circle(vis_frame, pt1, 12, (0, 0, 0), 2)
            cv2.putText(vis_frame, "LEFT", (pt1[0] + 15, pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.circle(vis_frame, pt2, 12, (255, 255, 255), -1)
            cv2.circle(vis_frame, pt2, 12, (0, 0, 0), 2)
            cv2.putText(vis_frame, "RIGHT", (pt2[0] + 15, pt2[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Info
            cv2.putText(vis_frame, "Court (Yellow) + Net Line (White)",
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite('net_line_preview.jpg', vis_frame)
            print(f"✅ Preview saved to: net_line_preview.jpg")
        
        return True
    
    def run(self):
        """Main execution"""
        net_points = self.interactive_selection(frame_number=100)
        
        if net_points is None:
            return False
        
        print(f"\n✅ Net line selected!")
        print(f"   2 points defined")
        
        success = self.save_net_line(net_points)
        
        if success:
            print("\n" + "="*70)
            print("✅ NET LINE SAVED!")
            print("="*70)
            print("\n💡 Now run tracking to see the net line")
        
        return success
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    
    selector = NetLineSelector(video_path)
    success = selector.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
