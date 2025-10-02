import cv2
import numpy as np
import json
from pathlib import Path
from scipy.interpolate import splprep, splev

class PerspectiveCourtCalibrator:
    """
    Perspective-aware court calibrator
    - Select multiple points per edge (2-5 points per side)
    - Automatic spline fitting for curved perspective edges
    - Drag-and-drop adjustment for fine-tuning
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Court dimensions
        self.court_width = 6.1
        self.court_length = 13.41
        
        # Points organized by edge
        self.edges = {
            'top': [],     # Top edge points
            'right': [],   # Right edge points
            'bottom': [],  # Bottom edge points
            'left': []     # Left edge points
        }
        self.current_edge = 'top'
        self.edge_order = ['top', 'right', 'bottom', 'left']
        self.edge_colors = {
            'top': (255, 0, 0),      # Blue
            'right': (0, 255, 0),    # Green
            'bottom': (0, 0, 255),   # Red
            'left': (255, 255, 0)    # Cyan
        }
        
        self.current_frame = None
        self.display_frame = None
        
        # Drag state
        self.dragging = False
        self.drag_edge = None
        self.drag_point_idx = None
        self.hover_edge = None
        self.hover_point_idx = None
        
        # Visual settings
        self.point_radius = 8
        self.hover_radius = 12
        self.drag_threshold = 15
        
        # Fitted corners
        self.fitted_corners = None
        
        print("🏟️ Perspective-Aware Court Calibrator")
        print(f"📹 Video: {Path(video_path).name}")
    
    def get_point_near_mouse(self, x, y):
        """Find if mouse is near any point"""
        for edge_name, points in self.edges.items():
            for i, pt in enumerate(points):
                dist = np.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
                if dist < self.drag_threshold:
                    return edge_name, i
        return None, None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        
        # HOVER detection
        if event == cv2.EVENT_MOUSEMOVE and not self.dragging:
            old_hover = (self.hover_edge, self.hover_point_idx)
            self.hover_edge, self.hover_point_idx = self.get_point_near_mouse(x, y)
            
            if old_hover != (self.hover_edge, self.hover_point_idx):
                self.update_display()
        
        # LEFT BUTTON DOWN
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near existing point
            edge, idx = self.get_point_near_mouse(x, y)
            
            if edge is not None:
                # Start dragging
                self.dragging = True
                self.drag_edge = edge
                self.drag_point_idx = idx
                print(f"🖐️  Dragging {edge} edge, point {idx + 1}...")
            else:
                # Add new point to current edge
                self.edges[self.current_edge].append([x, y])
                point_num = len(self.edges[self.current_edge])
                print(f"✅ {self.current_edge.upper()} edge: Point {point_num} at ({x}, {y})")
                self.update_display()
        
        # MOUSE MOVE (dragging)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.drag_edge is not None and self.drag_point_idx is not None:
                self.edges[self.drag_edge][self.drag_point_idx] = [x, y]
                self.update_display()
        
        # LEFT BUTTON UP
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            if self.drag_edge is not None and self.drag_point_idx is not None:
                self.edges[self.drag_edge][self.drag_point_idx] = [x, y]
                print(f"✅ {self.drag_edge.upper()} edge point {self.drag_point_idx + 1} adjusted to ({x}, {y})")
            
            self.dragging = False
            self.drag_edge = None
            self.drag_point_idx = None
            self.update_display()
    
    def fit_edge_curve(self, points):
        """Fit smooth curve through points"""
        if len(points) < 2:
            return None
        
        points = np.array(points)
        
        if len(points) == 2:
            # Just a line
            return points
        
        # Fit spline for smooth curve
        try:
            tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=min(3, len(points)-1))
            u_new = np.linspace(0, 1, 100)
            x_new, y_new = splev(u_new, tck)
            curve_points = np.column_stack([x_new, y_new])
            return curve_points
        except:
            return points
    
    def find_corner_intersections(self):
        """Find 4 corners from edge curves"""
        # Need at least 2 points per edge
        for edge_name, points in self.edges.items():
            if len(points) < 2:
                return None
        
        # Fit curves for each edge
        curves = {}
        for edge_name, points in self.edges.items():
            curve = self.fit_edge_curve(points)
            if curve is not None:
                curves[edge_name] = curve
        
        if len(curves) != 4:
            return None
        
        # Find intersection points
        def find_line_intersection(edge1_points, edge2_points):
            """Find intersection of two edge curves (use endpoints)"""
            # Use first and last points to define lines
            x1, y1 = edge1_points[0]
            x2, y2 = edge1_points[-1]
            x3, y3 = edge2_points[0]
            x4, y4 = edge2_points[-1]
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                # Parallel lines, use closest endpoints
                return tuple(edge1_points[0].astype(int))
            
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            return (int(x), int(y))
        
        # Find 4 corners
        try:
            top_left = find_line_intersection(curves['top'], curves['left'])
            top_right = find_line_intersection(curves['top'], curves['right'])
            bottom_right = find_line_intersection(curves['bottom'], curves['right'])
            bottom_left = find_line_intersection(curves['bottom'], curves['left'])
            
            return [top_left, top_right, bottom_right, bottom_left]
        except:
            return None
    
    def update_display(self):
        """Update display"""
        self.display_frame = self.current_frame.copy()
        
        # Draw edge curves
        for edge_name in self.edge_order:
            points = self.edges[edge_name]
            color = self.edge_colors[edge_name]
            
            if len(points) >= 2:
                # Fit and draw curve
                curve = self.fit_edge_curve(points)
                if curve is not None and len(curve) > 1:
                    curve_int = curve.astype(np.int32)
                    cv2.polylines(self.display_frame, [curve_int], False, color, 3)
            
            # Draw individual points
            for i, pt in enumerate(points):
                # Determine size and style
                if self.dragging and edge_name == self.drag_edge and i == self.drag_point_idx:
                    # Being dragged
                    cv2.circle(self.display_frame, tuple(pt), self.hover_radius + 3, (0, 0, 255), -1)
                    cv2.circle(self.display_frame, tuple(pt), self.hover_radius + 3, (255, 255, 255), 2)
                elif self.hover_edge == edge_name and self.hover_point_idx == i:
                    # Hovered
                    cv2.circle(self.display_frame, tuple(pt), self.hover_radius, (255, 255, 255), 3)
                    cv2.circle(self.display_frame, tuple(pt), self.hover_radius - 3, color, -1)
                else:
                    # Normal
                    cv2.circle(self.display_frame, tuple(pt), self.point_radius, color, -1)
                    cv2.circle(self.display_frame, tuple(pt), self.point_radius, (255, 255, 255), 1)
                
                # Point label
                label = f"{edge_name[0].upper()}{i+1}"
                cv2.putText(self.display_frame, label,
                           (pt[0] + 12, pt[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Try to find and draw corners
        self.fitted_corners = self.find_corner_intersections()
        
        if self.fitted_corners is not None:
            # Draw court polygon
            pts = np.array(self.fitted_corners, dtype=np.int32)
            
            # Semi-transparent overlay
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, self.display_frame, 0.85, 0, self.display_frame)
            
            # Court outline
            cv2.polylines(self.display_frame, [pts], True, (0, 255, 255), 3)
            
            # Corner markers
            corner_labels = ['TL', 'TR', 'BR', 'BL']
            for corner, label in zip(self.fitted_corners, corner_labels):
                cv2.circle(self.display_frame, corner, 10, (255, 255, 255), -1)
                cv2.circle(self.display_frame, corner, 10, (0, 0, 0), 2)
                cv2.putText(self.display_frame, label,
                           (corner[0] + 15, corner[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status bar
        self.draw_status_bar()
        
        cv2.imshow('Perspective Court Calibration', self.display_frame)
    
    def draw_status_bar(self):
        """Draw status information"""
        # Top bar
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_frame.shape[1], 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, self.display_frame, 0.25, 0, self.display_frame)
        
        # Current edge status
        edge_labels = {
            'top': '🔵 TOP edge',
            'right': '🟢 RIGHT edge',
            'bottom': '🔴 BOTTOM edge',
            'left': '🟡 LEFT edge'
        }
        
        status_lines = []
        
        if self.dragging:
            status_lines.append(f"🖐️  DRAGGING: {self.drag_edge.upper()} edge point {self.drag_point_idx + 1}")
        else:
            current_count = len(self.edges[self.current_edge])
            status_lines.append(f"Clicking: {edge_labels[self.current_edge]} (currently {current_count} points)")
        
        # Edge counts
        counts = " | ".join([f"{name.upper()}: {len(points)}" 
                            for name, points in self.edges.items()])
        status_lines.append(f"Points per edge: {counts}")
        
        # Instructions
        min_per_edge = min(len(points) for points in self.edges.values())
        if min_per_edge < 2:
            status_lines.append("⚠️  Need at least 2 points per edge")
        elif self.fitted_corners is not None:
            status_lines.append("✅ Court detected! Drag points to adjust or press ENTER to save")
        else:
            status_lines.append("Keep adding points...")
        
        # Draw status text
        y_offset = 25
        for i, line in enumerate(status_lines):
            color = (0, 255, 0) if "✅" in line else (0, 255, 255) if "⚠️" in line else (255, 255, 255)
            cv2.putText(self.display_frame, line, (10, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Controls
        controls = "N=Next edge | B=Back | R=Reset | U=Undo | D=Delete point | ENTER=Save | Q=Quit"
        cv2.putText(self.display_frame, controls, (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    def interactive_calibration(self, frame_number=100):
        """Interactive perspective-aware calibration"""
        print("\n" + "="*70)
        print("🎯 PERSPECTIVE-AWARE COURT CALIBRATION")
        print("="*70)
        print("")
        print("📐 This tool handles perspective distortion!")
        print("")
        print("For each edge, click 2-5 points along the court line:")
        print("  🔵 TOP edge    - Click left to right")
        print("  🟢 RIGHT edge  - Click top to bottom")
        print("  🔴 BOTTOM edge - Click right to left")
        print("  🟡 LEFT edge   - Click bottom to top")
        print("")
        print("💡 More points = better curve fitting for perspective")
        print("")
        print("⌨️  Controls:")
        print("  Left Click      - Add point to current edge")
        print("  Click & Drag    - Adjust existing point")
        print("  N               - Next edge")
        print("  B               - Back to previous edge")
        print("  U               - Undo last point on current edge")
        print("  D + Click       - Delete specific point")
        print("  R               - Reset all points")
        print("  ENTER           - Save calibration")
        print("  Q               - Quit")
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
        for edge in self.edges:
            self.edges[edge] = []
        self.current_edge = 'top'
        self.dragging = False
        self.fitted_corners = None
        
        # Create window
        window_name = 'Perspective Court Calibration'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.update_display()
        
        delete_mode = False
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\n❌ Calibration cancelled")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('n'):
                # Next edge
                if not self.dragging:
                    idx = self.edge_order.index(self.current_edge)
                    self.current_edge = self.edge_order[(idx + 1) % 4]
                    print(f"➡️  Switched to {self.current_edge.upper()} edge")
                    self.update_display()
            
            elif key == ord('b'):
                # Previous edge
                if not self.dragging:
                    idx = self.edge_order.index(self.current_edge)
                    self.current_edge = self.edge_order[(idx - 1) % 4]
                    print(f"⬅️  Switched to {self.current_edge.upper()} edge")
                    self.update_display()
            
            elif key == ord('r'):
                # Reset all
                if not self.dragging:
                    for edge in self.edges:
                        self.edges[edge] = []
                    self.current_edge = 'top'
                    self.fitted_corners = None
                    self.update_display()
                    print("🔄 Reset all points")
            
            elif key == ord('u'):
                # Undo last point on current edge
                if not self.dragging and self.edges[self.current_edge]:
                    removed = self.edges[self.current_edge].pop()
                    self.update_display()
                    print(f"↩️  Undo - Removed point from {self.current_edge.upper()}: {removed}")
            
            elif key == ord('d'):
                # Toggle delete mode
                delete_mode = not delete_mode
                print(f"🗑️  Delete mode: {'ON (click point to delete)' if delete_mode else 'OFF'}")
            
            elif key == 13 or key == 10:  # ENTER
                # Check if we have enough points
                min_points = min(len(points) for points in self.edges.values())
                
                if min_points < 2:
                    print(f"⚠️  Need at least 2 points per edge (minimum currently: {min_points})")
                    continue
                
                if self.fitted_corners is None:
                    print("⚠️  Could not fit court corners. Add more points.")
                    continue
                
                if not self.dragging:
                    # Show confirmation
                    confirm_frame = self.display_frame.copy()
                    text = "Saving calibration..."
                    cv2.putText(confirm_frame, text,
                               (confirm_frame.shape[1]//2 - 200, confirm_frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow(window_name, confirm_frame)
                    cv2.waitKey(500)
                    
                    cv2.destroyAllWindows()
                    return self.fitted_corners
        
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
        
        # Save edge points too
        edge_points_data = {}
        for edge_name, points in self.edges.items():
            edge_points_data[edge_name] = [pt for pt in points]
        
        calibration = {
            'homography': homography.tolist(),
            'court_width': self.court_width,
            'court_length': self.court_length,
            'image_points': image_points.tolist(),
            'court_points': court_points.tolist(),
            'edge_points': edge_points_data,
            'method': 'perspective_aware_multipoint',
            'video_path': str(self.video_path)
        }
        
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✅ Calibration saved to: {output_path}")
        
        # Create preview
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = self.cap.read()
        
        if ret:
            vis_frame = frame.copy()
            
            # Draw edge curves
            for edge_name in self.edge_order:
                points = self.edges[edge_name]
                color = self.edge_colors[edge_name]
                
                if len(points) >= 2:
                    curve = self.fit_edge_curve(points)
                    if curve is not None:
                        curve_int = curve.astype(np.int32)
                        cv2.polylines(vis_frame, [curve_int], False, color, 2)
                
                # Draw points
                for pt in points:
                    cv2.circle(vis_frame, tuple(pt), 4, color, -1)
            
            # Draw court boundary
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis_frame, [pts], True, (0, 255, 255), 4)
            
            # Draw corners
            corner_labels = ['TL', 'TR', 'BR', 'BL']
            for corner, label in zip(corners, corner_labels):
                cv2.circle(vis_frame, corner, 10, (255, 255, 255), -1)
                cv2.circle(vis_frame, corner, 10, (0, 0, 0), 2)
                cv2.putText(vis_frame, label, (corner[0] + 15, corner[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Info
            total_points = sum(len(points) for points in self.edges.values())
            cv2.putText(vis_frame, f"Perspective Calibration ({total_points} points)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite('auto_detection_preview.jpg', vis_frame)
            print(f"✅ Preview saved to: auto_detection_preview.jpg")
        
        return calibration
    
    def run(self):
        """Main execution"""
        print("\n" + "="*70)
        print("🎯 PERSPECTIVE COURT CALIBRATION")
        print("="*70)
        
        corners = self.interactive_calibration(frame_number=100)
        
        if corners is None:
            print("\n❌ Calibration cancelled")
            return False
        
        print(f"\n✅ Court corners fitted!")
        total_points = sum(len(points) for points in self.edges.values())
        print(f"   Used {total_points} points across 4 edges")
        for edge_name, points in self.edges.items():
            print(f"   {edge_name.upper():6s}: {len(points)} points")
        
        # Save
        self.save_calibration(corners)
        
        print("\n" + "="*70)
        print("✅ CALIBRATION COMPLETE!")
        print("="*70)
        print("\n💡 Run tracking:")
        print("   python enhanced_tracking_san4.py")
        print("   python stable_reid_tracking_san4.py")
        
        return True
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    
    calibrator = PerspectiveCourtCalibrator(video_path)
    success = calibrator.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
