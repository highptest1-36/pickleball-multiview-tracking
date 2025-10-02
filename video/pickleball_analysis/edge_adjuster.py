import cv2
import numpy as np
import json
from pathlib import Path

class EdgeAdjustableCalibrator:
    """
    Court calibrator with edge adjustment
    1. Click 4 corners to form rectangle
    2. Adjust each edge (top/right/bottom/left) to match actual white lines
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Court dimensions
        self.court_width = 6.1
        self.court_length = 13.41
        
        # 4 corners
        self.corners = []
        
        # Edge adjustment mode
        self.adjustment_mode = False
        self.selected_edge = None  # 'top', 'right', 'bottom', 'left'
        self.edge_offset = {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
        
        self.current_frame = None
        self.display_frame = None
        
        # Drag state for corner adjustment
        self.dragging_corner = None
        self.hover_corner = None
        
        print("🏟️ Edge-Adjustable Court Calibrator")
        print(f"📹 Video: {Path(video_path).name}")
    
    def get_corner_near_mouse(self, x, y):
        """Find if mouse is near any corner"""
        for i, corner in enumerate(self.corners):
            dist = np.sqrt((corner[0] - x)**2 + (corner[1] - y)**2)
            if dist < 20:
                return i
        return None
    
    def get_adjusted_corners(self):
        """Get corners adjusted by edge offsets"""
        if len(self.corners) != 4:
            return self.corners
        
        # Original corners: TL, TR, BR, BL
        corners = [list(c) for c in self.corners]
        
        # Apply edge offsets (perpendicular to each edge)
        # Top edge: move both TL and TR up/down
        if self.edge_offset['top'] != 0:
            corners[0][1] += self.edge_offset['top']  # TL
            corners[1][1] += self.edge_offset['top']  # TR
        
        # Right edge: move both TR and BR left/right
        if self.edge_offset['right'] != 0:
            corners[1][0] += self.edge_offset['right']  # TR
            corners[2][0] += self.edge_offset['right']  # BR
        
        # Bottom edge: move both BR and BL up/down
        if self.edge_offset['bottom'] != 0:
            corners[2][1] += self.edge_offset['bottom']  # BR
            corners[3][1] += self.edge_offset['bottom']  # BL
        
        # Left edge: move both BL and TL left/right
        if self.edge_offset['left'] != 0:
            corners[3][0] += self.edge_offset['left']  # BL
            corners[0][0] += self.edge_offset['left']  # TL
        
        return [tuple(c) for c in corners]
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        
        if not self.adjustment_mode:
            # PHASE 1: Corner selection
            if event == cv2.EVENT_LBUTTONDOWN:
                corner_idx = self.get_corner_near_mouse(x, y)
                
                if corner_idx is not None:
                    # Start dragging corner
                    self.dragging_corner = corner_idx
                elif len(self.corners) < 4:
                    # Add new corner
                    self.corners.append([x, y])
                    print(f"✅ Corner {len(self.corners)}: ({x}, {y})")
                    
                    if len(self.corners) == 4:
                        print("\n✅ 4 corners selected!")
                        print("   Press 'A' to enter adjustment mode")
                    
                    self.update_display()
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.dragging_corner is not None:
                    # Drag corner
                    self.corners[self.dragging_corner] = [x, y]
                    self.update_display()
                else:
                    # Hover detection
                    old_hover = self.hover_corner
                    self.hover_corner = self.get_corner_near_mouse(x, y)
                    if old_hover != self.hover_corner:
                        self.update_display()
            
            elif event == cv2.EVENT_LBUTTONUP:
                if self.dragging_corner is not None:
                    self.corners[self.dragging_corner] = [x, y]
                    print(f"✅ Corner {self.dragging_corner + 1} adjusted to: ({x}, {y})")
                    self.dragging_corner = None
                    self.update_display()
    
    def update_display(self):
        """Update display"""
        self.display_frame = self.current_frame.copy()
        
        if len(self.corners) > 0:
            adjusted_corners = self.get_adjusted_corners()
            
            # Draw rectangle if we have 4 corners
            if len(adjusted_corners) == 4:
                # Semi-transparent overlay
                overlay = self.display_frame.copy()
                pts = np.array(adjusted_corners, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.15, self.display_frame, 0.85, 0, self.display_frame)
                
                # Draw edges with different colors based on selection
                edges = [
                    ('top', 0, 1, (255, 0, 0)),      # TL -> TR (Blue)
                    ('right', 1, 2, (0, 255, 0)),    # TR -> BR (Green)
                    ('bottom', 2, 3, (0, 0, 255)),   # BR -> BL (Red)
                    ('left', 3, 0, (255, 255, 0))    # BL -> TL (Cyan)
                ]
                
                for edge_name, i1, i2, default_color in edges:
                    pt1 = tuple(adjusted_corners[i1])
                    pt2 = tuple(adjusted_corners[i2])
                    
                    # Highlight selected edge
                    if self.adjustment_mode and self.selected_edge == edge_name:
                        color = (255, 255, 255)  # White for selected
                        thickness = 5
                    else:
                        color = default_color
                        thickness = 3
                    
                    cv2.line(self.display_frame, pt1, pt2, color, thickness)
                    
                    # Draw edge label
                    mid_x = (pt1[0] + pt2[0]) // 2
                    mid_y = (pt1[1] + pt2[1]) // 2
                    
                    if self.adjustment_mode and self.selected_edge == edge_name:
                        label = f"{edge_name.upper()} [{self.edge_offset[edge_name]:+d}px]"
                        label_color = (255, 255, 255)
                    else:
                        label = edge_name[0].upper()
                        label_color = color
                    
                    # Background for label
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(self.display_frame,
                                 (mid_x - tw//2 - 5, mid_y - th - 5),
                                 (mid_x + tw//2 + 5, mid_y + 5),
                                 (0, 0, 0), -1)
                    
                    cv2.putText(self.display_frame, label,
                               (mid_x - tw//2, mid_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
            
            # Draw corners
            corner_labels = ['TL', 'TR', 'BR', 'BL']
            corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            
            for i, (corner, label, color) in enumerate(zip(adjusted_corners, corner_labels, corner_colors)):
                if self.dragging_corner == i:
                    # Being dragged
                    radius = 15
                    cv2.circle(self.display_frame, tuple(corner), radius, (0, 0, 255), -1)
                    cv2.circle(self.display_frame, tuple(corner), radius, (255, 255, 255), 3)
                elif self.hover_corner == i:
                    # Hovered
                    radius = 12
                    cv2.circle(self.display_frame, tuple(corner), radius, (255, 255, 255), 3)
                    cv2.circle(self.display_frame, tuple(corner), radius - 3, color, -1)
                else:
                    # Normal
                    radius = 10
                    cv2.circle(self.display_frame, tuple(corner), radius, color, -1)
                    cv2.circle(self.display_frame, tuple(corner), radius, (255, 255, 255), 2)
                
                # Label
                cv2.putText(self.display_frame, label,
                           (corner[0] + 15, corner[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Status bar
        self.draw_status_bar()
        
        cv2.imshow('Edge-Adjustable Calibration', self.display_frame)
    
    def draw_status_bar(self):
        """Draw status bar"""
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_frame.shape[1], 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, self.display_frame, 0.25, 0, self.display_frame)
        
        if not self.adjustment_mode:
            # Phase 1: Corner selection
            if len(self.corners) < 4:
                status = f"STEP 1: Click {4 - len(self.corners)} more corner(s)"
                color = (0, 255, 255)
            else:
                status = "STEP 1 Complete! Press 'A' to adjust edges"
                color = (0, 255, 0)
            
            controls = "Left Click = Add corner | Drag = Move corner | U = Undo | R = Reset | A = Adjust mode"
        else:
            # Phase 2: Edge adjustment
            if self.selected_edge is None:
                status = "STEP 2: Select edge to adjust (1=Top, 2=Right, 3=Bottom, 4=Left)"
                color = (0, 255, 255)
            else:
                status = f"Adjusting {self.selected_edge.upper()} edge: Use WASD or Arrow Keys (offset: {self.edge_offset[self.selected_edge]:+d}px)"
                color = (255, 255, 255)
            
            controls = "1-4 = Select edge | Arrow Keys or WASD = Adjust | ENTER = Save | ESC = Back to corners"
        
        cv2.putText(self.display_frame, status, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.putText(self.display_frame, controls, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def interactive_calibration(self, frame_number=100):
        """Interactive calibration"""
        print("\n" + "="*70)
        print("🎯 EDGE-ADJUSTABLE COURT CALIBRATION")
        print("="*70)
        print("")
        print("📐 TWO-STEP PROCESS:")
        print("")
        print("STEP 1: Set 4 Corners")
        print("  Click 4 corners in order:")
        print("    1. Top-Left")
        print("    2. Top-Right")
        print("    3. Bottom-Right")
        print("    4. Bottom-Left")
        print("  You can drag corners to adjust")
        print("")
        print("STEP 2: Adjust Edges (Press 'A' after step 1)")
        print("  Select edge by number:")
        print("    1 = TOP edge    (Blue)")
        print("    2 = RIGHT edge  (Green)")
        print("    3 = BOTTOM edge (Red)")
        print("    4 = LEFT edge   (Cyan)")
        print("  Use arrow keys to move selected edge:")
        print("    ↑/↓ = Move horizontal edges up/down")
        print("    ←/→ = Move vertical edges left/right")
        print("")
        print("⌨️  Controls:")
        print("  STEP 1: Left Click, Drag, U=Undo, R=Reset, A=Adjust mode")
        print("  STEP 2: 1-4=Select edge, Arrows=Adjust, ENTER=Save, ESC=Back")
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
        self.corners = []
        self.adjustment_mode = False
        self.selected_edge = None
        self.edge_offset = {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
        self.dragging_corner = None
        self.hover_corner = None
        
        # Create window
        window_name = 'Edge-Adjustable Calibration'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n❌ Calibration cancelled")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r'):
                # Reset
                self.corners = []
                self.adjustment_mode = False
                self.selected_edge = None
                self.edge_offset = {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
                self.update_display()
                print("🔄 Reset all")
            
            elif key == ord('u'):
                # Undo last corner
                if not self.adjustment_mode and self.corners:
                    removed = self.corners.pop()
                    self.update_display()
                    print(f"↩️  Undo corner: {removed}")
            
            elif key == ord('a'):
                # Enter adjustment mode
                if len(self.corners) == 4 and not self.adjustment_mode:
                    self.adjustment_mode = True
                    self.selected_edge = None
                    self.update_display()
                    print("\n📐 Adjustment mode activated!")
                    print("   Press 1-4 to select edge, then use arrow keys")
            
            elif key == 27:  # ESC
                # Exit adjustment mode
                if self.adjustment_mode:
                    self.adjustment_mode = False
                    self.selected_edge = None
                    self.update_display()
                    print("↩️  Back to corner mode")
            
            # Edge selection (1-4)
            elif self.adjustment_mode:
                if key == ord('1'):
                    self.selected_edge = 'top'
                    self.update_display()
                    print("Selected: TOP edge (use ↑↓ arrows)")
                
                elif key == ord('2'):
                    self.selected_edge = 'right'
                    self.update_display()
                    print("Selected: RIGHT edge (use ←→ arrows)")
                
                elif key == ord('3'):
                    self.selected_edge = 'bottom'
                    self.update_display()
                    print("Selected: BOTTOM edge (use ↑↓ arrows)")
                
                elif key == ord('4'):
                    self.selected_edge = 'left'
                    self.update_display()
                    print("Selected: LEFT edge (use ←→ arrows)")
                
                # Arrow keys for adjustment (also check WASD as backup)
                elif self.selected_edge is not None:
                    step = 2  # Increased step for visibility
                    adjusted = False
                    
                    # Up arrow or W
                    if key == 82 or key == ord('w') or key == ord('W'):
                        if self.selected_edge in ['top', 'bottom']:
                            self.edge_offset[self.selected_edge] -= step
                            adjusted = True
                    
                    # Down arrow or S
                    elif key == 84 or key == ord('s') or key == ord('S'):
                        if self.selected_edge in ['top', 'bottom']:
                            self.edge_offset[self.selected_edge] += step
                            adjusted = True
                    
                    # Left arrow or A
                    elif key == 81 or key == ord('a') or key == ord('A'):
                        if self.selected_edge in ['left', 'right']:
                            self.edge_offset[self.selected_edge] -= step
                            adjusted = True
                    
                    # Right arrow or D
                    elif key == 83 or key == ord('d') or key == ord('D'):
                        if self.selected_edge in ['left', 'right']:
                            self.edge_offset[self.selected_edge] += step
                            adjusted = True
                    
                    if adjusted:
                        self.update_display()
                        print(f"  {self.selected_edge.upper()}: {self.edge_offset[self.selected_edge]:+d}px")
            
            elif key == 13 or key == 10:  # ENTER
                if len(self.corners) == 4:
                    # Save
                    cv2.destroyAllWindows()
                    final_corners = self.get_adjusted_corners()
                    return final_corners
                else:
                    print(f"⚠️  Need 4 corners (currently {len(self.corners)})")
        
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
            'edge_offsets': self.edge_offset,
            'method': 'edge_adjustable',
            'video_path': str(self.video_path)
        }
        
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✅ Calibration saved to: {output_path}")
        
        # Preview
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = self.cap.read()
        
        if ret:
            vis_frame = frame.copy()
            
            # Draw court
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 4)
            
            # Draw corners
            corner_labels = ['TL', 'TR', 'BR', 'BL']
            for corner, label in zip(corners, corner_labels):
                cv2.circle(vis_frame, corner, 12, (0, 0, 255), -1)
                cv2.circle(vis_frame, corner, 12, (255, 255, 255), 2)
                cv2.putText(vis_frame, label, (corner[0] + 15, corner[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Info
            cv2.putText(vis_frame, "Edge-Adjustable Calibration",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            offset_text = f"Offsets: T{self.edge_offset['top']:+d} R{self.edge_offset['right']:+d} B{self.edge_offset['bottom']:+d} L{self.edge_offset['left']:+d}"
            cv2.putText(vis_frame, offset_text,
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imwrite('auto_detection_preview.jpg', vis_frame)
            print(f"✅ Preview saved to: auto_detection_preview.jpg")
        
        return calibration
    
    def run(self):
        """Main execution"""
        corners = self.interactive_calibration(frame_number=100)
        
        if corners is None:
            return False
        
        print(f"\n✅ Calibration complete!")
        print(f"   Final corners: {corners}")
        print(f"   Edge offsets: {self.edge_offset}")
        
        self.save_calibration(corners)
        
        print("\n" + "="*70)
        print("✅ READY TO TRACK!")
        print("="*70)
        print("\n💡 Run: python enhanced_tracking_san4.py")
        
        return True
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    
    calibrator = EdgeAdjustableCalibrator(video_path)
    success = calibrator.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
