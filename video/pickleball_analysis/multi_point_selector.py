import cv2
import numpy as np
import json
from pathlib import Path

class MultiPointCalibrator:
    """
    Multi-point court calibrator:
    - Click nhiều điểm dọc theo viền sân
    - Các điểm tự động nối lại thành polygon
    - Nhấn ENTER để fit rectangle và save
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Court dimensions
        self.court_width = 6.1
        self.court_length = 13.41
        
        # Points clicked by user
        self.points = []
        
        # Drag state
        self.dragging_point = None
        self.hover_point = None
        
        self.current_frame = None
        self.display_frame = None
        
        print("🏟️ Multi-Point Court Calibrator")
        print(f"📹 Video: {Path(video_path).name}")
    
    def get_point_near_mouse(self, x, y):
        """Find if mouse is near any point"""
        for i, point in enumerate(self.points):
            dist = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if dist < 15:
                return i
        return None
    
    def fit_rectangle_from_points(self):
        """Fit minimum area rectangle from all points"""
        if len(self.points) < 4:
            return None
        
        # Use OpenCV's minAreaRect to find best-fit rectangle
        points_array = np.array(self.points, dtype=np.float32)
        rect = cv2.minAreaRect(points_array)
        
        # Get 4 corners
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Sort corners: TL, TR, BR, BL
        # Sort by y first (top 2, bottom 2)
        sorted_by_y = sorted(box, key=lambda p: p[1])
        top_2 = sorted_by_y[:2]
        bottom_2 = sorted_by_y[2:]
        
        # Sort top 2 by x (left, right)
        top_sorted = sorted(top_2, key=lambda p: p[0])
        tl, tr = top_sorted
        
        # Sort bottom 2 by x (left, right)
        bottom_sorted = sorted(bottom_2, key=lambda p: p[0])
        bl, br = bottom_sorted
        
        corners = [tl, tr, br, bl]
        
        return corners
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            point_idx = self.get_point_near_mouse(x, y)
            
            if point_idx is not None:
                # Start dragging point
                self.dragging_point = point_idx
            else:
                # Add new point
                self.points.append([x, y])
                print(f"✅ Điểm {len(self.points)}: ({x}, {y})")
                self.update_display()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point is not None:
                # Drag point
                self.points[self.dragging_point] = [x, y]
                self.update_display()
            else:
                # Hover detection
                old_hover = self.hover_point
                self.hover_point = self.get_point_near_mouse(x, y)
                if old_hover != self.hover_point:
                    self.update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_point is not None:
                self.points[self.dragging_point] = [x, y]
                print(f"📍 Điểm {self.dragging_point + 1} di chuyển đến: ({x}, {y})")
                self.dragging_point = None
                self.update_display()
    
    def update_display(self):
        """Update display"""
        self.display_frame = self.current_frame.copy()
        
        if len(self.points) > 0:
            # Draw polygon connecting all points (YELLOW ONLY - NO GREEN!)
            if len(self.points) >= 2:
                pts = np.array(self.points, dtype=np.int32)
                
                # Draw filled polygon with transparency
                overlay = self.display_frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 255))
                cv2.addWeighted(overlay, 0.2, self.display_frame, 0.8, 0, self.display_frame)
                
                # Draw polygon edges (yellow)
                cv2.polylines(self.display_frame, [pts], True, (0, 255, 255), 3)
            
            # Draw all clicked points
            for i, point in enumerate(self.points):
                if self.dragging_point == i:
                    # Being dragged
                    cv2.circle(self.display_frame, tuple(point), 12, (0, 0, 255), -1)
                    cv2.circle(self.display_frame, tuple(point), 12, (255, 255, 255), 3)
                elif self.hover_point == i:
                    # Hovered
                    cv2.circle(self.display_frame, tuple(point), 10, (255, 255, 255), 3)
                    cv2.circle(self.display_frame, tuple(point), 7, (0, 255, 255), -1)
                else:
                    # Normal
                    cv2.circle(self.display_frame, tuple(point), 8, (0, 255, 255), -1)
                    cv2.circle(self.display_frame, tuple(point), 8, (255, 255, 255), 2)
                
                # Point number
                cv2.putText(self.display_frame, str(i + 1),
                           (point[0] + 12, point[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Status bar
        self.draw_status_bar()
        
        cv2.imshow('Multi-Point Calibrator', self.display_frame)
    
    def draw_status_bar(self):
        """Draw status bar"""
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_frame.shape[1], 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, self.display_frame, 0.25, 0, self.display_frame)
        
        if len(self.points) < 4:
            status = f"Click điểm dọc viền sân (tối thiểu 4, hiện tại {len(self.points)})"
            color = (0, 255, 255)
        else:
            status = f"✅ {len(self.points)} điểm - Kéo thả để điều chỉnh"
            color = (0, 255, 0)
        
        controls = "Click = Thêm điểm | Kéo thả = Di chuyển | U = Xóa cuối | R = Reset | ENTER = Lưu | Q = Thoát"
        
        cv2.putText(self.display_frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(self.display_frame, controls, (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def interactive_calibration(self, frame_number=100):
        """Interactive calibration"""
        print("\n" + "="*70)
        print("🎯 MULTI-POINT COURT CALIBRATION")
        print("="*70)
        print("")
        print("📐 CÁCH SỬ DỤNG:")
        print("")
        print("1. Click nhiều điểm dọc theo viền trắng của sân")
        print("   - Tối thiểu 4 điểm (nhiều điểm hơn = chính xác hơn)")
        print("   - Đi theo viền sân theo thứ tự")
        print("   - Các điểm sẽ tự động nối lại thành polygon VÀNG")
        print("")
        print("2. Kéo thả điểm để điều chỉnh vị trí")
        print("   - CHỈ hiển thị màu VÀNG duy nhất")
        print("   - Không có màu xanh")
        print("")
        print("3. Nhấn ENTER khi đã hài lòng để lưu")
        print("")
        print("⌨️  Phím tắt:")
        print("  Click Trái = Thêm điểm mới")
        print("  Kéo thả = Di chuyển điểm")
        print("  U = Xóa điểm cuối cùng")
        print("  R = Reset tất cả")
        print("  ENTER = Lưu calibration")
        print("  Q = Thoát không lưu")
        print("="*70)
        
        # Get frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            print("❌ Không đọc được frame")
            return None
        
        self.current_frame = frame.copy()
        self.display_frame = frame.copy()
        
        # Reset state
        self.points = []
        self.dragging_point = None
        self.hover_point = None
        
        # Create window
        window_name = 'Multi-Point Calibrator'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n❌ Hủy calibration")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r') or key == ord('R'):
                # Reset
                self.points = []
                self.update_display()
                print("🔄 Reset tất cả điểm")
            
            elif key == ord('u') or key == ord('U'):
                # Undo last point
                if self.points:
                    removed = self.points.pop()
                    self.update_display()
                    print(f"↩️  Xóa điểm: {removed}")
            
            elif key == 13 or key == 10:  # ENTER
                if len(self.points) >= 4:
                    # Save
                    cv2.destroyAllWindows()
                    corners = self.fit_rectangle_from_points()
                    return corners
                else:
                    print(f"⚠️  Cần tối thiểu 4 điểm (hiện tại {len(self.points)})")
        
        return None
    
    def save_calibration(self, corners, output_path='court_calibration_san4.json'):
        """Save calibration using actual user points (yellow polygon)"""
        # Use EXACTLY the 8 yellow points user clicked - NO CHANGES!
        # Map them to court rectangle 4 corners for standard homography
        
        # Court dimensions (standard pickleball court)
        # Full court: width=6.1m x length=13.41m
        # Split by WIDTH into 2 halves (each half: 3.05m x 13.41m)
        court_points = np.array([
            [0, 0],                              # TL
            [self.court_width, 0],               # TR  
            [self.court_width, self.court_length],  # BR
            [0, self.court_length]               # BL
        ], dtype=np.float32)
        
        # Get the fitted 4 corners from 8 yellow points for homography
        fitted_corners = np.array(corners, dtype=np.float32)
        
        # Compute homography using fitted corners
        homography, _ = cv2.findHomography(fitted_corners, court_points)
        
        # But SAVE the actual 8 yellow points user clicked
        yellow_points = [list(map(int, p)) for p in self.points]
        
        calibration = {
            'homography': homography.tolist(),
            'court_width': self.court_width,
            'court_length': self.court_length,
            'image_points': fitted_corners.tolist(),  # 4 corners for homography transform
            'court_points': court_points.tolist(),
            'yellow_polygon': yellow_points,  # EXACT 8 yellow points user clicked (unchanged!)
            'num_yellow_points': len(self.points),
            'court_split': 'by_width',  # Split court by WIDTH (3.05m each half)
            'method': 'yellow_polygon_exact',
            'video_path': str(self.video_path)
        }
        
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✅ Calibration đã lưu: {output_path}")
        print(f"   ✨ LƯU ĐÚNG {len(self.points)} ĐIỂM VÀNG (không thay đổi!)")
        print(f"   🎯 Chia sân theo CHIỀU RỘNG (width) thành 2 nửa")
        print(f"   📐 Mỗi nửa: {self.court_width/2:.2f}m x {self.court_length:.2f}m")
        
        # Preview
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = self.cap.read()
        
        if ret:
            vis_frame = frame.copy()
            
            # Draw user polygon (YELLOW ONLY - NO GREEN!)
            pts = np.array(self.points, dtype=np.int32)
            overlay = vis_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.25, vis_frame, 0.75, 0, vis_frame)
            cv2.polylines(vis_frame, [pts], True, (0, 255, 255), 4)
            
            # Draw points with numbers
            for i, point in enumerate(self.points):
                cv2.circle(vis_frame, tuple(point), 8, (0, 255, 255), -1)
                cv2.circle(vis_frame, tuple(point), 8, (255, 255, 255), 2)
                cv2.putText(vis_frame, str(i + 1), (point[0] + 12, point[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Info (NO GREEN RECTANGLE MENTIONED!)
            cv2.putText(vis_frame, f"YELLOW POLYGON: {len(self.points)} points (saved exactly as-is)",
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
            
            cv2.imwrite('auto_detection_preview.jpg', vis_frame)
            print(f"✅ Preview đã lưu: auto_detection_preview.jpg")
        
        return calibration
    
    def run(self):
        """Main execution"""
        corners = self.interactive_calibration(frame_number=100)
        
        if corners is None:
            return False
        
        print(f"\n✅ Calibration hoàn tất!")
        print(f"   ✨ Lưu đúng {len(self.points)} điểm VÀNG (không thay đổi)")
        print(f"   🎯 Chia sân theo CHIỀU RỘNG (2 nửa: {self.court_width/2:.2f}m mỗi bên)")
        
        self.save_calibration(corners)
        
        print("\n" + "="*70)
        print("✅ SẴN SÀNG TRACKING!")
        print("="*70)
        print("\n💡 Chạy: python enhanced_tracking_san4.py")
        
        return True
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    
    calibrator = MultiPointCalibrator(video_path)
    success = calibrator.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
