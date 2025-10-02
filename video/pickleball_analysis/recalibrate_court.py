import cv2
import numpy as np
import json

class SimpleCourtCalibrator:
    def __init__(self):
        self.video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
        self.points = []
        self.frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Xử lý click chuột"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"✅ Điểm {len(self.points)}: ({x}, {y})")
                
                if len(self.points) == 4:
                    print("🎯 Đã có đủ 4 điểm! Nhấn 's' để lưu hoặc 'r' để reset")
    
    def draw_points(self, frame):
        """Vẽ các điểm đã chọn"""
        display_frame = frame.copy()
        
        # Vẽ các điểm
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Red, Cyan, Magenta, Yellow
        labels = ['1-TOP LEFT', '2-TOP RIGHT', '3-BOTTOM RIGHT', '4-BOTTOM LEFT']
        
        for i, (point, color, label) in enumerate(zip(self.points, colors, labels)):
            cv2.circle(display_frame, point, 10, color, -1)
            cv2.circle(display_frame, point, 10, (255, 255, 255), 2)
            cv2.putText(display_frame, label, (point[0] + 15, point[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Vẽ đường nối nếu có đủ điểm
        if len(self.points) > 1:
            for i in range(len(self.points)):
                if i < len(self.points) - 1:
                    cv2.line(display_frame, self.points[i], self.points[i+1], (0, 255, 0), 2)
            
            # Nối điểm cuối với điểm đầu nếu có đủ 4 điểm
            if len(self.points) == 4:
                cv2.line(display_frame, self.points[3], self.points[0], (0, 255, 0), 2)
        
        # Hiển thị hướng dẫn
        instructions = [
            f"Click để chọn điểm {len(self.points) + 1}/4" if len(self.points) < 4 else "Nhấn 's' để lưu, 'r' để reset",
            "1=Top-Left, 2=Top-Right, 3=Bottom-Right, 4=Bottom-Left",
            "Chọn theo thứ tự: Trái trên -> Phải trên -> Phải dưới -> Trái dưới"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def calibrate(self):
        """Bắt đầu quá trình calibration"""
        print("🎯 COURT CALIBRATION FOR SAN4")
        print("📋 Hướng dẫn:")
        print("   1. Click vào 4 góc sân theo thứ tự")
        print("   2. Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
        print("   3. Nhấn 's' để lưu, 'r' để reset, 'q' để thoát")
        
        # Mở video
        cap = cv2.VideoCapture(self.video_path)
        
        # Đi đến frame tốt để calibration (frame 2000)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Không thể đọc video")
            return
        
        self.frame = frame.copy()
        
        # Setup window
        cv2.namedWindow('Court Calibration - San4', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Court Calibration - San4', 1200, 800)
        cv2.setMouseCallback('Court Calibration - San4', self.mouse_callback)
        
        while True:
            display_frame = self.draw_points(self.frame)
            cv2.imshow('Court Calibration - San4', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.points = []
                print("🔄 Reset - Chọn lại 4 điểm")
            elif key == ord('s') and len(self.points) == 4:
                self.save_calibration()
                break
        
        cv2.destroyAllWindows()
        cap.release()
    
    def save_calibration(self):
        """Lưu calibration mới"""
        print("💾 Đang lưu calibration mới...")
        
        # Kích thước sân pickleball chuẩn (mét)
        court_width = 6.1    # 20 feet = 6.1m
        court_length = 13.41  # 44 feet = 13.41m
        
        # Tọa độ thực của sân (mét)
        court_coords = np.array([
            [0, 0],                        # Top-left
            [court_width, 0],              # Top-right
            [court_width, court_length],   # Bottom-right
            [0, court_length]              # Bottom-left
        ], dtype=np.float32)
        
        # Tọa độ ảnh từ user click
        image_coords = np.array(self.points, dtype=np.float32)
        
        print("📐 Image points:", self.points)
        print("📐 Court coords:", court_coords.tolist())
        
        # Tính homography
        homography, status = cv2.findHomography(image_coords, court_coords)
        
        if homography is None:
            print("❌ Không thể tính homography!")
            return
        
        # Tạo dữ liệu calibration
        calibration_data = {
            'homography': homography.tolist(),
            'court_width': court_width,
            'court_length': court_length,
            'image_points': self.points,
            'court_points': court_coords.tolist(),
            'video': 'san4.mp4'
        }
        
        # Lưu file
        with open('court_calibration_san4.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print("✅ Calibration đã lưu thành công!")
        print(f"📏 Kích thước sân: {court_width}m × {court_length}m")
        print("📁 File: court_calibration_san4.json")
        
        # Test calibration ngay
        self.test_calibration(homography, court_width, court_length)
    
    def test_calibration(self, homography, court_width, court_length):
        """Test calibration vừa tạo"""
        print("🧪 Testing calibration...")
        
        # Vẽ court boundary lên frame
        test_frame = self.frame.copy()
        
        # Các góc sân
        court_corners = np.array([
            [0, 0], [court_width, 0],
            [court_width, court_length], [0, court_length]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform về image coordinates
        image_corners = cv2.perspectiveTransform(court_corners, np.linalg.inv(homography))
        image_corners = image_corners.reshape(-1, 2).astype(int)
        
        # Vẽ court boundary
        cv2.polylines(test_frame, [image_corners], True, (0, 255, 0), 5)
        
        # Vẽ net line
        net_start = np.array([[court_width/2, 0]], dtype=np.float32).reshape(-1, 1, 2)
        net_end = np.array([[court_width/2, court_length]], dtype=np.float32).reshape(-1, 1, 2)
        
        net_start_img = cv2.perspectiveTransform(net_start, np.linalg.inv(homography))[0][0].astype(int)
        net_end_img = cv2.perspectiveTransform(net_end, np.linalg.inv(homography))[0][0].astype(int)
        
        cv2.line(test_frame, tuple(net_start_img), tuple(net_end_img), (255, 255, 255), 6)
        
        # Add labels
        cv2.putText(test_frame, 'NEW CALIBRATION TEST', (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(test_frame, 'GREEN = Court boundary', (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(test_frame, 'WHITE = Net line', (50, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Hiển thị
        cv2.namedWindow('Calibration Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration Test', 1200, 800)
        cv2.imshow('Calibration Test', test_frame)
        
        print("✅ Test complete! Nhấn bất kỳ phím nào để đóng")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    calibrator = SimpleCourtCalibrator()
    calibrator.calibrate()

if __name__ == "__main__":
    main()