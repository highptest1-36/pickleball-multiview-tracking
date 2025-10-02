import cv2
import numpy as np
import json

class CourtBoundaryValidator:
    def __init__(self):
        # Load existing calibration
        try:
            with open('court_calibration_san4.json', 'r') as f:
                self.calibration = json.load(f)
                print("✅ Loaded existing calibration")
        except:
            print("❌ No existing calibration found")
            return
        
        self.homography = np.array(self.calibration['homography'])
        self.court_width = self.calibration['court_width']
        self.court_length = self.calibration['court_length']
        
        # Load video
        self.video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
        self.cap = cv2.VideoCapture(self.video_path)
        
        print(f"🏟️ Current court dimensions: {self.court_width:.1f}m × {self.court_length:.1f}m")
        
    def draw_court_boundaries(self, frame):
        """Draw the court boundaries based on current calibration"""
        # Standard pickleball court dimensions (in meters)
        # Court boundary points
        court_points = np.array([
            [0, 0],                                    # Top-left
            [self.court_width, 0],                     # Top-right  
            [self.court_width, self.court_length],     # Bottom-right
            [0, self.court_length]                     # Bottom-left
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform to image coordinates
        image_points = cv2.perspectiveTransform(court_points, np.linalg.inv(self.homography))
        image_points = image_points.reshape(-1, 2).astype(int)
        
        # Draw court boundary (GREEN)
        cv2.polylines(frame, [image_points], True, (0, 255, 0), 4)
        
        # Draw net line (WHITE) - in the middle
        net_start = np.array([[self.court_width/2, 0]], dtype=np.float32).reshape(-1, 1, 2)
        net_end = np.array([[self.court_width/2, self.court_length]], dtype=np.float32).reshape(-1, 1, 2)
        
        net_start_img = cv2.perspectiveTransform(net_start, np.linalg.inv(self.homography))[0][0].astype(int)
        net_end_img = cv2.perspectiveTransform(net_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        cv2.line(frame, tuple(net_start_img), tuple(net_end_img), (255, 255, 255), 6)
        
        # Draw service lines (BLUE dashed)
        # Non-volley zones (kitchen) - 2.13m from net on each side
        nvz_distance = 2.13  # Non-volley zone depth
        
        # Left side non-volley line
        nvz_left_start = np.array([[0, self.court_length/2 - nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        nvz_left_end = np.array([[self.court_width, self.court_length/2 - nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        
        nvz_left_start_img = cv2.perspectiveTransform(nvz_left_start, np.linalg.inv(self.homography))[0][0].astype(int)
        nvz_left_end_img = cv2.perspectiveTransform(nvz_left_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        # Right side non-volley line  
        nvz_right_start = np.array([[0, self.court_length/2 + nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        nvz_right_end = np.array([[self.court_width, self.court_length/2 + nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        
        nvz_right_start_img = cv2.perspectiveTransform(nvz_right_start, np.linalg.inv(self.homography))[0][0].astype(int)
        nvz_right_end_img = cv2.perspectiveTransform(nvz_right_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        # Draw non-volley zone lines (BLUE)
        cv2.line(frame, tuple(nvz_left_start_img), tuple(nvz_left_end_img), (255, 100, 0), 3)
        cv2.line(frame, tuple(nvz_right_start_img), tuple(nvz_right_end_img), (255, 100, 0), 3)
        
        # Draw center service line (YELLOW)
        center_service_start = np.array([[self.court_width/2, self.court_length/2 - nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        center_service_end = np.array([[self.court_width/2, self.court_length/2 + nvz_distance]], dtype=np.float32).reshape(-1, 1, 2)
        
        center_service_start_img = cv2.perspectiveTransform(center_service_start, np.linalg.inv(self.homography))[0][0].astype(int)
        center_service_end_img = cv2.perspectiveTransform(center_service_end, np.linalg.inv(self.homography))[0][0].astype(int)
        
        cv2.line(frame, tuple(center_service_start_img), tuple(center_service_end_img), (0, 255, 255), 3)
        
        # Add labels
        cv2.putText(frame, 'COURT BOUNDARY', (50, frame.shape[0] - 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, 'NET LINE', (50, frame.shape[0] - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, 'NON-VOLLEY ZONE', (50, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        cv2.putText(frame, 'SERVICE LINE', (50, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def validate_court_mapping(self):
        """Show current court boundary mapping on video"""
        print("🔍 Validating court boundary mapping...")
        print("📝 GREEN = Court boundary")
        print("📝 WHITE = Net line") 
        print("📝 BLUE = Non-volley zone")
        print("📝 YELLOW = Service center line")
        print("⏹️  Press 'q' to quit, 'r' to recalibrate")
        
        cv2.namedWindow('Court Boundary Validation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Court Boundary Validation', 1200, 800)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            
            frame_count += 1
            
            # Show every 30th frame for performance
            if frame_count % 30 == 0:
                # Draw court boundaries
                frame_with_court = self.draw_court_boundaries(frame.copy())
                
                # Add info overlay
                info_text = [
                    f"Frame: {frame_count}",
                    f"Court: {self.court_width:.1f}m x {self.court_length:.1f}m",
                    "Check if GREEN boundary matches real court!",
                    "Press 'r' if boundary is wrong"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(frame_with_court, text, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Court Boundary Validation', frame_with_court)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("🔧 Starting recalibration...")
                self.recalibrate_court()
                break
        
        cv2.destroyAllWindows()
    
    def recalibrate_court(self):
        """Start court recalibration process"""
        print("🎯 Starting manual court calibration...")
        print("📌 Click on court corners in this order:")
        print("   1. Top-left corner of court")
        print("   2. Top-right corner of court") 
        print("   3. Bottom-right corner of court")
        print("   4. Bottom-left corner of court")
        
        # Get first frame for calibration
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)  # Skip to frame 1000
        ret, frame = self.cap.read()
        
        if not ret:
            print("❌ Cannot read frame for calibration")
            return
        
        self.calibration_points = []
        self.calibration_frame = frame.copy()
        
        cv2.namedWindow('Court Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Court Calibration', 1200, 800)
        cv2.setMouseCallback('Court Calibration', self.mouse_callback)
        
        while len(self.calibration_points) < 4:
            display_frame = self.calibration_frame.copy()
            
            # Draw existing points
            for i, point in enumerate(self.calibration_points):
                cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
                cv2.putText(display_frame, f'{i+1}', 
                           (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instructions
            instructions = [
                f"Click point {len(self.calibration_points) + 1}/4",
                "1=Top-left, 2=Top-right", 
                "3=Bottom-right, 4=Bottom-left"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(display_frame, instruction, (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Court Calibration', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if len(self.calibration_points) == 4:
            self.save_calibration()
        
        cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for calibration"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.calibration_points) < 4:
            self.calibration_points.append((x, y))
            print(f"✅ Point {len(self.calibration_points)}: ({x}, {y})")
    
    def save_calibration(self):
        """Save new calibration"""
        print("💾 Calculating new homography...")
        
        # Standard pickleball court dimensions (meters)
        court_width = 6.1   # 20 feet
        court_length = 13.41  # 44 feet
        
        # Real-world court coordinates (in meters)
        court_coords = np.array([
            [0, 0],                        # Top-left
            [court_width, 0],              # Top-right
            [court_width, court_length],   # Bottom-right
            [0, court_length]              # Bottom-left
        ], dtype=np.float32)
        
        # Image coordinates from user clicks
        image_coords = np.array(self.calibration_points, dtype=np.float32)
        
        # Calculate homography
        homography, _ = cv2.findHomography(image_coords, court_coords)
        
        # Save calibration
        calibration_data = {
            'homography': homography.tolist(),
            'court_width': court_width,
            'court_length': court_length,
            'image_points': self.calibration_points,
            'court_points': court_coords.tolist()
        }
        
        with open('court_calibration_san4.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print("✅ New calibration saved!")
        print(f"📐 Court dimensions: {court_width}m × {court_length}m")
        
        # Test the new calibration
        self.homography = homography
        self.court_width = court_width
        self.court_length = court_length
        
        print("🔍 Testing new calibration...")
        self.validate_court_mapping()

def main():
    validator = CourtBoundaryValidator()
    validator.validate_court_mapping()

if __name__ == "__main__":
    main()