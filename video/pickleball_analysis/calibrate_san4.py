import cv2
import numpy as np
import json

class CourtCalibrator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.points = []
        self.frame = None
        self.window_name = "Court Calibration - San4"
        
        # Court dimensions in meters (standard pickleball court)
        self.court_width = 6.10  # 20 feet
        self.court_length = 13.41  # 44 feet
        
        # Court reference points (real world coordinates)
        self.court_template = np.array([
            [0, 0],                                    # Bottom-left corner
            [self.court_width, 0],                     # Bottom-right corner  
            [self.court_width, self.court_length],     # Top-right corner
            [0, self.court_length],                    # Top-left corner
            [0, self.court_length/2],                  # Left net post
            [self.court_width, self.court_length/2],   # Right net post
            [self.court_width/2, 0],                   # Bottom center
            [self.court_width/2, self.court_length],   # Top center
            [self.court_width/2, self.court_length/2], # Center of court
            [0, self.court_length/4],                  # Left service line bottom
            [self.court_width, self.court_length/4],   # Right service line bottom
            [0, 3*self.court_length/4],                # Left service line top  
            [self.court_width, 3*self.court_length/4]  # Right service line top
        ], dtype=np.float32)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            print(f"Point {len(self.points)}: ({x}, {y})")
            
            # Draw point
            cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.frame, f"{len(self.points)}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.frame)

    def calibrate(self):
        # Get first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot read video!")
            return None
            
        self.frame = frame.copy()
        
        print("=== COURT CALIBRATION FOR SAN4 ===")
        print("Click on these points in order:")
        print("1. Bottom-left corner")
        print("2. Bottom-right corner") 
        print("3. Top-right corner")
        print("4. Top-left corner")
        print("5. Left net post")
        print("6. Right net post")
        print("7. Bottom center")
        print("8. Top center")
        print("9. Center of court")
        print("10. Left service line bottom")
        print("11. Right service line bottom") 
        print("12. Left service line top")
        print("13. Right service line top")
        print("\nControls:")
        print("- Click to add points")
        print("- Press 'q' or 's' to save (need at least 4 points)")
        print("- Press 'r' to reset points")
        print("- Press ESC to cancel")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                if len(self.points) >= 4:
                    print(f"Saving calibration with {len(self.points)} points...")
                    break
                else:
                    print(f"Need at least 4 points, you have {len(self.points)}")
                    
            elif key == ord('r'):
                self.points = []
                self.frame = frame.copy()
                cv2.imshow(self.window_name, self.frame)
                print("Points reset!")
                
            elif key == ord('s'):  # Save with current points
                if len(self.points) >= 4:
                    print(f"Saving calibration with {len(self.points)} points...")
                    break
                else:
                    print(f"Need at least 4 points, you have {len(self.points)}")
                    
            elif key == 27:  # ESC key
                print("Calibration cancelled!")
                cv2.destroyAllWindows()
                self.cap.release()
                return None
        
        cv2.destroyAllWindows()
        self.cap.release()
        
        if len(self.points) < 4:
            print("Not enough points for calibration!")
            return None
            
        # Calculate homography
        image_points = np.array(self.points, dtype=np.float32)
        
        # Use first N points that match our template
        num_points = min(len(self.points), len(self.court_template))
        world_points = self.court_template[:num_points]
        
        if num_points >= 4:
            homography, _ = cv2.findHomography(image_points[:num_points], world_points)
            
            calibration_data = {
                'video_path': self.video_path,
                'image_points': image_points.tolist(),
                'world_points': world_points.tolist(), 
                'homography': homography.tolist(),
                'court_width': self.court_width,
                'court_length': self.court_length,
                'num_points': num_points
            }
            
            # Save calibration
            with open('court_calibration_san4.json', 'w') as f:
                json.dump(calibration_data, f, indent=2)
                
            print(f"✓ Calibration saved with {num_points} points!")
            print(f"✓ File saved: court_calibration_san4.json")
            return calibration_data
        else:
            print("Need at least 4 points for homography!")
            return None

if __name__ == "__main__":
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    calibrator = CourtCalibrator(video_path)
    calibration = calibrator.calibrate()
    
    if calibration:
        print("Calibration successful!")
        print("File saved: court_calibration_san4.json")
    else:
        print("Calibration failed!")