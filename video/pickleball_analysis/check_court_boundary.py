import cv2
import numpy as np
import json

def show_current_court_boundary():
    """Hiển thị court boundary hiện tại để kiểm tra"""
    
    # Load calibration hiện tại
    try:
        with open('court_calibration_san4.json', 'r') as f:
            calibration = json.load(f)
        print("✅ Loaded existing calibration")
    except:
        print("❌ No calibration file found")
        return
    
    homography = np.array(calibration['homography'])
    court_width = calibration['court_width'] 
    court_length = calibration['court_length']
    
    print(f"📐 Current court: {court_width:.1f}m × {court_length:.1f}m")
    
    # Load video
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Go to frame 2000 for good view
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Cannot read video frame")
        return
    
    # Draw court boundaries
    # Court corners
    court_corners = np.array([
        [0, 0],                                    # Top-left
        [court_width, 0],                          # Top-right  
        [court_width, court_length],               # Bottom-right
        [0, court_length]                          # Bottom-left
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    # Transform to image coordinates
    image_corners = cv2.perspectiveTransform(court_corners, np.linalg.inv(homography))
    image_corners = image_corners.reshape(-1, 2).astype(int)
    
    # Draw court boundary (THICK GREEN)
    cv2.polylines(frame, [image_corners], True, (0, 255, 0), 8)
    
    # Draw net line (WHITE)
    net_start = np.array([[court_width/2, 0]], dtype=np.float32).reshape(-1, 1, 2)
    net_end = np.array([[court_width/2, court_length]], dtype=np.float32).reshape(-1, 1, 2)
    
    net_start_img = cv2.perspectiveTransform(net_start, np.linalg.inv(homography))[0][0].astype(int)
    net_end_img = cv2.perspectiveTransform(net_end, np.linalg.inv(homography))[0][0].astype(int)
    
    cv2.line(frame, tuple(net_start_img), tuple(net_end_img), (255, 255, 255), 10)
    
    # Draw corner points with numbers
    corner_labels = ['1-TL', '2-TR', '3-BR', '4-BL']
    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
    
    for i, (corner, label, color) in enumerate(zip(image_corners, corner_labels, colors)):
        cv2.circle(frame, tuple(corner), 15, color, -1)
        cv2.putText(frame, label, (corner[0] + 20, corner[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    # Add labels
    cv2.putText(frame, 'CURRENT COURT BOUNDARY CHECK', (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, 'GREEN = Court boundary', (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, 'WHITE = Net line', (50, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, 'Does GREEN match real court?', (50, 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Save image for inspection
    cv2.imwrite('court_boundary_check.jpg', frame)
    print("💾 Saved court_boundary_check.jpg")
    
    # Display
    cv2.namedWindow('Court Boundary Check', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Court Boundary Check', 1200, 800)
    cv2.imshow('Court Boundary Check', frame)
    
    print("🔍 Court boundary displayed!")
    print("📝 Check if GREEN lines match the real court in video")
    print("⏹️  Press any key to close")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    show_current_court_boundary()