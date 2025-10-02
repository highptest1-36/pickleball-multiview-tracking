import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

def test_calibration():
    # Load calibration
    with open('court_calibration_san4.json', 'r') as f:
        calibration = json.load(f)
    
    homography = np.array(calibration['homography'])
    image_points = np.array(calibration['image_points'])
    
    # Load first frame
    video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Cannot read video!")
        return
    
    # Draw calibration points on frame
    frame_with_points = frame.copy()
    for i, point in enumerate(image_points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame_with_points, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(frame_with_points, str(i+1), (x+15, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Create court grid overlay
    court_width = calibration['court_width']
    court_length = calibration['court_length']
    
    # Create court grid points
    grid_points = []
    steps = 20
    for i in range(steps + 1):
        for j in range(steps + 1):
            x = (i / steps) * court_width
            y = (j / steps) * court_length
            grid_points.append([x, y])
    
    grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Transform grid to image coordinates
    inv_homography = np.linalg.inv(homography)
    image_grid = cv2.perspectiveTransform(grid_points, inv_homography)
    image_grid = image_grid.reshape(-1, 2)
    
    # Draw grid overlay
    frame_with_grid = frame.copy()
    for point in image_grid:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame_with_grid, (x, y), 2, (255, 0, 0), -1)
    
    # Show results
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Frame - San4')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(frame_with_points, cv2.COLOR_BGR2RGB))
    plt.title(f'Calibration Points ({len(image_points)} points)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(frame_with_grid, cv2.COLOR_BGR2RGB))
    plt.title('Court Grid Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('san4_calibration_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Calibration test completed!")
    print(f"✓ Used {len(image_points)} calibration points")
    print(f"✓ Court dimensions: {court_width:.2f}m x {court_length:.2f}m")
    print("✓ Test image saved as: san4_calibration_test.png")

if __name__ == "__main__":
    test_calibration()