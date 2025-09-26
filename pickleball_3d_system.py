# Pickleball 3D Reconstruction System
# Hệ thống tái tạo 3D từ 4 camera với God View và tính toán vật lý

import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import time
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
import pickle

class Camera3D:
    def __init__(self, camera_id, intrinsic_matrix=None, distortion_coeffs=None, 
                 rotation_vector=None, translation_vector=None):
        self.camera_id = camera_id
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.rotation_vector = rotation_vector
        self.translation_vector = translation_vector
        self.projection_matrix = None
        
    def set_extrinsics(self, rvec, tvec):
        """Thiết lập tham số ngoại"""
        self.rotation_vector = rvec
        self.translation_vector = tvec
        self.update_projection_matrix()
    
    def update_projection_matrix(self):
        """Cập nhật projection matrix"""
        if self.intrinsic_matrix is not None and self.rotation_vector is not None:
            R, _ = cv2.Rodrigues(self.rotation_vector)
            RT = np.hstack([R, self.translation_vector.reshape(-1, 1)])
            self.projection_matrix = self.intrinsic_matrix @ RT

class PickleballCourt3D:
    def __init__(self):
        # Kích thước sân pickleball (mét)
        self.length = 6.10  # 20 feet
        self.width = 4.27   # 14 feet
        self.net_height = 0.91  # 36 inches
        
        # Tọa độ 3D của sân (world coordinates)
        self.court_corners_3d = np.array([
            [0, 0, 0],                    # Góc 1
            [self.length, 0, 0],          # Góc 2
            [self.length, self.width, 0], # Góc 3
            [0, self.width, 0]            # Góc 4
        ], dtype=np.float32)
        
        # Đường net
        self.net_points_3d = np.array([
            [self.length/2, 0, 0],
            [self.length/2, 0, self.net_height],
            [self.length/2, self.width, self.net_height],
            [self.length/2, self.width, 0]
        ], dtype=np.float32)

class Pickleball3DTracker:
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.cameras = {}
        self.court = PickleballCourt3D()
        self.yolo_model = None
        self.tracking_history = []
        self.physics_data = []
        
    def initialize_yolo(self):
        """Khởi tạo YOLO model"""
        print("🔄 Đang tải YOLO model...")
        self.yolo_model = YOLO('yolo11n.pt')
        print("✅ YOLO model đã được tải!")
    
    def calibrate_cameras_manual(self):
        """Hiệu chuẩn camera thủ công với giả định"""
        print("📏 Đang hiệu chuẩn cameras...")
        
        # Tham số camera giả định (có thể điều chỉnh)
        image_size = (1920, 1080)
        focal_length = 800  # pixels
        
        # Ma trận nội K giả định
        K = np.array([
            [focal_length, 0, image_size[0]/2],
            [0, focal_length, image_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Hệ số méo giả định
        dist_coeffs = np.zeros(5, dtype=np.float32)
        
        # Vị trí camera giả định (4 góc sân)
        camera_positions = [
            # Camera 1: Góc trước trái
            {'rvec': np.array([0.3, 0.0, 0.0]), 'tvec': np.array([-2.0, -1.0, 2.0])},
            # Camera 2: Góc trước phải  
            {'rvec': np.array([0.3, 0.0, 0.5]), 'tvec': np.array([8.0, -1.0, 2.0])},
            # Camera 3: Góc sau trái
            {'rvec': np.array([-0.3, 0.0, 3.14]), 'tvec': np.array([-2.0, 5.5, 2.0])},
            # Camera 4: Góc sau phải
            {'rvec': np.array([-0.3, 0.0, 2.64]), 'tvec': np.array([8.0, 5.5, 2.0])}
        ]
        
        # Tạo camera objects
        for i in range(4):
            cam = Camera3D(
                camera_id=i+1,
                intrinsic_matrix=K.copy(),
                distortion_coeffs=dist_coeffs.copy(),
                rotation_vector=camera_positions[i]['rvec'],
                translation_vector=camera_positions[i]['tvec']
            )
            cam.update_projection_matrix()
            self.cameras[i+1] = cam
            
        print(f"✅ Đã hiệu chuẩn {len(self.cameras)} cameras")
        return True
    
    def detect_objects_multiview(self, frames):
        """Phát hiện objects trong tất cả các view"""
        detections = {}
        
        for i, frame in enumerate(frames):
            camera_id = i + 1
            results = self.yolo_model(frame, classes=[0, 37], conf=0.3, verbose=False)
            
            detections[camera_id] = []
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for j in range(len(boxes)):
                    box = boxes[j]
                    cls = int(box.cls.cpu().numpy())
                    conf = float(box.conf.cpu().numpy())
                    xyxy = box.xyxy.cpu().numpy()[0]
                    
                    # Tính center point
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    
                    detections[camera_id].append({
                        'class': cls,
                        'confidence': conf,
                        'bbox': xyxy,
                        'center': (center_x, center_y),
                        'class_name': 'person' if cls == 0 else 'ball'
                    })
        
        return detections
    
    def triangulate_3d_point(self, points_2d, camera_ids):
        """Triangulation để tính toán vị trí 3D"""
        if len(points_2d) < 2:
            return None
            
        # Lấy projection matrices
        proj_matrices = []
        points_2d_norm = []
        
        for i, point_2d in enumerate(points_2d):
            cam_id = camera_ids[i]
            if cam_id in self.cameras:
                proj_matrices.append(self.cameras[cam_id].projection_matrix)
                points_2d_norm.append(np.array(point_2d))
        
        if len(proj_matrices) < 2:
            return None
        
        # DLT triangulation
        try:
            A = []
            for i in range(len(points_2d_norm)):
                x, y = points_2d_norm[i]
                P = proj_matrices[i]
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])
            
            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[3]  # Normalize
            
            return X[:3]  # Return 3D point
        except:
            return None
    
    def match_detections_across_views(self, detections):
        """Ghép cặp detections giữa các view"""
        matches = {'persons': [], 'balls': []}
        
        # Separate by class
        persons_by_cam = {}
        balls_by_cam = {}
        
        for cam_id, dets in detections.items():
            persons_by_cam[cam_id] = [d for d in dets if d['class'] == 0]
            balls_by_cam[cam_id] = [d for d in dets if d['class'] == 37]
        
        # Match persons (simple matching by position similarity)
        self._match_objects_by_position(persons_by_cam, matches['persons'])
        self._match_objects_by_position(balls_by_cam, matches['balls'])
        
        return matches
    
    def _match_objects_by_position(self, objects_by_cam, matches_list):
        """Ghép objects dựa trên vị trí"""
        cam_ids = list(objects_by_cam.keys())
        
        # Simple matching: nếu có ít nhất 2 camera thấy object
        for i, cam_id1 in enumerate(cam_ids):
            for obj1 in objects_by_cam[cam_id1]:
                match_group = {'cameras': [cam_id1], 'points_2d': [obj1['center']], 'confidences': [obj1['confidence']]}
                
                for j, cam_id2 in enumerate(cam_ids[i+1:], i+1):
                    for obj2 in objects_by_cam[cam_id2]:
                        # Simple distance-based matching (có thể cải thiện)
                        if abs(obj1['center'][0] - obj2['center'][0]) < 200:  # Threshold
                            match_group['cameras'].append(cam_id2)
                            match_group['points_2d'].append(obj2['center'])
                            match_group['confidences'].append(obj2['confidence'])
                            break
                
                if len(match_group['cameras']) >= 2:
                    matches_list.append(match_group)
    
    def calculate_3d_positions(self, matches):
        """Tính toán vị trí 3D từ matches"""
        positions_3d = {'persons': [], 'balls': []}
        
        for obj_type in ['persons', 'balls']:
            for match in matches[obj_type]:
                point_3d = self.triangulate_3d_point(match['points_2d'], match['cameras'])
                if point_3d is not None:
                    positions_3d[obj_type].append({
                        'position_3d': point_3d,
                        'cameras': match['cameras'],
                        'confidence': np.mean(match['confidences'])
                    })
        
        return positions_3d
    
    def calculate_physics(self, ball_positions_history):
        """Tính toán vật lý của bóng"""
        physics = {}
        
        if len(ball_positions_history) < 2:
            return physics
        
        # Tính vận tốc
        dt = 1.0 / 30.0  # Giả định 30 FPS
        
        pos_current = ball_positions_history[-1]
        pos_previous = ball_positions_history[-2]
        
        velocity = (pos_current - pos_previous) / dt
        speed = np.linalg.norm(velocity)
        
        physics = {
            'velocity': velocity,
            'speed': speed,
            'height': pos_current[2],
            'position': pos_current
        }
        
        # Tính gia tốc nếu có đủ điểm
        if len(ball_positions_history) >= 3:
            pos_prev2 = ball_positions_history[-3]
            vel_previous = (pos_previous - pos_prev2) / dt
            acceleration = (velocity - vel_previous) / dt
            physics['acceleration'] = acceleration
        
        return physics
    
    def create_god_view(self, positions_3d, physics_data=None):
        """Tạo God View từ trên xuống"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 2D Top-down view
        ax1.set_xlim(-1, 7)
        ax1.set_ylim(-1, 5)
        ax1.set_aspect('equal')
        ax1.set_title('God View - Top Down', fontsize=14)
        ax1.set_xlabel('Length (m)')
        ax1.set_ylabel('Width (m)')
        
        # Vẽ sân
        court_corners_2d = self.court.court_corners_3d[:, :2]
        court_corners_2d = np.vstack([court_corners_2d, court_corners_2d[0]])  # Close the loop
        ax1.plot(court_corners_2d[:, 0], court_corners_2d[:, 1], 'k-', linewidth=2, label='Court')
        
        # Vẽ net
        net_x = self.court.length / 2
        ax1.plot([net_x, net_x], [0, self.court.width], 'r-', linewidth=3, label='Net')
        
        # Vẽ người chơi
        if 'persons' in positions_3d:
            for i, person in enumerate(positions_3d['persons']):
                pos = person['position_3d']
                ax1.scatter(pos[0], pos[1], c='blue', s=100, marker='o', alpha=0.7)
                ax1.annotate(f'P{i+1}', (pos[0], pos[1]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10)
        
        # Vẽ bóng
        if 'balls' in positions_3d:
            for i, ball in enumerate(positions_3d['balls']):
                pos = ball['position_3d']
                ax1.scatter(pos[0], pos[1], c='red', s=50, marker='o', alpha=0.8)
                ax1.annotate(f'Ball', (pos[0], pos[1]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, color='red')
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 3D View
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_xlim(0, 6)
        ax2.set_ylim(0, 4)
        ax2.set_zlim(0, 3)
        ax2.set_title('3D View', fontsize=14)
        ax2.set_xlabel('Length (m)')
        ax2.set_ylabel('Width (m)')
        ax2.set_zlabel('Height (m)')
        
        # Vẽ sân 3D
        court_corners = self.court.court_corners_3d
        for i in range(4):
            j = (i + 1) % 4
            ax2.plot([court_corners[i, 0], court_corners[j, 0]], 
                    [court_corners[i, 1], court_corners[j, 1]], 
                    [court_corners[i, 2], court_corners[j, 2]], 'k-', linewidth=2)
        
        # Vẽ net 3D
        net_points = self.court.net_points_3d
        ax2.plot([net_points[0, 0], net_points[1, 0]], 
                [net_points[0, 1], net_points[1, 1]], 
                [net_points[0, 2], net_points[1, 2]], 'r-', linewidth=3)
        ax2.plot([net_points[1, 0], net_points[2, 0]], 
                [net_points[1, 1], net_points[2, 1]], 
                [net_points[1, 2], net_points[2, 2]], 'r-', linewidth=3)
        ax2.plot([net_points[2, 0], net_points[3, 0]], 
                [net_points[2, 1], net_points[3, 1]], 
                [net_points[2, 2], net_points[3, 2]], 'r-', linewidth=3)
        
        # Vẽ người chơi 3D
        if 'persons' in positions_3d:
            for person in positions_3d['persons']:
                pos = person['position_3d']
                ax2.scatter(pos[0], pos[1], pos[2], c='blue', s=100, alpha=0.7)
        
        # Vẽ bóng 3D
        if 'balls' in positions_3d:
            for ball in positions_3d['balls']:
                pos = ball['position_3d']
                ax2.scatter(pos[0], pos[1], pos[2], c='red', s=50, alpha=0.8)
        
        # Thêm thông tin physics nếu có
        if physics_data:
            info_text = f"Ball Speed: {physics_data.get('speed', 0):.2f} m/s\n"
            info_text += f"Height: {physics_data.get('height', 0):.2f} m"
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def process_frame(self, frames, frame_count):
        """Xử lý một frame"""
        # Detect objects
        detections = self.detect_objects_multiview(frames)
        
        # Match detections across views
        matches = self.match_detections_across_views(detections)
        
        # Calculate 3D positions
        positions_3d = self.calculate_3d_positions(matches)
        
        # Store tracking history
        frame_data = {
            'frame': frame_count,
            'detections': detections,
            'positions_3d': positions_3d
        }
        self.tracking_history.append(frame_data)
        
        # Calculate physics for ball
        physics_data = None
        if 'balls' in positions_3d and len(positions_3d['balls']) > 0:
            # Lấy lịch sử vị trí bóng
            ball_history = []
            for hist in self.tracking_history[-10:]:  # 10 frames gần nhất
                if 'balls' in hist['positions_3d'] and len(hist['positions_3d']['balls']) > 0:
                    ball_history.append(hist['positions_3d']['balls'][0]['position_3d'])
            
            if len(ball_history) >= 2:
                physics_data = self.calculate_physics(ball_history)
                self.physics_data.append(physics_data)
        
        return positions_3d, physics_data
    
    def run_3d_tracking(self, max_frames=100):
        """Chạy 3D tracking chính"""
        print("🚀 Bắt đầu 3D Tracking...")
        
        # Initialize
        self.initialize_yolo()
        self.calibrate_cameras_manual()
        
        # Open video captures
        caps = []
        for i, path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"❌ Không thể mở video {i+1}")
                return False
            caps.append(cap)
        
        frame_count = 0
        start_time = time.time()
        
        print(f"📹 Xử lý {max_frames} frames...")
        
        try:
            while frame_count < max_frames:
                frames = []
                all_valid = True
                
                # Read frames
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        all_valid = False
                        break
                    # Resize để xử lý nhanh
                    frame = cv2.resize(frame, (960, 540))
                    frames.append(frame)
                
                if not all_valid or len(frames) < 4:
                    break
                
                # Process frame
                positions_3d, physics_data = self.process_frame(frames, frame_count)
                
                # Show progress
                if frame_count % 20 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    persons_count = len(positions_3d.get('persons', []))
                    balls_count = len(positions_3d.get('balls', []))
                    
                    print(f"Frame {frame_count}: {persons_count} persons, {balls_count} balls, {fps:.1f} FPS")
                    
                    # Create visualization every 20 frames
                    if frame_count > 0:
                        fig = self.create_god_view(positions_3d, physics_data)
                        plt.savefig(f'god_view_frame_{frame_count:04d}.png', dpi=100, bbox_inches='tight')
                        plt.close(fig)
                
                frame_count += 1
        
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            for cap in caps:
                cap.release()
        
        total_time = time.time() - start_time
        print(f"\n✅ Hoàn thành!")
        print(f"📊 Xử lý {frame_count} frames trong {total_time:.1f}s")
        print(f"🎯 Tốc độ trung bình: {frame_count/total_time:.1f} FPS")
        
        # Save results
        self.save_results()
        
        return True
    
    def save_results(self):
        """Lưu kết quả"""
        print("💾 Đang lưu kết quả...")
        
        # Save tracking history
        with open('tracking_3d_results.pkl', 'wb') as f:
            pickle.dump(self.tracking_history, f)
        
        # Save physics data
        with open('physics_data.pkl', 'wb') as f:
            pickle.dump(self.physics_data, f)
        
        print("✅ Đã lưu kết quả!")

def main():
    """Hàm main"""
    print("🏓 PICKLEBALL 3D RECONSTRUCTION SYSTEM")
    print("=" * 50)
    
    # Đường dẫn video
    base_path = r"C:\Users\highp\pickerball"
    video_folders = [
        "e4e66c2058ff-0.0.0.0-3000-1-0-mzle9eCKS2oQvLJa7rOE000000",
        "e4e66c2058ff-0.0.0.0-3000-2-0-vvkoKtKIUN7KS72O4bfR000000", 
        "e4e66c2058ff-0.0.0.0-3000-3-0-a4TtYafdNkjZQjVO5hll000000",
        "e4e66c2058ff-0.0.0.0-3000-4-0-ZhV2hb2DFg8xhbXYcpWn000000"
    ]
    
    # Tìm video files
    video_paths = []
    for folder in video_folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(folder_path, mp4_files[0])
                video_paths.append(video_path)
                print(f"✅ Found: {mp4_files[0]}")
    
    if len(video_paths) < 4:
        print(f"❌ Cần 4 videos, chỉ tìm thấy {len(video_paths)}")
        return
    
    # Create tracker
    tracker = Pickleball3DTracker(video_paths)
    
    # Run 3D tracking
    success = tracker.run_3d_tracking(max_frames=80)
    
    if success:
        print("\n🎉 3D Tracking hoàn thành!")
        print("📁 Kiểm tra files:")
        print("   - god_view_frame_*.png: Ảnh God View")
        print("   - tracking_3d_results.pkl: Dữ liệu tracking")
        print("   - physics_data.pkl: Dữ liệu vật lý")

if __name__ == "__main__":
    main()