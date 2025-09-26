# Pickleball Multi-View Tracking System
# Hệ thống tracking người chơi và bóng từ 4 góc camera

import os
import cv2
import numpy as np
from ultralytics import YOLO
import time

def check_video_files():
    """Kiểm tra và liệt kê các file video"""
    base_path = r"C:\Users\highp\pickerball"
    
    video_folders = [
        "e4e66c2058ff-0.0.0.0-3000-1-0-mzle9eCKS2oQvLJa7rOE000000",
        "e4e66c2058ff-0.0.0.0-3000-2-0-vvkoKtKIUN7KS72O4bfR000000", 
        "e4e66c2058ff-0.0.0.0-3000-3-0-a4TtYafdNkjZQjVO5hll000000",
        "e4e66c2058ff-0.0.0.0-3000-4-0-ZhV2hb2DFg8xhbXYcpWn000000"
    ]
    
    video_paths = []
    print("🔍 Đang kiểm tra video files...")
    
    for i, folder in enumerate(video_folders):
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
            if mp4_files:
                video_path = os.path.join(folder_path, mp4_files[0])
                video_paths.append(video_path)
                print(f"✅ Camera {i+1}: {mp4_files[0]}")
            else:
                print(f"❌ Camera {i+1}: Không tìm thấy file .mp4")
        else:
            print(f"❌ Camera {i+1}: Không tìm thấy folder")
    
    return video_paths

def load_yolo_model():
    """Tải YOLO model"""
    print("🔄 Đang tải YOLO11 model...")
    try:
        model = YOLO('yolo11n.pt')  # Sẽ tự động download
        print("✅ Model đã được tải thành công!")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        return None

def get_video_info(video_path):
    """Lấy thông tin video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': frame_count / fps if fps > 0 else 0
    }

def run_tracking(model, video_paths, max_frames=500):
    """Chạy tracking trên 4 video"""
    print(f"🎬 Bắt đầu tracking trên {len(video_paths)} video...")
    
    # Mở video captures
    caps = []
    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ Không thể mở video {i+1}")
            return False
        caps.append(cap)
    
    # Lấy thông tin video
    info = get_video_info(video_paths[0])
    if info is None:
        print("❌ Không thể lấy thông tin video")
        return False
    
    # Tạo video writer
    output_path = 'pickleball_tracking_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        info['fps'], 
        (info['width'] * 2, info['height'] * 2)
    )
    
    frame_count = 0
    start_time = time.time()
    
    print(f"📹 Xử lý video: {info['width']}x{info['height']}, {info['fps']} FPS")
    print(f"⏱️ Giới hạn: {max_frames} frames")
    
    try:
        while frame_count < max_frames:
            frames = []
            all_valid = True
            
            # Đọc frame từ tất cả video
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    all_valid = False
                    break
                frames.append(frame)
            
            if not all_valid or len(frames) < 4:
                print("📹 Hết video hoặc không đủ frames")
                break
            
            # Chạy tracking trên từng frame
            tracked_frames = []
            for i, frame in enumerate(frames):
                # YOLO tracking với class person(0) và sports_ball(37)
                results = model.track(
                    frame, 
                    persist=True, 
                    classes=[0, 37],  # person và sports_ball
                    conf=0.4,
                    verbose=False
                )
                
                # Vẽ kết quả
                annotated_frame = results[0].plot()
                
                # Thêm label camera
                cv2.putText(annotated_frame, f'Camera {i+1}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                tracked_frames.append(annotated_frame)
            
            # Kết hợp 4 frame thành 2x2 grid
            top_row = np.hstack((tracked_frames[0], tracked_frames[1]))
            bottom_row = np.hstack((tracked_frames[2], tracked_frames[3]))
            combined = np.vstack((top_row, bottom_row))
            
            # Ghi vào output
            out.write(combined)
            frame_count += 1
            
            # Hiển thị tiến độ
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                print(f"⏳ Frame {frame_count}/{max_frames} - Tốc độ: {fps_processing:.1f} FPS")
    
    except Exception as e:
        print(f"❌ Lỗi trong quá trình xử lý: {e}")
    
    finally:
        # Dọn dẹp
        for cap in caps:
            cap.release()
        out.release()
        
        total_time = time.time() - start_time
        print(f"\n✅ Hoàn thành!")
        print(f"📊 Đã xử lý {frame_count} frames trong {total_time:.1f}s")
        print(f"📁 Output: {os.path.abspath(output_path)}")
        
        return True

def main():
    """Hàm main"""
    print("🏓 PICKLEBALL MULTI-VIEW TRACKING SYSTEM")
    print("=" * 50)
    
    # 1. Kiểm tra video files
    video_paths = check_video_files()
    if len(video_paths) < 4:
        print(f"❌ Cần 4 video, chỉ tìm thấy {len(video_paths)}")
        return
    
    # 2. Hiển thị thông tin video
    info = get_video_info(video_paths[0])
    if info:
        print(f"\n📊 Thông tin video:")
        print(f"   - Độ phân giải: {info['width']}x{info['height']}")
        print(f"   - FPS: {info['fps']}")
        print(f"   - Thời lượng: {info['duration']:.1f}s")
        print(f"   - Tổng frames: {info['frame_count']}")
    
    # 3. Tải model
    model = load_yolo_model()
    if model is None:
        return
    
    # 4. Chạy tracking
    success = run_tracking(model, video_paths, max_frames=300)  # Giới hạn 300 frames để test
    
    if success:
        print("\n🎉 Tracking hoàn thành! Kiểm tra file output.")
    else:
        print("\n❌ Tracking thất bại!")

if __name__ == "__main__":
    main()