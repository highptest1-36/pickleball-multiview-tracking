# Pickleball Multi-View Tracking System (Fixed Version)
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

def run_simple_detection(model, video_paths, max_frames=200):
    """Chạy detection đơn giản (không tracking) để tránh lỗi"""
    print(f"🎬 Bắt đầu detection trên {len(video_paths)} video...")
    
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
    output_path = 'pickleball_detection_output.mp4'
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
            
            # Chạy detection (không tracking) trên từng frame
            detected_frames = []
            for i, frame in enumerate(frames):
                # YOLO detection với class person(0) và sports_ball(37)
                results = model(
                    frame, 
                    classes=[0, 37],  # person và sports_ball
                    conf=0.3,
                    verbose=False
                )
                
                # Vẽ kết quả detection
                annotated_frame = results[0].plot()
                
                # Thêm label camera
                cv2.putText(annotated_frame, f'Camera {i+1}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Thêm thông tin detection
                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                    person_count = sum(1 for cls in classes if int(cls) == 0)
                    ball_count = sum(1 for cls in classes if int(cls) == 37)
                    
                    cv2.putText(annotated_frame, f'Persons: {person_count}', (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(annotated_frame, f'Balls: {ball_count}', (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                detected_frames.append(annotated_frame)
            
            # Kết hợp 4 frame thành 2x2 grid
            top_row = np.hstack((detected_frames[0], detected_frames[1]))
            bottom_row = np.hstack((detected_frames[2], detected_frames[3]))
            combined = np.vstack((top_row, bottom_row))
            
            # Ghi vào output
            out.write(combined)
            frame_count += 1
            
            # Hiển thị tiến độ
            if frame_count % 25 == 0:
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

def run_tracking_safe(model, video_paths, max_frames=200):
    """Chạy tracking an toàn với bytetrack"""
    print(f"🎬 Bắt đầu tracking an toàn trên {len(video_paths)} video...")
    
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
    output_path = 'pickleball_tracking_safe.mp4'
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
    
    # Tạo tracker riêng cho mỗi camera
    trackers = [YOLO('yolo11n.pt') for _ in range(4)]
    
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
            
            # Chạy tracking trên từng frame với tracker riêng
            tracked_frames = []
            for i, (frame, tracker) in enumerate(zip(frames, trackers)):
                try:
                    # Sử dụng ByteTrack với persist=True
                    results = tracker.track(
                        frame, 
                        persist=True,
                        classes=[0, 37],  # person và sports_ball  
                        conf=0.3,
                        iou=0.7,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )
                    
                    # Vẽ kết quả tracking
                    annotated_frame = results[0].plot(labels=True, boxes=True)
                    
                except Exception as track_error:
                    print(f"⚠️ Lỗi tracking camera {i+1}: {track_error}")
                    # Fallback về detection nếu tracking lỗi
                    results = tracker(frame, classes=[0, 37], conf=0.3, verbose=False)
                    annotated_frame = results[0].plot()
                
                # Thêm label camera
                cv2.putText(annotated_frame, f'Camera {i+1}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Thêm thông tin detection
                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                    person_count = sum(1 for cls in classes if int(cls) == 0)
                    ball_count = sum(1 for cls in classes if int(cls) == 37)
                    
                    cv2.putText(annotated_frame, f'Persons: {person_count}', (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(annotated_frame, f'Balls: {ball_count}', (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                tracked_frames.append(annotated_frame)
            
            # Kết hợp 4 frame thành 2x2 grid
            top_row = np.hstack((tracked_frames[0], tracked_frames[1]))
            bottom_row = np.hstack((tracked_frames[2], tracked_frames[3]))
            combined = np.vstack((top_row, bottom_row))
            
            # Ghi vào output
            out.write(combined)
            frame_count += 1
            
            # Hiển thị tiến độ
            if frame_count % 25 == 0:
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
    
    # 4. Chọn chế độ
    print(f"\n🔧 Chọn chế độ xử lý:")
    print(f"   1. Detection đơn giản (nhanh, ổn định)")
    print(f"   2. Tracking an toàn (chậm hơn, có tracking ID)")
    
    # Mặc định chạy detection đơn giản
    mode = 1
    
    if mode == 1:
        print(f"🚀 Chạy chế độ Detection...")
        success = run_simple_detection(model, video_paths, max_frames=150)
    else:
        print(f"🚀 Chạy chế độ Tracking...")
        success = run_tracking_safe(model, video_paths, max_frames=150)
    
    if success:
        print("\n🎉 Xử lý hoàn thành! Kiểm tra file output.")
    else:
        print("\n❌ Xử lý thất bại!")

if __name__ == "__main__":
    main()