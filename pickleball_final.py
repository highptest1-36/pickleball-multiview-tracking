# Pickleball Multi-View Tracking System (Final Version)
# Hệ thống tracking người chơi và bóng từ 4 góc camera - Xử lý resize frame

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

def get_all_video_info(video_paths):
    """Lấy thông tin tất cả video"""
    all_info = []
    for i, path in enumerate(video_paths):
        info = get_video_info(path)
        if info:
            print(f"📹 Video {i+1}: {info['width']}x{info['height']}, {info['fps']:.1f}fps")
            all_info.append(info)
        else:
            print(f"❌ Không thể đọc video {i+1}")
    return all_info

def run_multi_view_detection(model, video_paths, max_frames=100):
    """Chạy detection trên 4 video với resize chuẩn hóa"""
    print(f"🎬 Bắt đầu multi-view detection...")
    
    # Mở video captures
    caps = []
    video_infos = []
    
    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ Không thể mở video {i+1}")
            return False
        caps.append(cap)
        
        # Lấy thông tin từng video
        info = get_video_info(path)
        video_infos.append(info)
    
    # Sử dụng kích thước chuẩn (resize tất cả về cùng kích thước)
    standard_width = 960  # Giảm để xử lý nhanh hơn
    standard_height = 540
    target_fps = 30
    
    print(f"📐 Chuẩn hóa tất cả video về: {standard_width}x{standard_height}")
    
    # Tạo video writer
    output_path = 'pickleball_multiview_final.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        target_fps, 
        (standard_width * 2, standard_height * 2)  # 2x2 grid
    )
    
    frame_count = 0
    start_time = time.time()
    
    print(f"⏱️ Giới hạn: {max_frames} frames")
    print(f"🎬 Bắt đầu xử lý...")
    
    try:
        while frame_count < max_frames:
            frames = []
            all_valid = True
            
            # Đọc frame từ tất cả video
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    all_valid = False
                    break
                
                # Resize frame về kích thước chuẩn
                frame_resized = cv2.resize(frame, (standard_width, standard_height))
                frames.append(frame_resized)
            
            if not all_valid or len(frames) < 4:
                print("📹 Hết video hoặc không đủ frames")
                break
            
            # Chạy detection trên từng frame
            detected_frames = []
            total_persons = 0
            total_balls = 0
            
            for i, frame in enumerate(frames):
                # YOLO detection
                results = model(
                    frame, 
                    classes=[0, 37],  # person(0) và sports_ball(37)
                    conf=0.25,  # Giảm confidence để detect nhiều hơn
                    verbose=False
                )
                
                # Vẽ kết quả detection
                annotated_frame = results[0].plot()
                
                # Thêm label camera
                cv2.putText(annotated_frame, f'Camera {i+1}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Đếm objects
                person_count = 0
                ball_count = 0
                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
                    person_count = sum(1 for cls in classes if int(cls) == 0)
                    ball_count = sum(1 for cls in classes if int(cls) == 37)
                    
                    total_persons += person_count
                    total_balls += ball_count
                    
                    # Hiển thị số lượng
                    cv2.putText(annotated_frame, f'Persons: {person_count}', (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(annotated_frame, f'Balls: {ball_count}', (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                detected_frames.append(annotated_frame)
            
            # Kết hợp 4 frame thành 2x2 grid
            top_row = np.hstack((detected_frames[0], detected_frames[1]))
            bottom_row = np.hstack((detected_frames[2], detected_frames[3]))
            combined = np.vstack((top_row, bottom_row))
            
            # Thêm thông tin tổng quan
            info_text = f"Frame {frame_count+1} | Total Persons: {total_persons} | Total Balls: {total_balls}"
            cv2.putText(combined, info_text, (10, standard_height * 2 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Ghi vào output
            out.write(combined)
            frame_count += 1
            
            # Hiển thị tiến độ
            if frame_count % 20 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                print(f"⏳ Frame {frame_count}/{max_frames} - "
                      f"Tốc độ: {fps_processing:.1f} FPS - "
                      f"Persons: {total_persons}, Balls: {total_balls}")
    
    except Exception as e:
        print(f"❌ Lỗi trong quá trình xử lý: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Dọn dẹp
        print("🔄 Đang dọn dẹp...")
        for cap in caps:
            cap.release()
        out.release()
        
        total_time = time.time() - start_time
        print(f"\n✅ Hoàn thành!")
        print(f"📊 Thống kê xử lý:")
        print(f"   - Frames xử lý: {frame_count}")
        print(f"   - Thời gian: {total_time:.1f}s")
        print(f"   - Tốc độ trung bình: {frame_count/total_time:.1f} FPS")
        print(f"📁 Output video: {os.path.abspath(output_path)}")
        
        # Kiểm tra kích thước file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            print(f"💾 Kích thước file: {file_size:.1f} MB")
        
        return frame_count > 0

def main():
    """Hàm main"""
    print("🏓 PICKLEBALL MULTI-VIEW TRACKING SYSTEM - FINAL")
    print("=" * 55)
    
    # 1. Kiểm tra video files
    video_paths = check_video_files()
    if len(video_paths) < 4:
        print(f"❌ Cần 4 video, chỉ tìm thấy {len(video_paths)}")
        return
    
    # 2. Hiển thị thông tin tất cả video
    print(f"\n📊 Thông tin chi tiết video:")
    video_infos = get_all_video_info(video_paths)
    
    # 3. Tải model
    model = load_yolo_model()
    if model is None:
        return
    
    # 4. Chạy multi-view detection
    print(f"\n🚀 Bắt đầu xử lý multi-view...")
    success = run_multi_view_detection(model, video_paths, max_frames=100)
    
    if success:
        print("\n🎉 THÀNH CÔNG! Video multi-view đã được tạo.")
        print("💡 Mở file 'pickleball_multiview_final.mp4' để xem kết quả.")
    else:
        print("\n❌ Xử lý thất bại!")

if __name__ == "__main__":
    main()