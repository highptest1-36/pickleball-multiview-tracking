#!/usr/bin/env python3
"""
360-Degree Multi-View Video Stitching System
- Ghép 4 góc camera thành 1 view 360°
- Video gốc không có visualization 3D
- Smooth frame blending và synchronized playback
"""

import cv2
import numpy as np
import os
import time

class MultiViewVideoStitcher:
    def __init__(self):
        print("🎥 SYNCHRONIZED MULTI-VIEW VIDEO STITCHER")
        print("=" * 50)
        
        # Video paths (4 camera angles)
        self.video_paths = [
            "e4e66c2058ff-0.0.0.0-3000-1-0-mzle9eCKS2oQvLJa7rOE000000/h20250926093017-20250926093526m.mp4",
            "e4e66c2058ff-0.0.0.0-3000-2-0-vvkoKtKIUN7KS72O4bfR000000/h20250926093019-20250926093526m.mp4", 
            "e4e66c2058ff-0.0.0.0-3000-3-0-a4TtYafdNkjZQjVO5hll000000/h20250926093021-20250926093526m.mp4",
            "e4e66c2058ff-0.0.0.0-3000-4-0-ZhV2hb2DFg8xhbXYcpWn000000/h20250926093022-20250926093526m.mp4"
        ]
        
        # Reference camera (Camera 1 as master)
        self.reference_camera = 0
        self.frame_offsets = [0, 0, 0, 0]  # Frame offsets for sync
        self.sync_detected = False
        
        # Verify and load videos
        self.caps = []
        self.frame_counts = []
        
        for i, path in enumerate(self.video_paths):
            if os.path.exists(path):
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    self.caps.append(cap)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.frame_counts.append(frame_count)
                    print(f"✅ Camera {i+1}: {os.path.basename(path)} ({frame_count} frames)")
                else:
                    print(f"❌ Camera {i+1}: Cannot open video")
                    return
            else:
                print(f"❌ Camera {i+1}: File not found")
                return
        
        if len(self.caps) != 4:
            print(f"❌ Error: Need exactly 4 cameras, found {len(self.caps)}")
            return
            
        # Get video properties
        self.fps = self.caps[0].get(cv2.CAP_PROP_FPS)
        self.width = int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.min_frames = min(self.frame_counts)
        
        print(f"📹 Video specs: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        print(f"📊 Frames to process: {self.min_frames}")
        
        # Setup synchronization detection
        self.setup_sync_detection()
        
        # Setup output dimensions for different layouts
        self.setup_layouts()
        
    def setup_sync_detection(self):
        """Setup synchronization detection using corner timestamps or common reference points"""
        print("\n🔍 Analyzing video synchronization...")
        
        # Try to detect sync markers in first few frames
        self.detect_sync_markers()
        
    def detect_sync_markers(self, check_frames=30):
        """Detect synchronization markers (timestamps, common events) in videos"""
        print(f"🔍 Checking first {check_frames} frames for sync markers...")
        
        # Store original positions
        original_positions = [cap.get(cv2.CAP_PROP_POS_FRAMES) for cap in self.caps]
        
        # Reset all videos to start
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Look for timestamp digits in corner of frames
        timestamp_patterns = []
        
        for frame_idx in range(check_frames):
            frame_timestamps = []
            
            for cam_idx, cap in enumerate(self.caps):
                ret, frame = cap.read()
                if ret:
                    # Extract timestamp region (top-right corner where timestamps usually appear)
                    h, w = frame.shape[:2]
                    timestamp_region = frame[0:60, w-200:w]  # Top-right corner
                    
                    # Convert to grayscale for digit detection
                    gray = cv2.cvtColor(timestamp_region, cv2.COLOR_BGR2GRAY)
                    
                    # Use simple template matching or OCR approach
                    # For now, use pixel intensity patterns as fingerprint
                    fingerprint = self.create_frame_fingerprint(gray)
                    frame_timestamps.append(fingerprint)
                else:
                    frame_timestamps.append(None)
            
            timestamp_patterns.append(frame_timestamps)
        
        # Analyze patterns to find best alignment
        self.calculate_frame_offsets(timestamp_patterns)
        
        # Reset to original positions
        for i, cap in enumerate(self.caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_positions[i])
        
        print(f"✅ Synchronization analysis complete")
        print(f"📊 Frame offsets: {self.frame_offsets}")
    
    def create_frame_fingerprint(self, gray_region):
        """Create a unique fingerprint for frame region (for timestamp matching)"""
        # Resize to standard size
        resized = cv2.resize(gray_region, (50, 15))
        
        # Calculate histogram
        hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
        
        # Use mean, std, and some histogram bins as fingerprint
        fingerprint = {
            'mean': np.mean(resized),
            'std': np.std(resized),
            'hist_peaks': np.argsort(hist.flatten())[-5:].tolist()  # Top 5 histogram bins
        }
        
        return fingerprint
    
    def calculate_frame_offsets(self, timestamp_patterns):
        """Calculate frame offsets to synchronize videos"""
        if not timestamp_patterns:
            print("⚠️ No timestamp patterns found, using default sync")
            return
        
        # Use reference camera (camera 0) as master
        ref_patterns = [pattern[self.reference_camera] for pattern in timestamp_patterns if pattern[self.reference_camera] is not None]
        
        if not ref_patterns:
            print("⚠️ No valid reference patterns, using default sync")
            return
        
        # For each other camera, find best matching offset
        for cam_idx in range(len(self.caps)):
            if cam_idx == self.reference_camera:
                self.frame_offsets[cam_idx] = 0
                continue
            
            cam_patterns = [pattern[cam_idx] for pattern in timestamp_patterns if pattern[cam_idx] is not None]
            
            if not cam_patterns:
                continue
            
            # Find best correlation offset
            best_offset = 0
            best_correlation = -1
            
            # Try different offsets
            for offset in range(-10, 11):  # Check +/- 10 frames
                correlation = self.calculate_pattern_correlation(ref_patterns, cam_patterns, offset)
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_offset = offset
            
            self.frame_offsets[cam_idx] = best_offset
            print(f"📹 Camera {cam_idx+1} offset: {best_offset} frames (correlation: {best_correlation:.3f})")
        
        self.sync_detected = True
    
    def calculate_pattern_correlation(self, ref_patterns, cam_patterns, offset):
        """Calculate correlation between reference and camera patterns with offset"""
        if not ref_patterns or not cam_patterns:
            return 0
        
        correlations = []
        
        # Compare overlapping regions
        for i in range(max(0, -offset), min(len(ref_patterns), len(cam_patterns) - offset)):
            cam_idx = i + offset
            if 0 <= cam_idx < len(cam_patterns):
                ref_fp = ref_patterns[i]
                cam_fp = cam_patterns[cam_idx]
                
                # Simple correlation based on fingerprint similarity
                if ref_fp and cam_fp:
                    mean_diff = abs(ref_fp['mean'] - cam_fp['mean'])
                    std_diff = abs(ref_fp['std'] - cam_fp['std'])
                    
                    # Normalize correlation (lower difference = higher correlation)
                    correlation = 1.0 / (1.0 + mean_diff + std_diff)
                    correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0
        
    def setup_layouts(self):
        """Setup different stitching layouts"""
        # Standard frame size for processing
        self.frame_w, self.frame_h = 480, 270  # 16:9 aspect ratio
        
        # Layout 1: 2x2 Grid
        self.grid_layout = {
            'width': self.frame_w * 2,
            'height': self.frame_h * 2,
            'positions': [
                (0, 0),                           # Camera 1: Top-left
                (self.frame_w, 0),                # Camera 2: Top-right  
                (0, self.frame_h),                # Camera 3: Bottom-left
                (self.frame_w, self.frame_h)      # Camera 4: Bottom-right
            ]
        }
        
        # Layout 2: Horizontal strip
        self.strip_layout = {
            'width': self.frame_w * 4,
            'height': self.frame_h,
            'positions': [
                (0, 0),                           # Camera 1
                (self.frame_w, 0),                # Camera 2
                (self.frame_w * 2, 0),            # Camera 3
                (self.frame_w * 3, 0)             # Camera 4
            ]
        }
        
        # Layout 3: Picture-in-Picture with main view
        self.pip_layout = {
            'width': self.frame_w * 2,
            'height': int(self.frame_h * 1.5),
            'main_size': (self.frame_w * 2, self.frame_h),
            'pip_size': (self.frame_w // 2, self.frame_h // 2),
            'positions': [
                (0, 0, 'main'),                           # Camera 1: Main view
                (0, self.frame_h, 'pip'),                 # Camera 2: PiP bottom-left
                (self.frame_w // 2, self.frame_h, 'pip'), # Camera 3: PiP bottom-center  
                (self.frame_w, self.frame_h, 'pip')       # Camera 4: PiP bottom-right
            ]
        }
    
    def create_2x2_grid(self, frames):
        """Create 2x2 grid layout"""
        if len(frames) != 4:
            return None
            
        # Resize all frames
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (self.frame_w, self.frame_h))
            resized_frames.append(resized)
        
        # Create 2x2 grid
        top_row = np.hstack([resized_frames[0], resized_frames[1]])
        bottom_row = np.hstack([resized_frames[2], resized_frames[3]])
        grid = np.vstack([top_row, bottom_row])
        
        # Add camera labels
        self.add_camera_labels(grid, self.grid_layout)
        
        return grid
    
    def create_horizontal_strip(self, frames):
        """Create horizontal strip layout"""
        if len(frames) != 4:
            return None
            
        # Resize all frames
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (self.frame_w, self.frame_h))
            resized_frames.append(resized)
        
        # Create horizontal strip
        strip = np.hstack(resized_frames)
        
        # Add camera labels
        for i, pos in enumerate(self.strip_layout['positions']):
            x, y = pos
            cv2.putText(strip, f'Cam {i+1}', (x + 10, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(strip, f'Cam {i+1}', (x + 10, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        return strip
    
    def create_picture_in_picture(self, frames, main_cam_idx=0):
        """Create Picture-in-Picture layout with rotating main camera"""
        if len(frames) != 4:
            return None
            
        # Create canvas
        canvas = np.zeros((self.pip_layout['height'], self.pip_layout['width'], 3), dtype=np.uint8)
        
        # Main camera (larger view)
        main_frame = cv2.resize(frames[main_cam_idx], self.pip_layout['main_size'])
        canvas[0:self.frame_h, 0:self.frame_w*2] = main_frame
        
        # Add main camera label
        cv2.putText(canvas, f'MAIN - Camera {main_cam_idx+1}', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(canvas, f'MAIN - Camera {main_cam_idx+1}', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 1)
        
        # Small PiP cameras
        pip_idx = 0
        for i in range(4):
            if i != main_cam_idx:
                pip_frame = cv2.resize(frames[i], self.pip_layout['pip_size'])
                x = pip_idx * (self.frame_w // 2)
                y = self.frame_h
                canvas[y:y + self.frame_h//2, x:x + self.frame_w//2] = pip_frame
                
                # Add PiP labels
                cv2.putText(canvas, f'Cam {i+1}', (x + 5, y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(canvas, f'Cam {i+1}', (x + 5, y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                pip_idx += 1
        
        return canvas
    
    def create_360_rotation(self, frames, frame_idx):
        """Create seamless multi-view with reference camera as anchor"""
        if len(frames) != 4:
            return None
        
        # Use reference camera as main view with smooth switching
        # Switch main camera every 2 seconds but with instant transitions
        switch_interval = int(self.fps * 2)  # 2 seconds per camera
        main_cam_idx = (frame_idx // switch_interval) % 4
        
        # Create canvas
        canvas = np.zeros((int(self.frame_h * 1.5), self.frame_w * 2, 3), dtype=np.uint8)
        
        # Main camera view (2x size)
        main_frame = cv2.resize(frames[main_cam_idx], (self.frame_w * 2, self.frame_h))
        canvas[0:self.frame_h, :] = main_frame
        
        # Add main camera indicator with sync status
        sync_status = "SYNCED" if self.sync_detected else "DEFAULT"
        main_label = f'MAIN: Camera {main_cam_idx+1} ({sync_status})'
        cv2.putText(canvas, main_label, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(canvas, main_label, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        
        # Show frame offsets for debugging
        offset_text = f'Offsets: {self.frame_offsets}'
        cv2.putText(canvas, offset_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Small views for other cameras (3 remaining cameras)
        pip_width = self.frame_w // 2
        pip_height = self.frame_h // 2
        pip_y = self.frame_h
        
        pip_idx = 0
        for cam_idx in range(4):
            if cam_idx != main_cam_idx:
                pip_frame = cv2.resize(frames[cam_idx], (pip_width, pip_height))
                pip_x = pip_idx * pip_width
                
                # Place PiP frame
                canvas[pip_y:pip_y + pip_height, pip_x:pip_x + pip_width] = pip_frame
                
                # Add PiP labels with sync info
                offset_info = f'+{self.frame_offsets[cam_idx]}f' if self.frame_offsets[cam_idx] >= 0 else f'{self.frame_offsets[cam_idx]}f'
                pip_label = f'Cam {cam_idx+1} ({offset_info})'
                cv2.putText(canvas, pip_label, (pip_x + 5, pip_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(canvas, pip_label, (pip_x + 5, pip_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                pip_idx += 1
        
        return canvas
    
    def add_camera_labels(self, frame, layout):
        """Add camera labels to frame"""
        for i, pos in enumerate(layout['positions']):
            x, y = pos
            # White text with black outline
            cv2.putText(frame, f'Camera {i+1}', (x + 10, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f'Camera {i+1}', (x + 10, y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    def read_synchronized_frames(self):
        """Read frames from all cameras with proper synchronization"""
        frames = []
        
        for cam_idx, cap in enumerate(self.caps):
            # Apply frame offset for synchronization
            if self.sync_detected and self.frame_offsets[cam_idx] != 0:
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                target_pos = current_pos + self.frame_offsets[cam_idx]
                
                # Ensure we don't go out of bounds
                max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                target_pos = max(0, min(target_pos, max_frames - 1))
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
            
            ret, frame = cap.read()
            if not ret:
                return None
            frames.append(frame)
        
        return frames
    
    def stitch_videos(self, layout='360', output_filename='multiview_stitched.mp4', max_frames=300):
        """Main stitching function"""
        print(f"\n🚀 Starting video stitching with '{layout}' layout")
        print(f"📺 Processing up to {max_frames} frames")
        
        # Determine output dimensions based on layout
        if layout == 'grid':
            out_width, out_height = self.grid_layout['width'], self.grid_layout['height']
        elif layout == 'strip':
            out_width, out_height = self.strip_layout['width'], self.strip_layout['height']
        elif layout in ['pip', '360']:
            out_width, out_height = self.pip_layout['width'], self.pip_layout['height']
        else:
            print(f"❌ Unknown layout: {layout}")
            return
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, self.fps, (out_width, out_height))
        
        if not out.isOpened():
            print("❌ Error: Cannot create output video")
            return
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while frame_idx < min(max_frames, self.min_frames):
                # Read synchronized frames
                frames = self.read_synchronized_frames()
                if frames is None:
                    print("📹 End of video reached")
                    break
                
                # Create stitched frame based on layout
                if layout == 'grid':
                    stitched = self.create_2x2_grid(frames)
                elif layout == 'strip':
                    stitched = self.create_horizontal_strip(frames)
                elif layout == 'pip':
                    # Rotate main camera every 2 seconds
                    main_cam = (frame_idx // int(self.fps * 2)) % 4
                    stitched = self.create_picture_in_picture(frames, main_cam)
                elif layout == '360':
                    stitched = self.create_360_rotation(frames, frame_idx)
                
                if stitched is not None:
                    # Add frame counter
                    cv2.putText(stitched, f'Frame: {frame_idx}', (out_width - 150, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(stitched, f'Frame: {frame_idx}', (out_width - 150, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    
                    # Write frame
                    out.write(stitched)
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_idx / elapsed if elapsed > 0 else 0
                    progress = (frame_idx / min(max_frames, self.min_frames)) * 100
                    print(f"Frame {frame_idx}: {fps:.1f} FPS | Progress: {progress:.1f}%")
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        # Cleanup
        out.release()
        for cap in self.caps:
            cap.release()
        
        # Final stats
        elapsed = time.time() - start_time
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        
        print(f"\n✅ Video stitching completed!")
        print(f"📊 Final Stats:")
        print(f"   - Layout: {layout}")
        print(f"   - Frames processed: {frame_idx}")
        print(f"   - Output resolution: {out_width}x{out_height}")
        print(f"   - Processing time: {elapsed:.1f}s") 
        print(f"   - Average FPS: {avg_fps:.1f}")
        print(f"   - Output file: {output_filename}")
        
        return True

    def create_all_layouts(self, max_frames=200):
        """Create videos with all different layouts"""
        layouts = [
            ('grid', 'grid_view.mp4', '2x2 Grid Layout'),
            ('strip', 'strip_view.mp4', 'Horizontal Strip Layout'), 
            ('pip', 'pip_view.mp4', 'Picture-in-Picture Layout'),
            ('360', '360_view.mp4', '360-Degree Rotating View')
        ]
        
        for layout, filename, description in layouts:
            print(f"\n🎬 Creating {description}")
            self.stitch_videos(layout, filename, max_frames)
            
            # Reset video captures
            for i, cap in enumerate(self.caps):
                cap.release()
                self.caps[i] = cv2.VideoCapture(self.video_paths[i])


if __name__ == "__main__":
    stitcher = MultiViewVideoStitcher()
    
    # Create synchronized multi-view with frame matching
    print("\n🎯 Creating Synchronized Multi-View Video")
    print("📍 Using Camera 1 as reference anchor")
    stitcher.stitch_videos(layout='360', output_filename='synced_multiview.mp4', max_frames=300)