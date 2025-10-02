import cv2
import numpy as np
import json
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class FinalSan4Analysis:
    def __init__(self):
        # Load calibration
        with open('court_calibration_san4.json', 'r') as f:
            self.calibration = json.load(f)
        
        self.homography = np.array(self.calibration['homography'])
        self.court_width = self.calibration['court_width']
        self.court_length = self.calibration['court_length']
        
        # Video setup
        self.video_path = r"C:\Users\highp\pickerball\video\data_video\san4.mp4"
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Video: {self.total_frames} frames at {self.fps} FPS")
        
        # YOLO setup
        from ultralytics import YOLO
        self.model = YOLO('yolov8x.pt')
        
        # Tracking data
        self.player_tracks = defaultdict(lambda: deque(maxlen=400))
        self.ball_tracks = deque(maxlen=150)
        self.current_frame = 0
        self.player_positions_smooth = {}
        
        # Setup visualization
        plt.ion()
        self.setup_display()
        
    def setup_display(self):
        """Setup the display windows"""
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Original video
        self.ax1.set_title('🏓 San4 - Live Video Feed', fontsize=14, fontweight='bold')
        self.ax1.axis('off')
        
        # 2D Court view
        self.setup_court_view()
        
        # Player heatmap
        self.ax3.set_title('🔥 Player Movement Heatmap', fontsize=14, fontweight='bold')
        
        # Statistics
        self.ax4.set_title('📊 Live Game Statistics', fontsize=14, fontweight='bold')
        self.ax4.axis('off')
        
        plt.tight_layout()
        
    def setup_court_view(self):
        """Setup 2D court visualization"""
        self.ax2.clear()
        self.ax2.set_xlim(-0.3, self.court_width + 0.3)
        self.ax2.set_ylim(-0.3, self.court_length + 0.3)
        self.ax2.set_aspect('equal')
        self.ax2.set_title('🎾 Real-time Court Tracking', fontsize=14, fontweight='bold')
        
        # Court background
        court_rect = patches.Rectangle((0, 0), self.court_width, self.court_length,
                                     linewidth=4, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.5)
        self.ax2.add_patch(court_rect)
        
        # Net (vertical line)
        net_y = self.court_length / 2
        self.ax2.plot([0, self.court_width], [net_y, net_y], 'k-', linewidth=8, label='Net')
        
        # Service areas
        service_line1 = self.court_length / 4
        service_line2 = 3 * self.court_length / 4
        self.ax2.plot([0, self.court_width], [service_line1, service_line1], 'b--', linewidth=3, alpha=0.8)
        self.ax2.plot([0, self.court_width], [service_line2, service_line2], 'b--', linewidth=3, alpha=0.8)
        
        # Center line
        center_x = self.court_width / 2
        self.ax2.plot([center_x, center_x], [0, self.court_length], 'b--', linewidth=3, alpha=0.8)
        
        # Side labels
        self.ax2.text(self.court_width/4, -0.15, 'LEFT SIDE', ha='center', fontsize=14, fontweight='bold', color='blue')
        self.ax2.text(3*self.court_width/4, -0.15, 'RIGHT SIDE', ha='center', fontsize=14, fontweight='bold', color='red')
        
        self.ax2.set_xlabel('Width (m)', fontsize=12)
        self.ax2.set_ylabel('Length (m)', fontsize=12)
        self.ax2.grid(True, alpha=0.4)
        
    def is_on_court(self, court_pos):
        """Check if position is actually on the court"""
        return (0 <= court_pos[0] <= self.court_width and 
                0 <= court_pos[1] <= self.court_length)
    
    def get_side(self, court_pos):
        """Determine which side of court (left=0, right=1)"""
        return 0 if court_pos[0] < self.court_width / 2 else 1
    
    def transform_to_court(self, image_points):
        """Transform image coordinates to court coordinates"""
        if len(image_points) == 0:
            return []
        
        points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        court_points = cv2.perspectiveTransform(points, self.homography)
        return court_points.reshape(-1, 2)
    
    def smooth_position(self, player_id, new_pos):
        """Smooth player position using exponential moving average"""
        alpha = 0.6  # Smoothing factor
        
        if player_id in self.player_positions_smooth:
            old_pos = self.player_positions_smooth[player_id]
            smoothed_pos = [
                alpha * new_pos[0] + (1 - alpha) * old_pos[0],
                alpha * new_pos[1] + (1 - alpha) * old_pos[1]
            ]
        else:
            smoothed_pos = new_pos
        
        self.player_positions_smooth[player_id] = smoothed_pos
        return smoothed_pos
    
    def process_frame(self, frame):
        """Process frame for tracking"""
        results = self.model.track(frame, persist=True, verbose=False, 
                                 conf=0.25, iou=0.6)
        
        players = []
        balls = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                # Handle tracking IDs
                if result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy()
                else:
                    ids = [None] * len(boxes)
                
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
                    class_name = self.model.names[int(cls)]
                    
                    if class_name == 'person' and conf > 0.4:
                        track_id = int(ids[i]) if ids[i] is not None else f"temp_{i}_{self.current_frame}"
                        x, y, w, h = box
                        
                        players.append({
                            'id': track_id,
                            'pos': [x, y],
                            'conf': conf
                        })
                        
                    elif class_name in ['sports ball', 'ball'] and conf > 0.15:
                        x, y, w, h = box
                        ball_size = max(w, h)
                        
                        # Filter reasonable ball sizes
                        if 10 < ball_size < 60:
                            balls.append({
                                'pos': [x, y],
                                'conf': conf,
                                'size': ball_size
                            })
        
        return players, balls
    
    def filter_active_players(self, valid_players):
        """Filter to keep max 2 players per side"""
        left_players = []
        right_players = []
        
        for player_id, court_pos, conf in valid_players:
            side = self.get_side(court_pos)
            
            # Score based on confidence and tracking history
            track_bonus = min(len(self.player_tracks[player_id]) * 0.005, 0.2)
            total_score = conf + track_bonus
            
            if side == 0:
                left_players.append((player_id, court_pos, total_score))
            else:
                right_players.append((player_id, court_pos, total_score))
        
        # Keep top 2 per side
        left_players.sort(key=lambda x: x[2], reverse=True)
        right_players.sort(key=lambda x: x[2], reverse=True)
        
        return left_players[:2] + right_players[:2]
    
    def update_tracking(self, players, balls):
        """Update tracking data"""
        # Process players
        if players:
            player_positions = [p['pos'] for p in players]
            court_positions = self.transform_to_court(player_positions)
            
            # Filter players on court
            valid_players = []
            for player, court_pos in zip(players, court_positions):
                if self.is_on_court(court_pos):
                    valid_players.append((player['id'], court_pos, player['conf']))
            
            # Keep max 2 players per side
            active_players = self.filter_active_players(valid_players)
            
            # Update tracking with smoothing
            current_active_ids = set()
            for player_id, court_pos, conf in active_players:
                current_active_ids.add(player_id)
                smoothed_pos = self.smooth_position(player_id, court_pos)
                
                self.player_tracks[player_id].append({
                    'frame': self.current_frame,
                    'court_pos': smoothed_pos,
                    'confidence': conf
                })
            
            # Clean up inactive players
            inactive_threshold = 90
            players_to_remove = []
            for player_id in list(self.player_tracks.keys()):
                if (player_id not in current_active_ids and 
                    (len(self.player_tracks[player_id]) == 0 or 
                     self.current_frame - self.player_tracks[player_id][-1]['frame'] > inactive_threshold)):
                    players_to_remove.append(player_id)
            
            for player_id in players_to_remove:
                del self.player_tracks[player_id]
                if player_id in self.player_positions_smooth:
                    del self.player_positions_smooth[player_id]
        
        # Process balls
        if balls:
            ball_positions = [b['pos'] for b in balls]
            court_positions = self.transform_to_court(ball_positions)
            
            for i, court_pos in enumerate(court_positions):
                if self.is_on_court(court_pos):
                    self.ball_tracks.append({
                        'frame': self.current_frame,
                        'court_pos': court_pos,
                        'confidence': balls[i]['conf']
                    })
    
    def update_visualization(self, frame, players, balls):
        """Update all visualizations"""
        # Original video with detections
        self.ax1.clear()
        self.ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Draw detection boxes
        for player in players:
            x, y = player['pos']
            self.ax1.plot(x, y, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
            self.ax1.text(x+25, y-25, f"P{player['id']}", color='red', fontweight='bold', fontsize=10)
        
        for ball in balls:
            x, y = ball['pos']
            self.ax1.plot(x, y, 'y*', markersize=15, markeredgecolor='red', markeredgewidth=2)
        
        self.ax1.set_title(f'🏓 San4 - Frame {self.current_frame:,}/{self.total_frames:,} ({self.current_frame/self.total_frames*100:.1f}%)')
        self.ax1.axis('off')
        
        # 2D Court view
        self.setup_court_view()
        
        # Player visualization with improved graphics
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'darkred']
        
        for i, (player_id, track) in enumerate(self.player_tracks.items()):
            if len(track) > 0:
                color = colors[i % len(colors)]
                
                # Current position
                current_pos = track[-1]['court_pos']
                self.ax2.scatter(current_pos[0], current_pos[1], c=color, s=200,
                               marker='o', edgecolors='white', linewidth=4,
                               label=f'Player {player_id}', alpha=0.9, zorder=10)
                
                # Trail with fade effect
                if len(track) > 5:
                    recent_track = list(track)[-80:]  # Last 80 positions
                    trail_x = [t['court_pos'][0] for t in recent_track]
                    trail_y = [t['court_pos'][1] for t in recent_track]
                    
                    # Draw fading trail
                    for j in range(1, len(trail_x)):
                        alpha = (j / len(trail_x)) ** 0.5 * 0.8
                        width = 1 + alpha * 3
                        self.ax2.plot([trail_x[j-1], trail_x[j]], [trail_y[j-1], trail_y[j]], 
                                    color=color, alpha=alpha, linewidth=width, zorder=5)
        
        # Ball visualization
        if len(self.ball_tracks) > 0:
            recent_balls = [b for b in self.ball_tracks if self.current_frame - b['frame'] < 60]
            
            if recent_balls:
                # Ball trail
                if len(recent_balls) > 2:
                    ball_x = [b['court_pos'][0] for b in recent_balls]
                    ball_y = [b['court_pos'][1] for b in recent_balls]
                    self.ax2.plot(ball_x, ball_y, 'yellow', linewidth=4, alpha=0.8, zorder=8)
                
                # Recent ball positions
                for i, ball in enumerate(recent_balls[-10:]):
                    pos = ball['court_pos']
                    age = len(recent_balls) - i
                    alpha = max(0.3, 1.0 - age * 0.1)
                    size = 60 + 80 * alpha
                    self.ax2.scatter(pos[0], pos[1], c='yellow', s=size, marker='*',
                                   edgecolors='red', linewidth=3, alpha=alpha, zorder=9)
        
        # Side assignment
        left_players = []
        right_players = []
        
        for player_id, track in self.player_tracks.items():
            if len(track) > 0:
                recent_pos = track[-1]['court_pos']
                if self.get_side(recent_pos) == 0:
                    left_players.append(player_id)
                else:
                    right_players.append(player_id)
        
        # Statistics
        self.ax4.clear()
        self.ax4.axis('off')
        
        recent_ball_count = len([b for b in self.ball_tracks if self.current_frame - b['frame'] < 30])
        
        stats_text = f"""📊 LIVE GAME ANALYSIS - San4

🎬 Video Progress: {self.current_frame:,} / {self.total_frames:,} frames
📈 Completion: {self.current_frame/self.total_frames*100:.1f}%
⏱️  Time: {self.current_frame/self.fps:.1f}s / {self.total_frames/self.fps:.1f}s

👥 ACTIVE PLAYERS: {len(self.player_tracks)}/4

🔵 LEFT SIDE ({len(left_players)}/2):
   {', '.join([f'Player {p}' for p in left_players]) if left_players else '• No players'}

🔴 RIGHT SIDE ({len(right_players)}/2):
   {', '.join([f'Player {p}' for p in right_players]) if right_players else '• No players'}

🎾 BALL TRACKING:
   Total detections: {len(self.ball_tracks)}
   Recent (30 frames): {recent_ball_count}

⚙️  COURT SETUP:
   Dimensions: {self.court_width:.1f}m × {self.court_length:.1f}m
   Calibration: {len(self.calibration['image_points'])} points

🔧 SYSTEM STATUS:
   ✅ Only on-court players tracked
   ✅ Max 2 players per side
   ✅ Position smoothing active
   ✅ Automatic cleanup enabled
        """
        
        self.ax4.text(0.02, 0.98, stats_text, transform=self.ax4.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue', alpha=0.9))
        
        # Heatmap
        self.update_heatmap()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def update_heatmap(self):
        """Update player movement heatmap"""
        self.ax3.clear()
        
        # Collect positions from last 1000 frames
        all_positions = []
        for track in self.player_tracks.values():
            for entry in track:
                if self.current_frame - entry['frame'] < 1000:  # Recent positions only
                    pos = entry['court_pos']
                    if self.is_on_court(pos):
                        all_positions.append(pos)
        
        if len(all_positions) > 20:
            positions = np.array(all_positions)
            
            # Create high-resolution heatmap
            H, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1],
                                             bins=[30, 40],
                                             range=[[0, self.court_width], [0, self.court_length]])
            
            # Plot heatmap
            extent = [0, self.court_width, 0, self.court_length]
            self.ax3.imshow(H.T, extent=extent, origin='lower', cmap='hot', alpha=0.8)
            
            # Add court lines on heatmap
            net_y = self.court_length / 2
            self.ax3.plot([0, self.court_width], [net_y, net_y], 'cyan', linewidth=4, alpha=0.9)
            center_x = self.court_width / 2
            self.ax3.plot([center_x, center_x], [0, self.court_length], 'cyan', linewidth=2, alpha=0.7)
        
        self.ax3.set_xlim(0, self.court_width)
        self.ax3.set_ylim(0, self.court_length)
        self.ax3.set_title('🔥 Movement Heatmap (Recent 1000 frames)')
        self.ax3.set_xlabel('Width (m)')
        self.ax3.set_ylabel('Length (m)')
    
    def run(self):
        """Main execution loop"""
        print("🚀 Starting Advanced San4 Pickleball Analysis...")
        print("🔥 Features: Smooth tracking, ball detection, 4-player limit, heatmaps")
        print("⏹️  Press Ctrl+C to stop")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("🏁 End of video reached")
                    break
                
                self.current_frame += 1
                
                # Process every frame for smooth tracking
                players, balls = self.process_frame(frame)
                self.update_tracking(players, balls)
                
                # Update display every 2nd frame
                if self.current_frame % 2 == 0:
                    self.update_visualization(frame, players, balls)
                
                # Progress report
                if self.current_frame % 200 == 0:
                    active_players = len(self.player_tracks)
                    recent_balls = len([b for b in self.ball_tracks if self.current_frame - b['frame'] < 30])
                    print(f"⚡ Frame {self.current_frame:,} | Players: {active_players} | Recent balls: {recent_balls}")
                
        except KeyboardInterrupt:
            print("\n⏹️  Analysis stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cap.release()
            plt.ioff()
            print("🎉 Analysis complete! Window will remain open.")
            plt.show()  # Keep window open

if __name__ == "__main__":
    analyzer = FinalSan4Analysis()
    analyzer.run()