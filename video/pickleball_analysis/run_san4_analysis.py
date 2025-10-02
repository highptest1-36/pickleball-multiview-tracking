import cv2
import numpy as np
import json
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time

class San4PickleballAnalysis:
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
        
        print(f"Video: {self.total_frames} frames at {self.fps} FPS")
        
        # YOLO setup
        from ultralytics import YOLO
        self.model = YOLO('yolov8x.pt')  # Use larger model for better accuracy
        
        # Tracking data
        self.player_tracks = defaultdict(lambda: deque(maxlen=300))  # Longer history
        self.ball_tracks = deque(maxlen=100)  # More ball history
        self.current_frame = 0
        self.player_positions_smooth = {}  # For smoothing
        
        # Setup visualization
        plt.ion()
        self.setup_display()
        
    def setup_display(self):
        """Setup the display windows"""
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Original video
        self.ax1.set_title('San4 - Original Video', fontsize=14, fontweight='bold')
        self.ax1.axis('off')
        
        # 2D Court view
        self.setup_court_view()
        
        # Player heatmap
        self.ax3.set_title('Player Movement Heatmap', fontsize=14, fontweight='bold')
        
        # Statistics
        self.ax4.set_title('Game Statistics', fontsize=14, fontweight='bold')
        self.ax4.axis('off')
        
        plt.tight_layout()
        
    def setup_court_view(self):
        """Setup 2D court visualization"""
        self.ax2.clear()
        self.ax2.set_xlim(-0.5, self.court_width + 0.5)
        self.ax2.set_ylim(-0.5, self.court_length + 0.5)
        self.ax2.set_aspect('equal')
        self.ax2.set_title('2D Court Tracking - San4', fontsize=14, fontweight='bold')
        
        # Court background
        court_rect = patches.Rectangle((0, 0), self.court_width, self.court_length,
                                     linewidth=3, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.4)
        self.ax2.add_patch(court_rect)
        
        # Net (vertical line)
        net_y = self.court_length / 2
        self.ax2.plot([0, self.court_width], [net_y, net_y], 'k-', linewidth=6, label='Net')
        
        # Service areas
        service_line1 = self.court_length / 4
        service_line2 = 3 * self.court_length / 4
        self.ax2.plot([0, self.court_width], [service_line1, service_line1], 'b--', linewidth=2, alpha=0.8)
        self.ax2.plot([0, self.court_width], [service_line2, service_line2], 'b--', linewidth=2, alpha=0.8)
        
        # Center line
        center_x = self.court_width / 2
        self.ax2.plot([center_x, center_x], [0, self.court_length], 'b--', linewidth=2, alpha=0.8)
        
        # Side labels
        self.ax2.text(self.court_width/4, -0.2, 'LEFT SIDE', ha='center', fontsize=12, fontweight='bold', color='blue')
        self.ax2.text(3*self.court_width/4, -0.2, 'RIGHT SIDE', ha='center', fontsize=12, fontweight='bold', color='red')
        
        self.ax2.set_xlabel('Width (m)')
        self.ax2.set_ylabel('Length (m)')
        self.ax2.grid(True, alpha=0.3)
        
    def transform_to_court(self, image_points):
        """Transform image coordinates to court coordinates"""
        if len(image_points) == 0:
            return []
        
        points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        court_points = cv2.perspectiveTransform(points, self.homography)
        return court_points.reshape(-1, 2)
    
    def process_frame(self, frame):
        """Process frame for tracking"""
        results = self.model.track(frame, persist=True, verbose=False, 
                                 conf=0.3, iou=0.5)  # Lower confidence, better tracking
        
        players = []
        balls = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xywh.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                # Handle cases with and without tracking IDs
                if result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy()
                else:
                    ids = [None] * len(boxes)
                
                for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
                    class_name = self.model.names[int(cls)]
                    
                    if class_name == 'person' and conf > 0.4:  # Lower threshold for persons
                        track_id = int(ids[i]) if ids[i] is not None else f"temp_{i}"
                        x, y, w, h = box
                        
                        players.append({
                            'id': track_id,
                            'pos': [x, y],
                            'conf': conf,
                            'bbox': [x-w/2, y-h/2, w, h]  # Add bbox for better tracking
                        })
                        
                    elif class_name in ['sports ball', 'ball'] and conf > 0.2:  # Very low threshold for balls
                        x, y, w, h = box
                        balls.append({
                            'pos': [x, y],
                            'conf': conf,
                            'size': max(w, h)  # Ball size for filtering
                        })
        
        return players, balls
    
    def is_on_court(self, court_pos):
        """Check if position is actually on the court (not just nearby)"""
        return (0 <= court_pos[0] <= self.court_width and 
                0 <= court_pos[1] <= self.court_length)
    
    def get_side(self, court_pos):
        """Determine which side of court (left=0, right=1)"""
        return 0 if court_pos[0] < self.court_width / 2 else 1
    
    def smooth_position(self, player_id, new_pos):
        """Smooth player position using exponential moving average"""
        alpha = 0.7  # Smoothing factor
        
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
    
    def filter_active_players(self, valid_players):
        """Filter to keep only top 2 players per side based on recent activity and consistency"""
        left_players = []
        right_players = []
        
        # Score players based on confidence and tracking history
        for player_id, court_pos, conf in valid_players:
            side = self.get_side(court_pos)
            
            # Bonus score for existing tracks (continuity)
            continuity_bonus = len(self.player_tracks[player_id]) * 0.01
            total_score = conf + continuity_bonus
            
            if side == 0:
                left_players.append((player_id, court_pos, total_score))
            else:
                right_players.append((player_id, court_pos, total_score))
        
        # Keep top 2 players per side (by total score)
        left_players.sort(key=lambda x: x[2], reverse=True)
        right_players.sort(key=lambda x: x[2], reverse=True)
        
        active_players = left_players[:2] + right_players[:2]
        return active_players

    def update_tracking(self, players, balls):
        """Update tracking data - only for players actually on court"""
        # Process players
        if players:
            player_positions = [p['pos'] for p in players]
            court_positions = self.transform_to_court(player_positions)
            
            # Filter players that are actually on court
            valid_players = []
            for player, court_pos in zip(players, court_positions):
                if self.is_on_court(court_pos):
                    valid_players.append((player['id'], court_pos, player['conf']))
            
            # Filter to max 2 players per side
            active_players = self.filter_active_players(valid_players)
            
            # Update tracking only for active players with smoothing
            current_active_ids = set()
            for player_id, court_pos, conf in active_players:
                current_active_ids.add(player_id)
                
                # Apply position smoothing
                smoothed_pos = self.smooth_position(player_id, court_pos)
                
                self.player_tracks[player_id].append({
                    'frame': self.current_frame,
                    'court_pos': smoothed_pos,
                    'confidence': conf
                })
            
            # Remove old inactive players (haven't been seen for 60 frames)
            inactive_threshold = 60
            players_to_remove = []
            for player_id in list(self.player_tracks.keys()):
                if player_id not in current_active_ids:
                    # Check if player hasn't been seen recently
                    if len(self.player_tracks[player_id]) == 0 or \
                       (self.current_frame - self.player_tracks[player_id][-1]['frame']) > inactive_threshold:
                        players_to_remove.append(player_id)
            
            for player_id in players_to_remove:
                del self.player_tracks[player_id]
        
        # Process balls with better filtering
        if balls:
            # Filter balls by size and confidence
            filtered_balls = []
            for ball in balls:
                # Ball should be small and reasonably confident
                if (ball['conf'] > 0.2 and 
                    'size' in ball and 
                    10 < ball['size'] < 100):  # Reasonable ball size
                    filtered_balls.append(ball)
            
            if filtered_balls:
                ball_positions = [b['pos'] for b in filtered_balls]
                court_positions = self.transform_to_court(ball_positions)
                
                for i, court_pos in enumerate(court_positions):
                    if self.is_on_court(court_pos):  # Only balls actually on court
                        self.ball_tracks.append({
                            'frame': self.current_frame,
                            'court_pos': court_pos,
                            'confidence': filtered_balls[i]['conf']
                        })
    
    def update_visualization(self, frame, players, balls):
        """Update all visualizations"""
        # Original video with detections
        self.ax1.clear()
        self.ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Draw detection boxes
        for player in players:
            x, y = player['pos']
            self.ax1.plot(x, y, 'ro', markersize=8)
            self.ax1.text(x+20, y-20, f"P{player['id']}", color='red', fontweight='bold')
        
        for ball in balls:
            x, y = ball['pos']
            self.ax1.plot(x, y, 'y*', markersize=12)
        
        self.ax1.set_title(f'San4 - Frame {self.current_frame}/{self.total_frames}')
        self.ax1.axis('off')
        
        # 2D Court view
        self.setup_court_view()
        
        # Current positions and trails with better visualization
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        for i, (player_id, track) in enumerate(self.player_tracks.items()):
            if len(track) > 0:
                color = colors[i % len(colors)]
                
                # Current position
                current_pos = track[-1]['court_pos']
                self.ax2.scatter(current_pos[0], current_pos[1], c=color, s=150,
                               marker='o', edgecolors='black', linewidth=3,
                               label=f'Player {player_id}', alpha=0.9)
                
                # Trail with fade effect (longer trail, more points)
                if len(track) > 1:
                    recent_track = list(track)[-60:]  # Last 60 positions for smoother trail
                    trail_x = [t['court_pos'][0] for t in recent_track]
                    trail_y = [t['court_pos'][1] for t in recent_track]
                    
                    # Draw trail with varying alpha for fade effect
                    for j in range(1, len(trail_x)):
                        alpha = (j / len(trail_x)) * 0.8  # Fade from 0 to 0.8
                        self.ax2.plot([trail_x[j-1], trail_x[j]], [trail_y[j-1], trail_y[j]], 
                                    color=color, alpha=alpha, linewidth=3)
        
        # Ball positions with better visualization
        if len(self.ball_tracks) > 0:
            recent_balls = list(self.ball_tracks)[-20:]  # Last 20 ball positions
            for i, ball in enumerate(recent_balls):
                pos = ball['court_pos']
                alpha = (i + 1) / len(recent_balls)  # Fade effect
                size = 80 + 40 * alpha  # Size increases for recent balls
                self.ax2.scatter(pos[0], pos[1], c='yellow', s=size, marker='*',
                               edgecolors='red', linewidth=2, alpha=alpha)
            
            # Draw ball trail
            if len(recent_balls) > 1:
                ball_x = [b['court_pos'][0] for b in recent_balls]
                ball_y = [b['court_pos'][1] for b in recent_balls]
                self.ax2.plot(ball_x, ball_y, 'y-', linewidth=3, alpha=0.7, label='Ball trail')
        
        # Court side assignment - only active players
        left_players = []
        right_players = []
        
        for player_id, track in self.player_tracks.items():
            if len(track) > 0:
                # Use most recent position to determine side
                recent_pos = track[-1]['court_pos']
                if self.get_side(recent_pos) == 0:
                    left_players.append(player_id)
                else:
                    right_players.append(player_id)
        
        # Ensure max 2 players per side for display
        left_players = left_players[:2]
        right_players = right_players[:2]
        
        # Statistics
        self.ax4.clear()
        self.ax4.axis('off')
        
        stats_text = f"""GAME STATISTICS - San4
        
Frame: {self.current_frame:,} / {self.total_frames:,}
Progress: {self.current_frame/self.total_frames*100:.1f}%

ACTIVE PLAYERS: {len(self.player_tracks)} (Max 4)

LEFT SIDE ({len(left_players)}/2 players):
{', '.join([f'P{p}' for p in left_players]) if left_players else 'None'}

RIGHT SIDE ({len(right_players)}/2 players):
{', '.join([f'P{p}' for p in right_players]) if right_players else 'None'}

BALL TRACKING:
Recent detections: {len(self.ball_tracks)}

COURT CALIBRATION:
Dimensions: {self.court_width:.1f}m × {self.court_length:.1f}m
Calibration points: {len(self.calibration['image_points'])}

FILTERING RULES:
✓ Only players ON court
✓ Max 2 players per side
✓ Remove inactive players
        """
        
        self.ax4.text(0.05, 0.95, stats_text, transform=self.ax4.transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Heatmap
        self.update_heatmap()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def update_heatmap(self):
        """Update player movement heatmap"""
        self.ax3.clear()
        
        # Collect all player positions - only from players on court
        all_positions = []
        for track in self.player_tracks.values():
            for entry in track:
                pos = entry['court_pos']
                if self.is_on_court(pos):
                    all_positions.append(pos)
        
        if len(all_positions) > 10:
            positions = np.array(all_positions)
            
            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1],
                                             bins=[20, 30],
                                             range=[[0, self.court_width], [0, self.court_length]])
            
            # Plot heatmap
            extent = [0, self.court_width, 0, self.court_length]
            self.ax3.imshow(H.T, extent=extent, origin='lower', cmap='hot', alpha=0.7)
            
            # Add court lines
            net_y = self.court_length / 2
            self.ax3.plot([0, self.court_width], [net_y, net_y], 'w-', linewidth=3)
            
        self.ax3.set_xlim(0, self.court_width)
        self.ax3.set_ylim(0, self.court_length)
        self.ax3.set_title('Player Movement Heatmap')
        self.ax3.set_xlabel('Width (m)')
        self.ax3.set_ylabel('Length (m)')
    
    def run(self):
        """Main execution loop"""
        print("🏓 Starting San4 Pickleball Analysis...")
        print("Press Ctrl+C to stop")
        
        try:
            skip_frames = 1  # Process every frame for smoother tracking
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video")
                    break
                
                self.current_frame += 1
                
                # Process every frame for smooth tracking
                players, balls = self.process_frame(frame)
                self.update_tracking(players, balls)
                
                # Update display every 2nd frame for performance
                if self.current_frame % 2 == 0:
                    self.update_visualization(frame, players, balls)
                
                # Progress report
                if self.current_frame % 150 == 0:
                    active_players = len(self.player_tracks)
                    recent_balls = len([b for b in self.ball_tracks if self.current_frame - b['frame'] < 30])
                    print(f"Frame {self.current_frame}, Active players: {active_players}, Recent balls: {recent_balls}")
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cap.release()
            plt.ioff()
            plt.show()
            print("🏁 Analysis complete!")

if __name__ == "__main__":
    analyzer = San4PickleballAnalysis()
    analyzer.run()