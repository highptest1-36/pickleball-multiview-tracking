"""
Simple 2D Court Viewer - Hiển thị real-time court view với tracking data

Tool đơn giản để xem players di chuyển trên sân 2D.
"""

import cv2
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque

class Simple2DCourtViewer:
    """Simple 2D Court viewer với court calibration và zone logic."""
    
    def __init__(self, court_config_path: str = "config/court_points.json"):
        """Initialize simple 2D court viewer với court calibration."""
        # Load court calibration
        self.homography_matrix = None
        self.court_bounds = None
        self.load_court_calibration(court_config_path)
        
        # Court dimensions (bird's-eye view) 
        self.court_width = 1000   # Simpler size
        self.court_height = 500   
        self.padding = 50
        
        # Canvas dimensions
        self.canvas_width = self.court_width + 2 * self.padding
        self.canvas_height = self.court_height + 2 * self.padding
        
        # Court coordinates
        self.court_x = self.padding
        self.court_y = self.padding
        
        # Court zones (chia đôi sân theo chiều RỘNG - vertical split)
        self.net_x = self.court_x + self.court_width // 2  # Net ở giữa theo chiều rộng
        self.left_side = {
            'x1': self.court_x,
            'y1': self.court_y, 
            'x2': self.net_x,
            'y2': self.court_y + self.court_height
        }
        self.right_side = {
            'x1': self.net_x,
            'y1': self.court_y,
            'x2': self.court_x + self.court_width,
            'y2': self.court_y + self.court_height
        }
        
        # Colors
        self.court_color = (34, 139, 34)  # Forest green
        self.line_color = (255, 255, 255)  # White
        
        # Player colors (up to 6 players)
        self.player_colors = [
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue  
            (0, 255, 0),    # Green
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 255, 0),  # Yellow
        ]
        
        self.ball_color = (255, 255, 255)  # White
        
        # Tracking data với zone management
        self.player_positions = defaultdict(deque)  # player_id -> deque of positions
        self.ball_positions = deque(maxlen=30)      # Ball trail
        self.heatmap_points = defaultdict(list)     # Simple heatmap points
        
        # Zone management (mỗi bên sân tối đa 2 người)
        self.left_side_players = set()   # Player IDs ở bên trái
        self.right_side_players = set()  # Player IDs ở bên phải
        self.max_players_per_side = 2
        
        # Display settings
        self.trail_length = 20
        self.point_size = 8
        
        print("🏓 Simple 2D Court Viewer initialized with court calibration")

    def load_court_calibration(self, config_path: str):
        """
        Load court calibration data từ config file.
        
        Args:
            config_path: Đường dẫn tới court config file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Get san1 calibration (đang dùng san1)
            san1_config = config['cameras']['san1']
            
            if san1_config['calibration_status'] == 'calibrated':
                # Load homography matrix
                self.homography_matrix = np.array(san1_config['homography_matrix'], dtype=np.float32)
                
                # Get court corners để tạo bounds
                corners = san1_config['detailed_points']
                all_points = list(corners.values())
                
                # Create bounding polygon từ court corners
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                
                self.court_bounds = {
                    'min_x': min(x_coords),
                    'max_x': max(x_coords),
                    'min_y': min(y_coords), 
                    'max_y': max(y_coords)
                }
                
                print(f"✅ Loaded court calibration for san1")
                print(f"   Court bounds: {self.court_bounds}")
            else:
                print("⚠️ san1 chưa được calibrated, dùng simple transformation")
                
        except Exception as e:
            print(f"⚠️ Error loading court calibration: {e}")
            print("   Sẽ dùng simple transformation")

    def is_point_in_court(self, x: float, y: float) -> bool:
        """
        Kiểm tra xem điểm có nằm trong sân không.
        
        Args:
            x, y: Tọa độ trong camera view
            
        Returns:
            True nếu point trong sân
        """
        if self.court_bounds is None:
            return True  # Nếu không có calibration, accept tất cả
        
        # Relaxed bounding box check (expand bounds by 20%)
        margin_x = (self.court_bounds['max_x'] - self.court_bounds['min_x']) * 0.2
        margin_y = (self.court_bounds['max_y'] - self.court_bounds['min_y']) * 0.2
        
        expanded_bounds = {
            'min_x': self.court_bounds['min_x'] - margin_x,
            'max_x': self.court_bounds['max_x'] + margin_x,
            'min_y': self.court_bounds['min_y'] - margin_y,
            'max_y': self.court_bounds['max_y'] + margin_y
        }
        
        return (expanded_bounds['min_x'] <= x <= expanded_bounds['max_x'] and
                expanded_bounds['min_y'] <= y <= expanded_bounds['max_y'])

    def transform_point_calibrated(self, x: float, y: float) -> Tuple[int, int]:
        """
        Transform point với homography matrix - FIX để mapping chính xác.
        
        Args:
            x, y: Tọa độ trong camera view
            
        Returns:
            (x, y) trong court coordinate system
        """
        if self.homography_matrix is None:
            return self.transform_point_simple(x, y)
        
        try:
            # Apply homography transformation
            point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point, self.homography_matrix)
            
            # Get transformed coordinates (đã là pixels trong target court space)
            tx, ty = transformed[0][0]
            
            # Scale transformed coordinates to our display court
            # Giả sử target court size là 1341x610 pixels (từ config)
            target_width = 1341
            target_height = 610
            
            # Scale to display court
            scale_x = self.court_width / target_width
            scale_y = self.court_height / target_height
            
            court_x = int(tx * scale_x) + self.court_x
            court_y = int(ty * scale_y) + self.court_y
            
            # Clamp to court bounds với tolerance
            court_x = max(self.court_x - 20, min(court_x, self.court_x + self.court_width + 20))
            court_y = max(self.court_y - 20, min(court_y, self.court_y + self.court_height + 20))
            
            return court_x, court_y
            
        except Exception as e:
            print(f"Transform error: {e}, fallback to simple")
            return self.transform_point_simple(x, y)

    def get_court_side(self, court_x: int, court_y: int) -> str:
        """
        Xác định player ở bên nào của sân - theo VERTICAL split.
        
        Args:
            court_x, court_y: Tọa độ trong court system
            
        Returns:
            'left' hoặc 'right'
        """
        if court_x < self.net_x:  # Chia theo chiều rộng
            return 'left'
        else:
            return 'right'

    def can_add_player_to_side(self, side: str, player_id: int) -> bool:
        """
        Kiểm tra có thể thêm player vào side không - ALLOW existing players to update.
        
        Args:
            side: 'left' hoặc 'right'
            player_id: ID của player
            
        Returns:
            True nếu có thể thêm
        """
        if side == 'left':
            # Nếu player đã ở side này hoặc side chưa đầy
            return (player_id in self.left_side_players or 
                   len(self.left_side_players) < self.max_players_per_side)
        else:
            return (player_id in self.right_side_players or 
                   len(self.right_side_players) < self.max_players_per_side)

    def add_player_to_side(self, side: str, player_id: int):
        """
        Thêm player vào side - ALLOW movement between sides.
        
        Args:
            side: 'left' hoặc 'right'
            player_id: ID của player
        """
        # ALWAYS allow updating position - remove from other side first
        self.left_side_players.discard(player_id)
        self.right_side_players.discard(player_id)
        
        # Add to correct side if space available
        if side == 'left' and len(self.left_side_players) < self.max_players_per_side:
            self.left_side_players.add(player_id)
        elif side == 'right' and len(self.right_side_players) < self.max_players_per_side:
            self.right_side_players.add(player_id)
        """
        Tạo canvas với sân pickleball đơn giản.
        
        Returns:
            Canvas image
        """
        # Create background
        canvas = np.full((self.canvas_height, self.canvas_width, 3), 
                        (50, 50, 50), dtype=np.uint8)
        
        # Draw court background
        cv2.rectangle(canvas, 
                     (self.court_x, self.court_y),
                     (self.court_x + self.court_width, self.court_y + self.court_height),
                     self.court_color, -1)
        
        # Court boundary
        cv2.rectangle(canvas,
                     (self.court_x, self.court_y),
                     (self.court_x + self.court_width, self.court_y + self.court_height),
                     self.line_color, 3)
        
        # Net (center line)
        net_y = self.court_y + self.court_height // 2
        cv2.line(canvas,
                (self.court_x, net_y),
                (self.court_x + self.court_width, net_y),
                self.line_color, 4)
        
        # Service lines (simplified)
        service_distance = self.court_height // 4
        
        # Top service line
        cv2.line(canvas,
                (self.court_x, self.court_y + service_distance),
                (self.court_x + self.court_width, self.court_y + service_distance),
                self.line_color, 2)
        
        # Bottom service line  
        cv2.line(canvas,
                (self.court_x, self.court_y + self.court_height - service_distance),
                (self.court_x + self.court_width, self.court_y + self.court_height - service_distance),
                self.line_color, 2)
        
        # Center service line
        center_x = self.court_x + self.court_width // 2
        cv2.line(canvas,
                (center_x, self.court_y),
                (center_x, self.court_y + service_distance),
                self.line_color, 2)
        cv2.line(canvas,
                (center_x, self.court_y + self.court_height - service_distance),
                (center_x, self.court_y + self.court_height),
                self.line_color, 2)
        
        # Non-volley zone (kitchen) - simplified
        kitchen_y1 = net_y - service_distance // 2
        kitchen_y2 = net_y + service_distance // 2
        cv2.rectangle(canvas,
                     (self.court_x, kitchen_y1),
                     (self.court_x + self.court_width, kitchen_y2),
                     (0, 255, 255), 2)
        
        # Hiển thị zone labels
        cv2.putText(canvas, "LEFT SIDE", 
                   (self.court_x + 10, self.court_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.line_color, 2)
        cv2.putText(canvas, "RIGHT SIDE", 
                   (self.court_x + 10, self.court_y + self.court_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.line_color, 2)
        
        # Hiển thị player count cho mỗi side
        cv2.putText(canvas, f"Players: {len(self.left_side_players)}/{self.max_players_per_side}", 
                   (self.court_x + 200, self.court_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(canvas, f"Players: {len(self.right_side_players)}/{self.max_players_per_side}", 
                   (self.court_x + 200, self.court_y + self.court_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return canvas

    def transform_point_simple(self, x: float, y: float) -> Tuple[int, int]:
        """
        Simple transformation - mapping dựa trên court bounds đã calibrated.
        
        Args:
            x, y: Tọa độ trong camera view
            
        Returns:
            (x, y) trong court coordinate system
        """
        if self.court_bounds is not None:
            # Use calibrated bounds for better mapping
            bounds_width = self.court_bounds['max_x'] - self.court_bounds['min_x']
            bounds_height = self.court_bounds['max_y'] - self.court_bounds['min_y']
            
            # Normalize to 0-1 based on court bounds
            norm_x = (x - self.court_bounds['min_x']) / bounds_width
            norm_y = (y - self.court_bounds['min_y']) / bounds_height
            
            # Map to court coordinates
            court_x = int(norm_x * self.court_width) + self.court_x
            court_y = int(norm_y * self.court_height) + self.court_y
            
        else:
            # Fallback: simple linear mapping (assume 1920x1080 video)
            scale_x = self.court_width / 1920.0
            scale_y = self.court_height / 1080.0
            
            court_x = int(x * scale_x) + self.court_x
            court_y = int(y * scale_y) + self.court_y
        
        # Clamp to court bounds
        court_x = max(self.court_x, min(court_x, self.court_x + self.court_width))
        court_y = max(self.court_y, min(court_y, self.court_y + self.court_height))
        
        return court_x, court_y

    def add_player_position(self, player_id: int, x: float, y: float, timestamp: float):
        """
        Thêm vị trí player mới - ALWAYS update position if in court.
        
        Args:
            player_id: ID của player
            x, y: Tọa độ trong camera view
            timestamp: Timestamp
        """
        # Kiểm tra xem point có trong sân không
        in_court = self.is_point_in_court(x, y)
        
        if not in_court:
            return  # Bỏ qua nếu ngoài sân
        
        # Transform to court coordinates
        if self.homography_matrix is not None:
            court_x, court_y = self.transform_point_calibrated(x, y)
        else:
            court_x, court_y = self.transform_point_simple(x, y)
        
        # Xác định side
        side = self.get_court_side(court_x, court_y)
        
        # ALWAYS update position - don't restrict by side limits for existing players
        current_in_left = player_id in self.left_side_players
        current_in_right = player_id in self.right_side_players
        
        # Allow movement if player already exists OR if side has space
        can_update = (current_in_left or current_in_right or 
                     self.can_add_player_to_side(side, player_id))
        
        if can_update:
            # Add player to side
            self.add_player_to_side(side, player_id)
            
            # Add to trail - ALWAYS add new position
            self.player_positions[player_id].append((court_x, court_y, timestamp, side))
            if len(self.player_positions[player_id]) > self.trail_length:
                self.player_positions[player_id].popleft()
            
            # Add to simple heatmap
            self.heatmap_points[player_id].append((court_x, court_y))

    def add_ball_position(self, x: float, y: float, timestamp: float):
        """
        Thêm vị trí ball mới - chỉ nếu trong sân.
        
        Args:
            x, y: Tọa độ trong camera view
            timestamp: Timestamp  
        """
        # Kiểm tra xem point có trong sân không
        if not self.is_point_in_court(x, y):
            return  # Bỏ qua nếu ngoài sân
            
        # Transform to court coordinates
        if self.homography_matrix is not None:
            court_x, court_y = self.transform_point_calibrated(x, y)
        else:
            court_x, court_y = self.transform_point_simple(x, y)
        
        # Add to trail
        self.ball_positions.append((court_x, court_y, timestamp))

    def create_court_canvas(self) -> np.ndarray:
        """
        Tạo canvas với sân pickleball đơn giản - CHỈ hình chữ nhật và net.
        
        Returns:
            Canvas image
        """
        # Create background
        canvas = np.full((self.canvas_height, self.canvas_width, 3), 
                        (50, 50, 50), dtype=np.uint8)
        
        # Draw court background (simple rectangle)
        cv2.rectangle(canvas, 
                     (self.court_x, self.court_y),
                     (self.court_x + self.court_width, self.court_y + self.court_height),
                     self.court_color, -1)
        
        # Court boundary (outer rectangle)
        cv2.rectangle(canvas,
                     (self.court_x, self.court_y),
                     (self.court_x + self.court_width, self.court_y + self.court_height),
                     self.line_color, 3)
        
        # NET - đường chia giữa theo chiều RỘNG (vertical line)
        net_x = self.court_x + self.court_width // 2
        cv2.line(canvas,
                (net_x, self.court_y),
                (net_x, self.court_y + self.court_height),
                self.line_color, 4)
        
        # Zone labels (vertical split)
        cv2.putText(canvas, "LEFT", 
                   (self.court_x + 20, self.court_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.line_color, 2)
        cv2.putText(canvas, "RIGHT", 
                   (net_x + 20, self.court_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.line_color, 2)
        
        # Player count cho mỗi side
        cv2.putText(canvas, f"Players: {len(self.left_side_players)}/2", 
                   (self.court_x + 20, self.court_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(canvas, f"Players: {len(self.right_side_players)}/2", 
                   (net_x + 20, self.court_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return canvas

    def draw_simple_heatmap(self, canvas: np.ndarray, player_id: int) -> np.ndarray:
        """
        Vẽ simple heatmap bằng circles.
        
        Args:
            canvas: Canvas để vẽ
            player_id: Player ID
            
        Returns:
            Canvas với heatmap
        """
        result = canvas.copy()
        
        if player_id not in self.heatmap_points:
            return result
        
        points = self.heatmap_points[player_id]
        if not points:
            return result
        
        # Get player color
        color = self.player_colors[player_id % len(self.player_colors)]
        
        # Draw heat points as circles with varying alpha
        overlay = result.copy()
        
        for i, (x, y) in enumerate(points):
            # Fade older points
            alpha = 0.1 + 0.4 * (i / len(points))
            cv2.circle(overlay, (x, y), 20, color, -1)
        
        # Blend overlay
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        
        return result

    def draw_players(self, canvas: np.ndarray) -> np.ndarray:
        """
        Vẽ players và trails lên canvas - IMPROVED với side colors.
        
        Args:
            canvas: Canvas để vẽ
            
        Returns:
            Canvas với players
        """
        result = canvas.copy()
        
        for player_id, positions in self.player_positions.items():
            if not positions:
                continue
            
            # Get player color
            color = self.player_colors[player_id % len(self.player_colors)]
            
            # Get player side
            current_side = positions[-1][3] if len(positions[-1]) > 3 else "unknown"
            
            # Different color intensity based on side
            if current_side == "left":
                # Side A - brighter colors
                color = tuple(min(255, int(c * 1.2)) for c in color)
            else:
                # Side B - original colors
                pass
            
            # Draw trail
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    # Fade trail
                    alpha = i / len(positions)
                    thickness = max(1, int(4 * alpha))
                    
                    pt1 = (positions[i-1][0], positions[i-1][1])
                    pt2 = (positions[i][0], positions[i][1])
                    cv2.line(result, pt1, pt2, color, thickness)
            
            # Draw current position - LARGER for visibility
            if positions:
                current_pos = positions[-1]
                cv2.circle(result, (current_pos[0], current_pos[1]), 
                          self.point_size + 2, color, -1)
                cv2.circle(result, (current_pos[0], current_pos[1]), 
                          self.point_size + 5, color, 3)
                
                # Draw player ID với side indicator
                side_letter = "L" if current_side == "left" else "R"
                label = f"P{player_id}({side_letter})"
                cv2.putText(result, label, 
                           (current_pos[0] + 15, current_pos[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result

    def draw_ball(self, canvas: np.ndarray) -> np.ndarray:
        """
        Vẽ ball và trail lên canvas - IMPROVED với better visibility.
        
        Args:
            canvas: Canvas để vẽ
            
        Returns:
            Canvas với ball
        """
        result = canvas.copy()
        
        if not self.ball_positions:
            return result
        
        # Draw ball trail với gradient effect
        if len(self.ball_positions) > 1:
            for i in range(1, len(self.ball_positions)):
                alpha = i / len(self.ball_positions)
                thickness = max(1, int(3 * alpha))
                
                pt1 = (self.ball_positions[i-1][0], self.ball_positions[i-1][1])
                pt2 = (self.ball_positions[i][0], self.ball_positions[i][1])
                
                # Yellow to white gradient
                trail_color = (int(255 * alpha), int(255 * alpha), 255)
                cv2.line(result, pt1, pt2, trail_color, thickness)
        
        # Draw current ball position - LARGER và prominent
        current_pos = self.ball_positions[-1]
        
        # Ball glow effect
        cv2.circle(result, (current_pos[0], current_pos[1]), 12, (0, 255, 255), 2)
        cv2.circle(result, (current_pos[0], current_pos[1]), 8, self.ball_color, -1)
        cv2.circle(result, (current_pos[0], current_pos[1]), 4, (255, 255, 0), -1)
        
        # Ball label với movement direction
        label_y = current_pos[1] - 20
        cv2.putText(result, "BALL", 
                   (current_pos[0] - 15, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ball_color, 2)
        
        # Show ball speed if có enough positions
        if len(self.ball_positions) >= 2:
            prev_pos = self.ball_positions[-2]
            curr_pos = self.ball_positions[-1]
            
            # Calculate distance moved
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            distance = int((dx*dx + dy*dy)**0.5)
            
            if distance > 5:  # Only show if significant movement
                cv2.putText(result, f"Speed: {distance}", 
                           (current_pos[0] - 20, label_y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return result

    def draw_info_panel(self, canvas: np.ndarray, frame_info: Dict[str, Any]) -> np.ndarray:
        """
        Vẽ info panel nhỏ.
        
        Args:
            canvas: Canvas để vẽ
            frame_info: Thông tin frame
            
        Returns:
            Canvas với info panel
        """
        result = canvas.copy()
        
        # Small info panel ở góc trên trái
        panel_w = 250
        panel_h = 120
        
        # Background
        cv2.rectangle(result, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(result, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 2)
        
        # Title
        cv2.putText(result, "2D COURT VIEW", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Info
        y_pos = 60
        cv2.putText(result, f"Frame: {frame_info.get('frame', 0)}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_pos += 20
        cv2.putText(result, f"Time: {frame_info.get('time', 0):.1f}s", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Player count per side
        y_pos += 20
        cv2.putText(result, f"Left: {len(self.left_side_players)}, Right: {len(self.right_side_players)}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result

    def render_frame(self, frame_info: Dict[str, Any], show_heatmap: bool = False, 
                    heatmap_player: Optional[int] = None) -> np.ndarray:
        """
        Render một frame của court view.
        
        Args:
            frame_info: Thông tin frame
            show_heatmap: Có hiển thị heatmap không
            heatmap_player: Player ID để hiển thị heatmap
            
        Returns:
            Rendered frame
        """
        # Create court canvas
        canvas = self.create_court_canvas()
        
        # Add simple heatmap
        if show_heatmap:
            if heatmap_player is not None:
                canvas = self.draw_simple_heatmap(canvas, heatmap_player)
            else:
                # Show all players heatmap
                for player_id in self.player_positions.keys():
                    canvas = self.draw_simple_heatmap(canvas, player_id)
        
        # Draw players
        canvas = self.draw_players(canvas)
        
        # Draw ball
        canvas = self.draw_ball(canvas)
        
        # Draw info panel
        canvas = self.draw_info_panel(canvas, frame_info)
        
        return canvas

def run_simple_demo(video_path: str, tracking_csv: str, max_frames: int = 100):
    """
    Chạy simple demo với tracking data.
    
    Args:
        video_path: Đường dẫn video
        tracking_csv: Đường dẫn tracking CSV
        max_frames: Số frames tối đa
    """
    # Load tracking data
    try:
        tracking_df = pd.read_csv(tracking_csv)
        print(f"📊 Loaded {len(tracking_df)} tracking records")
    except Exception as e:
        print(f"❌ Error loading tracking data: {e}")
        return
    
    # Initialize viewer với court config
    viewer = Simple2DCourtViewer("config/court_points.json")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("🔴 Starting simple 2D court demo...")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  'h': Toggle heatmap")
    print("  '1-6': Show specific player heatmap") 
    print("  '0': Show all players heatmap")
    print("  'q': Quit")
    
    frame_count = 0
    paused = False
    show_heatmap = False
    heatmap_player = None
    
    # Create windows side-by-side
    cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2D Court View", cv2.WINDOW_NORMAL)
    
    # Position windows side by side
    cv2.moveWindow("Original Video", 100, 100)
    cv2.moveWindow("2D Court View", 800, 100)
    
    cv2.resizeWindow("Original Video", 640, 480)
    cv2.resizeWindow("2D Court View", 800, 600)
    
    while frame_count < max_frames:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get tracking data for current frame
            frame_detections = tracking_df[tracking_df['frame_id'] == frame_count]
            
            # Update court with new positions
            for _, detection in frame_detections.iterrows():
                obj_id = int(detection['object_id'])
                center_x = detection['center_x']
                center_y = detection['center_y'] 
                timestamp = detection['timestamp']
                class_name = detection['class']
                
                if 'player' in class_name or class_name == 'person':
                    viewer.add_player_position(obj_id, center_x, center_y, timestamp)
                elif class_name == 'ball':
                    viewer.add_ball_position(center_x, center_y, timestamp)
            
            frame_count += 1
        
        # Render court view
        frame_info = {
            'frame': frame_count,
            'time': frame_count / fps,
            'detections': len(frame_detections) if not paused else 0
        }
        
        court_view = viewer.render_frame(frame_info, show_heatmap, heatmap_player)
        
        # Show original video with detections
        if 'frame' in locals():
            display_frame = frame.copy()
            
            # Draw detections on original
            if not paused and 'frame_detections' in locals():
                for _, detection in frame_detections.iterrows():
                    x1, y1 = int(detection['bbox_x1']), int(detection['bbox_y1'])
                    x2, y2 = int(detection['bbox_x2']), int(detection['bbox_y2'])
                    center_x, center_y = int(detection['center_x']), int(detection['center_y'])
                    obj_id = int(detection['object_id'])
                    class_name = detection['class']
                    
                    # Color based on class
                    if 'player' in class_name or class_name == 'person':
                        color = viewer.player_colors[obj_id % len(viewer.player_colors)]
                    else:
                        color = viewer.ball_color
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(display_frame, (center_x, center_y), 5, color, -1)
                    cv2.putText(display_frame, f"{class_name}_{obj_id}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow("Original Video", display_frame)
        
        cv2.imshow("2D Court View", court_view)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            status = "PAUSED" if paused else "PLAYING"
            print(f"📺 {status} at frame {frame_count}")
        elif key == ord('h'):
            show_heatmap = not show_heatmap
            print(f"🔥 Heatmap: {'ON' if show_heatmap else 'OFF'}")
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            heatmap_player = int(chr(key)) - 1
            print(f"🔥 Showing heatmap for Player {heatmap_player}")
        elif key == ord('0'):
            heatmap_player = None
            print(f"🔥 Showing heatmap for all players")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Simple demo completed ({frame_count} frames)")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple 2D Court Viewer")
    parser.add_argument('--video', type=str, required=True,
                       help='Đường dẫn video')
    parser.add_argument('--tracking', type=str, required=True,
                       help='Đường dẫn tracking data CSV')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Số frames tối đa')
    
    args = parser.parse_args()
    
    run_simple_demo(args.video, args.tracking, args.max_frames)

if __name__ == "__main__":
    main()