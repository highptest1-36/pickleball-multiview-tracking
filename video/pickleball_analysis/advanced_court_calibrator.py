"""
Advanced Court Calibration Tool - Chọn nhiều điểm trên sân để tạo perspective transformation chính xác

Tool này cho phép:
1. Chọn nhiều điểm trên sân (không chỉ 4 góc)
2. Chọn các điểm đặc biệt: net, service lines, baselines, sidelines
3. Tính toán homography matrix từ nhiều điểm tương ứng
4. Test và fine-tune transformation
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Optional, Dict, Any
import argparse
import math

class AdvancedCourtCalibrator:
    def __init__(self, video_path: str):
        """
        Initialize advanced court calibrator.
        
        Args:
            video_path: Đường dẫn video để calibrate
        """
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Court points mapping
        self.court_points = {}  # Tên điểm -> [x, y]
        self.frame = None
        self.original_frame = None
        
        # Target court dimensions (bird's-eye view) - 100 pixels = 1 meter
        self.target_width = 1341   # 13.41m * 100 pixels/m  
        self.target_height = 610   # 6.1m * 100 pixels/m
        self.padding = 50
        
        # Define court reference points (bird's-eye view coordinates)
        self.reference_points = self._create_reference_points()
        
        # Current selection mode
        self.selection_mode = "corners"
        self.available_modes = ["corners", "lines", "zones", "custom"]
        
        # Colors for different point types
        self.colors = {
            "corners": (0, 255, 0),      # Green
            "net": (255, 255, 0),        # Cyan
            "service": (255, 0, 255),    # Magenta
            "baseline": (0, 0, 255),     # Red
            "sideline": (255, 165, 0),   # Orange
            "center": (255, 255, 255),   # White
            "kitchen": (0, 255, 255),    # Yellow
            "custom": (128, 128, 128)    # Gray
        }
        
        self.window_name = f"Advanced Court Calibration - {self.video_name}"
        
        print(f"🎯 Advanced Court Calibrator initialized")
        print(f"📐 Target court: {self.target_width}x{self.target_height} pixels")

    def _create_reference_points(self) -> Dict[str, Tuple[float, float]]:
        """
        Tạo reference points cho sân pickleball chuẩn (bird's-eye view).
        
        Returns:
            Dictionary mapping tên điểm -> (x, y) coordinates
        """
        # Court dimensions in pixels (với padding)
        court_left = self.padding
        court_right = self.target_width - self.padding
        court_top = self.padding
        court_bottom = self.target_height - self.padding
        
        court_width = court_right - court_left  # 1241 pixels
        court_height = court_bottom - court_top  # 510 pixels
        
        # Center coordinates
        center_x = (court_left + court_right) / 2
        center_y = (court_top + court_bottom) / 2
        
        # Service line distance from net (2.13m = 213 pixels)
        service_distance = 213
        
        # Kitchen/NVZ depth (2.13m = 213 pixels) 
        kitchen_depth = 213
        
        reference = {
            # Corner points
            "corner_tl": (court_left, court_top),
            "corner_tr": (court_right, court_top), 
            "corner_bl": (court_left, court_bottom),
            "corner_br": (court_right, court_bottom),
            
            # Net line (center)
            "net_left": (court_left, center_y),
            "net_right": (court_right, center_y),
            "net_center": (center_x, center_y),
            
            # Baseline (back lines)
            "baseline_top_left": (court_left, court_top),
            "baseline_top_right": (court_right, court_top),
            "baseline_bottom_left": (court_left, court_bottom),
            "baseline_bottom_right": (court_right, court_bottom),
            
            # Service lines
            "service_top_left": (court_left, court_top + service_distance),
            "service_top_right": (court_right, court_top + service_distance),
            "service_bottom_left": (court_left, court_bottom - service_distance),
            "service_bottom_right": (court_right, court_bottom - service_distance),
            
            # Center service lines
            "center_service_top": (center_x, court_top + service_distance),
            "center_service_bottom": (center_x, court_bottom - service_distance),
            
            # Kitchen/Non-Volley Zone
            "kitchen_tl": (court_left, center_y - kitchen_depth),
            "kitchen_tr": (court_right, center_y - kitchen_depth),
            "kitchen_bl": (court_left, center_y + kitchen_depth),
            "kitchen_br": (court_right, center_y + kitchen_depth),
            
            # Sidelines mid-points
            "sideline_left_mid": (court_left, center_y),
            "sideline_right_mid": (court_right, center_y),
            
            # Additional intersection points
            "center_line_top": (center_x, court_top),
            "center_line_bottom": (center_x, court_bottom),
            
            # Quarter court points
            "quarter_tl": (court_left + court_width/4, court_top + court_height/4),
            "quarter_tr": (court_right - court_width/4, court_top + court_height/4),
            "quarter_bl": (court_left + court_width/4, court_bottom - court_height/4),
            "quarter_br": (court_right - court_width/4, court_bottom - court_height/4),
        }
        
        return reference

    def get_point_category(self, point_name: str) -> str:
        """
        Lấy category của point để xác định màu.
        
        Args:
            point_name: Tên điểm
            
        Returns:
            Category name
        """
        if "corner" in point_name:
            return "corners"
        elif "net" in point_name:
            return "net"
        elif "service" in point_name:
            return "service"
        elif "baseline" in point_name:
            return "baseline"
        elif "sideline" in point_name:
            return "sideline"
        elif "kitchen" in point_name:
            return "kitchen"
        elif "center" in point_name:
            return "center"
        else:
            return "custom"

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback để click chọn điểm.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Tìm điểm reference gần nhất
            nearest_point = self.find_nearest_reference_point(x, y)
            
            if nearest_point:
                point_name, ref_coords = nearest_point
                self.court_points[point_name] = [x, y]
                
                # Get color for this point type
                category = self.get_point_category(point_name)
                color = self.colors.get(category, (255, 255, 255))
                
                print(f"📍 Selected: {point_name} at ({x}, {y})")
                
                # Draw point
                cv2.circle(self.frame, (x, y), 6, color, -1)
                cv2.circle(self.frame, (x, y), 8, color, 2)
                cv2.putText(self.frame, point_name, (x+10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                cv2.imshow(self.window_name, self.frame)
                
                print(f"✅ Total points selected: {len(self.court_points)}")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to remove nearest point
            self.remove_nearest_point(x, y)

    def find_nearest_reference_point(self, x: int, y: int, max_distance: int = 50) -> Optional[Tuple[str, Tuple[float, float]]]:
        """
        Tìm reference point gần nhất với click position.
        
        Args:
            x, y: Click coordinates
            max_distance: Khoảng cách tối đa để consider
            
        Returns:
            (point_name, reference_coords) hoặc None
        """
        min_distance = float('inf')
        nearest = None
        
        # Hiển thị all available points gần click
        nearby_points = []
        
        for point_name, ref_coords in self.reference_points.items():
            if point_name in self.court_points:
                continue  # Skip already selected points
            
            # Calculate distance (không cần transform vì chỉ so sánh relative)
            distance = math.sqrt((x - ref_coords[0])**2 + (y - ref_coords[1])**2)
            
            if distance < max_distance:
                nearby_points.append((point_name, distance))
        
        if nearby_points:
            # Sort by distance và hiển thị options
            nearby_points.sort(key=lambda x: x[1])
            print(f"\n🎯 Available points near click:")
            for i, (name, dist) in enumerate(nearby_points[:5]):
                print(f"  {i+1}. {name} (distance: {dist:.1f})")
            
            # Chọn point gần nhất
            nearest_name = nearby_points[0][0]
            return (nearest_name, self.reference_points[nearest_name])
        
        return None

    def remove_nearest_point(self, x: int, y: int):
        """
        Remove điểm gần nhất với right click.
        
        Args:
            x, y: Click coordinates
        """
        min_distance = float('inf')
        nearest_point = None
        
        for point_name, coords in self.court_points.items():
            distance = math.sqrt((x - coords[0])**2 + (y - coords[1])**2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point_name
        
        if nearest_point and min_distance < 30:
            del self.court_points[nearest_point]
            print(f"🗑️  Removed: {nearest_point}")
            
            # Redraw frame
            self.frame = self.original_frame.copy()
            self.draw_current_points()
            cv2.imshow(self.window_name, self.frame)

    def draw_current_points(self):
        """Vẽ tất cả điểm đã chọn."""
        for point_name, coords in self.court_points.items():
            category = self.get_point_category(point_name)
            color = self.colors.get(category, (255, 255, 255))
            
            cv2.circle(self.frame, tuple(coords), 6, color, -1)
            cv2.circle(self.frame, tuple(coords), 8, color, 2)
            cv2.putText(self.frame, point_name, (coords[0]+10, coords[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_reference_overlay(self):
        """Vẽ overlay hiển thị reference points để user biết chọn đâu."""
        overlay = self.frame.copy()
        
        # Draw suggested court outline
        court_points = [
            self.reference_points["corner_tl"],
            self.reference_points["corner_tr"], 
            self.reference_points["corner_br"],
            self.reference_points["corner_bl"]
        ]
        
        court_pts = np.array(court_points, dtype=np.int32)
        cv2.polylines(overlay, [court_pts], True, (100, 100, 100), 2)
        
        # Draw net line
        net_start = self.reference_points["net_left"]
        net_end = self.reference_points["net_right"]
        cv2.line(overlay, 
                (int(net_start[0]), int(net_start[1])),
                (int(net_end[0]), int(net_end[1])), 
                (100, 100, 100), 2)
        
        # Blend with main frame
        self.frame = cv2.addWeighted(self.frame, 0.8, overlay, 0.2, 0)

    def calibrate_court(self) -> bool:
        """
        Interactive court calibration với nhiều điểm.
        
        Returns:
            True nếu calibration thành công
        """
        print("🎬 Opening video for advanced calibration...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return False
        
        # Get first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Cannot read first frame")
            return False
        
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 900)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n🖱️  ADVANCED CALIBRATION INSTRUCTIONS:")
        print("════════════════════════════════════════")
        print("1. Left click gần vị trí muốn chọn")
        print("2. Tool sẽ suggest các điểm reference gần nhất")
        print("3. Right click để xóa điểm đã chọn")
        print("4. Chọn ít nhất 8-10 điểm để có transformation tốt")
        print("5. Ưu tiên chọn:")
        print("   📍 4 góc sân (corners)")
        print("   📍 Net line (left, center, right)")
        print("   📍 Service lines")
        print("   📍 Baseline intersections")
        print("\nControls:")
        print("  's' - Save calibration")
        print("  'r' - Reset tất cả điểm")
        print("  'h' - Show/hide reference overlay")
        print("  'l' - List selected points")
        print("  'q' - Quit")
        print("════════════════════════════════════════")
        
        show_overlay = True
        
        while True:
            display_frame = self.frame.copy()
            
            if show_overlay:
                self.draw_reference_overlay()
                display_frame = self.frame.copy()
            
            # Draw info panel
            self.draw_info_panel(display_frame)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("🚪 Calibration cancelled")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('r'):
                print("🔄 Reset all points")
                self.court_points = {}
                self.frame = self.original_frame.copy()
            
            elif key == ord('h'):
                show_overlay = not show_overlay
                print(f"👁️  Reference overlay: {'ON' if show_overlay else 'OFF'}")
                self.frame = self.original_frame.copy()
                self.draw_current_points()
            
            elif key == ord('l'):
                self.list_selected_points()
            
            elif key == ord('s'):
                if len(self.court_points) >= 4:
                    print("💾 Saving calibration...")
                    success = self.save_calibration()
                    if success:
                        print("✅ Calibration saved successfully!")
                        cv2.destroyAllWindows()
                        return True
                    else:
                        print("❌ Failed to save calibration")
                else:
                    print(f"⚠️  Need at least 4 points (current: {len(self.court_points)})")
        
        cv2.destroyAllWindows()
        return False

    def draw_info_panel(self, frame: np.ndarray):
        """Vẽ info panel với statistics."""
        panel_h = 120
        panel_w = 400
        
        # Background
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "COURT CALIBRATION", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Stats
        y_pos = 60
        cv2.putText(frame, f"Points selected: {len(self.court_points)}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 20
        status = "Ready to save" if len(self.court_points) >= 4 else "Need more points"
        color = (0, 255, 0) if len(self.court_points) >= 4 else (0, 255, 255)
        cv2.putText(frame, f"Status: {status}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Legend
        y_pos += 25
        cv2.putText(frame, "Legend:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        legend_items = [
            ("Corners", self.colors["corners"]),
            ("Net", self.colors["net"]),
            ("Service", self.colors["service"]),
            ("Baseline", self.colors["baseline"])
        ]
        
        x_start = 80
        for i, (name, color) in enumerate(legend_items):
            x_pos = x_start + i * 80
            cv2.circle(frame, (x_pos, y_pos + 5), 4, color, -1)
            cv2.putText(frame, name, (x_pos + 10, y_pos + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    def list_selected_points(self):
        """In danh sách các điểm đã chọn."""
        print(f"\n📋 SELECTED POINTS ({len(self.court_points)}):")
        print("─" * 40)
        
        categories = {}
        for point_name in self.court_points.keys():
            category = self.get_point_category(point_name)
            if category not in categories:
                categories[category] = []
            categories[category].append(point_name)
        
        for category, points in categories.items():
            print(f"📍 {category.upper()}: {len(points)} points")
            for point in points:
                coords = self.court_points[point]
                print(f"   • {point}: ({coords[0]}, {coords[1]})")
        print("─" * 40)

    def calculate_homography(self) -> Optional[np.ndarray]:
        """
        Tính homography matrix từ các điểm đã chọn.
        
        Returns:
            Homography matrix hoặc None nếu không đủ điểm
        """
        if len(self.court_points) < 4:
            print("❌ Need at least 4 points for homography calculation")
            return None
        
        # Prepare point pairs
        source_points = []
        target_points = []
        
        for point_name, camera_coords in self.court_points.items():
            if point_name in self.reference_points:
                source_points.append(camera_coords)
                target_points.append(self.reference_points[point_name])
        
        if len(source_points) < 4:
            print("❌ Need at least 4 valid reference points")
            return None
        
        source_pts = np.array(source_points, dtype=np.float32)
        target_pts = np.array(target_points, dtype=np.float32)
        
        print(f"🧮 Calculating homography from {len(source_points)} point pairs...")
        
        # Calculate homography
        if len(source_points) == 4:
            # Exact solution
            homography = cv2.getPerspectiveTransform(source_pts, target_pts)
        else:
            # Over-determined system (more than 4 points)
            homography, mask = cv2.findHomography(source_pts, target_pts, 
                                                 cv2.RANSAC, 5.0)
        
        if homography is not None:
            print("✅ Homography calculation successful")
            
            # Calculate reprojection error
            error = self.calculate_reprojection_error(source_pts, target_pts, homography)
            print(f"📊 Reprojection error: {error:.2f} pixels")
            
            return homography
        else:
            print("❌ Homography calculation failed")
            return None

    def calculate_reprojection_error(self, source_pts: np.ndarray, 
                                   target_pts: np.ndarray, 
                                   homography: np.ndarray) -> float:
        """
        Tính reprojection error để đánh giá chất lượng homography.
        
        Args:
            source_pts: Source points
            target_pts: Target points  
            homography: Homography matrix
            
        Returns:
            Average reprojection error in pixels
        """
        # Transform source points
        transformed_pts = cv2.perspectiveTransform(
            source_pts.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        
        # Calculate distances
        errors = np.sqrt(np.sum((transformed_pts - target_pts)**2, axis=1))
        
        return np.mean(errors)

    def save_calibration(self) -> bool:
        """
        Save calibration data với homography matrix.
        
        Returns:
            True nếu save thành công
        """
        try:
            # Calculate homography
            homography = self.calculate_homography()
            if homography is None:
                return False
            
            # Load existing config
            config_file = "config/court_points.json"
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Update camera data
            camera_key = self.video_name
            
            # Convert selected points to the expected format
            corner_mapping = {
                "corner_tl": "top_left",
                "corner_tr": "top_right", 
                "corner_br": "bottom_right",
                "corner_bl": "bottom_left"
            }
            
            court_corners = {}
            for ref_name, config_name in corner_mapping.items():
                if ref_name in self.court_points:
                    court_corners[config_name] = self.court_points[ref_name]
                else:
                    # Use default values if corners not selected
                    court_corners[config_name] = [0, 0]
            
            config['cameras'][camera_key] = {
                "description": f"Camera {camera_key} (advanced calibration)",
                "court_corners": court_corners,
                "calibration_status": "calibrated",
                "calibration_method": "advanced_multi_point",
                "points_used": len(self.court_points),
                "all_points": self.court_points,
                "reference_points": {k: list(v) for k, v in self.reference_points.items()},
                "homography_matrix": homography.tolist(),
                "notes": f"Advanced calibration with {len(self.court_points)} points"
            }
            
            # Save back to file
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Advanced calibration saved to: {config_file}")
            print(f"🎯 Camera: {camera_key}")
            print(f"📊 Points used: {len(self.court_points)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False

    def test_calibration(self, max_frames: int = 100) -> bool:
        """
        Test advanced calibration.
        
        Args:
            max_frames: Số frames để test
            
        Returns:
            True nếu test thành công
        """
        homography = self.calculate_homography()
        if homography is None:
            return False
        
        print(f"🧪 Testing advanced calibration...")
        print("Controls: 'q' to quit, SPACE to pause/resume")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
        
        frame_count = 0
        paused = False
        
        while frame_count < max_frames:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Apply transformation
            transformed = cv2.warpPerspective(frame, homography,
                                            (self.target_width, self.target_height))
            
            # Draw court lines
            court_frame = self.draw_court_lines(transformed)
            
            # Show side by side
            display_original = cv2.resize(frame, (700, 500))
            display_transformed = cv2.resize(court_frame, (700, 500))
            
            # Labels
            cv2.putText(display_original, "ORIGINAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_transformed, "BIRD'S-EYE VIEW", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Info
            cv2.putText(display_transformed, f"Points: {len(self.court_points)}", 
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            combined = np.hstack([display_original, display_transformed])
            cv2.imshow("Advanced Calibration Test", combined)
            
            if not paused:
                frame_count += 1
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Advanced calibration test completed")
        return True

    def draw_court_lines(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ court lines trên bird's-eye view."""
        court_frame = frame.copy()
        
        # Colors
        line_color = (255, 255, 255)
        net_color = (0, 255, 255)
        kitchen_color = (255, 255, 0)
        
        # Court boundary
        cv2.rectangle(court_frame, (self.padding, self.padding),
                     (self.target_width - self.padding, self.target_height - self.padding),
                     line_color, 2)
        
        # Net
        net_y = self.target_height // 2
        cv2.line(court_frame, (self.padding, net_y),
                (self.target_width - self.padding, net_y), net_color, 3)
        
        # Service lines
        service_dist = 213  # 2.13m
        cv2.line(court_frame, (self.padding, self.padding + service_dist),
                (self.target_width - self.padding, self.padding + service_dist), line_color, 2)
        cv2.line(court_frame, (self.padding, self.target_height - self.padding - service_dist),
                (self.target_width - self.padding, self.target_height - self.padding - service_dist), line_color, 2)
        
        # Center service line
        center_x = self.target_width // 2
        cv2.line(court_frame, (center_x, self.padding),
                (center_x, self.padding + service_dist), line_color, 2)
        cv2.line(court_frame, (center_x, self.target_height - self.padding - service_dist),
                (center_x, self.target_height - self.padding), line_color, 2)
        
        # Kitchen
        cv2.rectangle(court_frame, (self.padding, net_y - service_dist//2),
                     (self.target_width - self.padding, net_y + service_dist//2),
                     kitchen_color, 2)
        
        return court_frame

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Advanced Court Calibration Tool")
    parser.add_argument('--video', type=str, required=True,
                       help='Đường dẫn video để calibrate')
    parser.add_argument('--test', action='store_true',
                       help='Test existing calibration')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Max frames for testing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    calibrator = AdvancedCourtCalibrator(args.video)
    
    if args.test:
        success = calibrator.test_calibration(args.max_frames)
    else:
        success = calibrator.calibrate_court()
        
        if success:
            test_input = input("🧪 Test calibration now? (y/n): ")
            if test_input.lower() == 'y':
                calibrator.test_calibration()

if __name__ == "__main__":
    main()