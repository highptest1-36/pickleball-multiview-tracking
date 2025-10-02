"""
Analysis Module

Module này xử lý:
1. Phân tích chuyển động và tính toán vận tốc
2. Tính quãng đường di chuyển (covered distance)
3. Phân tích running pace
4. Tạo các metrics thống kê
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import math
from collections import defaultdict

from .utils import calculate_distance, calculate_velocity, pixel_to_meters, Timer

class MovementAnalyzer:
    """Class phân tích chuyển động của players và ball."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo MovementAnalyzer.
        
        Args:
            config: Cấu hình từ config.yaml
        """
        self.config = config
        self.analysis_config = config['analysis']
        self.court_config = config['court']
        
        # Analysis parameters
        self.smoothing_enabled = self.analysis_config['smoothing']['enabled']
        self.smoothing_window = self.analysis_config['smoothing']['window_size']
        self.min_distance_threshold = self.analysis_config['velocity']['min_distance_threshold']
        
        # Court dimensions
        self.court_width_m = self.court_config['width_meters']
        self.court_height_m = self.court_config['height_meters']
        
        logger.info("MovementAnalyzer đã được khởi tạo")

    def analyze_tracking_data(self, tracking_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích toàn bộ tracking data.
        
        Args:
            tracking_df: DataFrame chứa tracking data
            
        Returns:
            Dictionary chứa kết quả phân tích
        """
        logger.info(f"Bắt đầu phân tích {len(tracking_df)} tracking records")
        
        results = {
            'player_analysis': {},
            'ball_analysis': {},
            'match_statistics': {},
            'movement_patterns': {}
        }
        
        with Timer("Movement analysis"):
            # Phân tích theo từng object
            for object_id in tracking_df['object_id'].unique():
                object_data = tracking_df[tracking_df['object_id'] == object_id]
                object_class = object_data['class'].iloc[0]
                
                if object_class == 'player':
                    player_analysis = self.analyze_player_movement(object_data)
                    results['player_analysis'][f'player_{object_id}'] = player_analysis
                elif object_class == 'ball':
                    ball_analysis = self.analyze_ball_movement(object_data)
                    results['ball_analysis'][f'ball_{object_id}'] = ball_analysis
            
            # Phân tích tổng thể
            results['match_statistics'] = self.calculate_match_statistics(tracking_df)
            results['movement_patterns'] = self.analyze_movement_patterns(tracking_df)
        
        logger.info("Hoàn thành phân tích chuyển động")
        return results

    def analyze_player_movement(self, player_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích chuyển động của một player.
        
        Args:
            player_data: DataFrame chứa data của player
            
        Returns:
            Dictionary chứa phân tích player
        """
        if len(player_data) < 2:
            return {'error': 'Insufficient data for analysis'}
        
        # Convert pixel coordinates to meters
        positions_meters = []
        for _, row in player_data.iterrows():
            # Assuming bird's-eye view with 10 pixels = 1 meter
            x_m = row['center_x'] / 10.0
            y_m = row['center_y'] / 10.0
            positions_meters.append((x_m, y_m))
        
        timestamps = player_data['timestamp'].tolist()
        
        # Calculate velocities
        velocities = calculate_velocity(positions_meters, timestamps, self.smoothing_window)
        
        # Calculate speeds (magnitude of velocity)
        speeds = [math.sqrt(vx**2 + vy**2) for vx, vy in velocities]
        
        # Calculate distances
        distances = []
        total_distance = 0
        for i in range(1, len(positions_meters)):
            dist = calculate_distance(positions_meters[i-1], positions_meters[i])
            if dist > self.min_distance_threshold:  # Filter noise
                distances.append(dist)
                total_distance += dist
            else:
                distances.append(0)
        
        # Calculate accelerations
        accelerations = self._calculate_accelerations(velocities, timestamps)
        
        # Calculate running pace (min/km)
        avg_speed_ms = np.mean([s for s in speeds if s > 0]) if speeds else 0
        avg_speed_kmh = avg_speed_ms * 3.6
        running_pace_min_per_km = (60 / avg_speed_kmh) if avg_speed_kmh > 0 else 0
        
        # Analyze movement zones
        movement_zones = self._analyze_movement_zones(positions_meters)
        
        # Calculate direction changes
        direction_changes = self._calculate_direction_changes(velocities)
        
        analysis = {
            'duration_seconds': timestamps[-1] - timestamps[0] if timestamps else 0,
            'total_distance_meters': total_distance,
            'avg_speed_ms': avg_speed_ms,
            'max_speed_ms': max(speeds) if speeds else 0,
            'avg_speed_kmh': avg_speed_kmh,
            'max_speed_kmh': max(speeds) * 3.6 if speeds else 0,
            'running_pace_min_per_km': running_pace_min_per_km,
            'direction_changes': direction_changes,
            'movement_zones': movement_zones,
            'positions': positions_meters,
            'velocities': velocities,
            'speeds': speeds,
            'accelerations': accelerations,
            'timestamps': timestamps
        }
        
        return analysis

    def analyze_ball_movement(self, ball_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích chuyển động của bóng.
        
        Args:
            ball_data: DataFrame chứa data của bóng
            
        Returns:
            Dictionary chứa phân tích bóng
        """
        if len(ball_data) < 2:
            return {'error': 'Insufficient data for analysis'}
        
        # Convert pixel coordinates to meters
        positions_meters = []
        for _, row in ball_data.iterrows():
            x_m = row['center_x'] / 10.0
            y_m = row['center_y'] / 10.0
            positions_meters.append((x_m, y_m))
        
        timestamps = ball_data['timestamp'].tolist()
        
        # Calculate velocities
        velocities = calculate_velocity(positions_meters, timestamps, smoothing_window=2)  # Less smoothing for ball
        
        # Calculate speeds
        speeds = [math.sqrt(vx**2 + vy**2) for vx, vy in velocities]
        
        # Detect ball hits (sudden direction/speed changes)
        ball_hits = self._detect_ball_hits(velocities, speeds, timestamps)
        
        # Calculate trajectory metrics
        max_height = max([pos[1] for pos in positions_meters]) if positions_meters else 0
        min_height = min([pos[1] for pos in positions_meters]) if positions_meters else 0
        
        analysis = {
            'duration_seconds': timestamps[-1] - timestamps[0] if timestamps else 0,
            'max_speed_ms': max(speeds) if speeds else 0,
            'max_speed_kmh': max(speeds) * 3.6 if speeds else 0,
            'avg_speed_ms': np.mean(speeds) if speeds else 0,
            'ball_hits_detected': len(ball_hits),
            'ball_hits': ball_hits,
            'max_height': max_height,
            'min_height': min_height,
            'height_variation': max_height - min_height,
            'positions': positions_meters,
            'velocities': velocities,
            'speeds': speeds,
            'timestamps': timestamps
        }
        
        return analysis

    def _calculate_accelerations(self, velocities: List[Tuple[float, float]], 
                                timestamps: List[float]) -> List[Tuple[float, float]]:
        """
        Tính gia tốc từ vận tốc.
        
        Args:
            velocities: List vận tốc [(vx, vy), ...]
            timestamps: List timestamp
            
        Returns:
            List gia tốc [(ax, ay), ...]
        """
        if len(velocities) < 2:
            return [(0, 0)] * len(velocities)
        
        accelerations = []
        
        for i in range(len(velocities)):
            if i == 0:
                # First frame
                dt = timestamps[i+1] - timestamps[i] if i+1 < len(timestamps) else 1
                dvx = velocities[i+1][0] - velocities[i][0] if i+1 < len(velocities) else 0
                dvy = velocities[i+1][1] - velocities[i][1] if i+1 < len(velocities) else 0
            elif i == len(velocities) - 1:
                # Last frame
                dt = timestamps[i] - timestamps[i-1]
                dvx = velocities[i][0] - velocities[i-1][0]
                dvy = velocities[i][1] - velocities[i-1][1]
            else:
                # Middle frames
                dt = timestamps[i+1] - timestamps[i-1]
                dvx = velocities[i+1][0] - velocities[i-1][0]
                dvy = velocities[i+1][1] - velocities[i-1][1]
            
            if dt > 0:
                ax = dvx / dt
                ay = dvy / dt
            else:
                ax, ay = 0, 0
                
            accelerations.append((ax, ay))
        
        return accelerations

    def _analyze_movement_zones(self, positions: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Phân tích zones di chuyển trên sân.
        
        Args:
            positions: List vị trí [(x, y), ...]
            
        Returns:
            Dictionary phân tích zones
        """
        if not positions:
            return {}
        
        # Chia sân thành các zones
        court_width = self.court_width_m
        court_height = self.court_height_m
        
        # Zones: left/right court, front/back court
        zones = {
            'left_court': 0,
            'right_court': 0,
            'front_court': 0,
            'back_court': 0,
            'net_area': 0
        }
        
        net_area_margin = 1.0  # 1 meter around net
        
        for x, y in positions:
            # Left/Right
            if x < court_width / 2:
                zones['left_court'] += 1
            else:
                zones['right_court'] += 1
            
            # Front/Back
            if y < court_height / 2:
                zones['front_court'] += 1
            else:
                zones['back_court'] += 1
            
            # Net area
            if abs(y - court_height/2) < net_area_margin:
                zones['net_area'] += 1
        
        # Convert to percentages
        total_positions = len(positions)
        for zone in zones:
            zones[zone] = (zones[zone] / total_positions) * 100 if total_positions > 0 else 0
        
        # Calculate court coverage
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        coverage = {
            'x_range': max(x_coords) - min(x_coords) if x_coords else 0,
            'y_range': max(y_coords) - min(y_coords) if y_coords else 0,
            'coverage_area_percentage': 0  # Could implement convex hull area calculation
        }
        
        zones['coverage'] = coverage
        return zones

    def _calculate_direction_changes(self, velocities: List[Tuple[float, float]]) -> int:
        """
        Tính số lần thay đổi hướng đáng kể.
        
        Args:
            velocities: List vận tốc
            
        Returns:
            Số lần thay đổi hướng
        """
        if len(velocities) < 3:
            return 0
        
        direction_changes = 0
        angle_threshold = 45  # degrees
        
        for i in range(1, len(velocities) - 1):
            v1 = velocities[i-1]
            v2 = velocities[i+1]
            
            # Calculate angle between velocity vectors
            if (v1[0]**2 + v1[1]**2) > 0 and (v2[0]**2 + v2[1]**2) > 0:
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.degrees(math.acos(cos_angle))
                
                if angle > angle_threshold:
                    direction_changes += 1
        
        return direction_changes

    def _detect_ball_hits(self, velocities: List[Tuple[float, float]], 
                         speeds: List[float], timestamps: List[float]) -> List[Dict[str, Any]]:
        """
        Phát hiện các lần đánh bóng dựa trên thay đổi vận tốc đột ngột.
        
        Args:
            velocities: List vận tốc
            speeds: List tốc độ
            timestamps: List timestamp
            
        Returns:
            List các ball hits detected
        """
        if len(speeds) < 3:
            return []
        
        ball_hits = []
        speed_change_threshold = 2.0  # m/s
        
        for i in range(1, len(speeds) - 1):
            speed_before = speeds[i-1]
            speed_current = speeds[i]
            speed_after = speeds[i+1]
            
            # Detect sudden speed increase (hit)
            if (speed_current > speed_before + speed_change_threshold or
                speed_after > speed_current + speed_change_threshold):
                
                hit = {
                    'timestamp': timestamps[i],
                    'frame_index': i,
                    'speed_before': speed_before,
                    'speed_after': max(speed_current, speed_after),
                    'speed_change': max(speed_current - speed_before, speed_after - speed_current)
                }
                ball_hits.append(hit)
        
        return ball_hits

    def calculate_match_statistics(self, tracking_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Tính toán thống kê tổng thể của match.
        
        Args:
            tracking_df: DataFrame tracking data
            
        Returns:
            Dictionary thống kê match
        """
        stats = {
            'match_duration': 0,
            'total_players': 0,
            'total_balls': 0,
            'avg_players_on_court': 0,
            'ball_in_play_percentage': 0,
            'most_active_player': None,
            'rally_statistics': {}
        }
        
        if tracking_df.empty:
            return stats
        
        # Basic statistics
        stats['match_duration'] = tracking_df['timestamp'].max() - tracking_df['timestamp'].min()
        stats['total_players'] = len(tracking_df[tracking_df['class'] == 'player']['object_id'].unique())
        stats['total_balls'] = len(tracking_df[tracking_df['class'] == 'ball']['object_id'].unique())
        
        # Calculate average players on court per frame
        players_per_frame = tracking_df[tracking_df['class'] == 'player'].groupby('frame_id')['object_id'].nunique()
        stats['avg_players_on_court'] = players_per_frame.mean() if len(players_per_frame) > 0 else 0
        
        # Ball in play percentage
        frames_with_ball = tracking_df[tracking_df['class'] == 'ball']['frame_id'].nunique()
        total_frames = tracking_df['frame_id'].nunique()
        stats['ball_in_play_percentage'] = (frames_with_ball / total_frames * 100) if total_frames > 0 else 0
        
        # Most active player (player với track dài nhất)
        if stats['total_players'] > 0:
            player_track_lengths = tracking_df[tracking_df['class'] == 'player'].groupby('object_id').size()
            most_active_id = player_track_lengths.idxmax()
            stats['most_active_player'] = f"player_{most_active_id}"
        
        return stats

    def analyze_movement_patterns(self, tracking_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích patterns chuyển động tổng thể.
        
        Args:
            tracking_df: DataFrame tracking data
            
        Returns:
            Dictionary movement patterns
        """
        patterns = {
            'court_coverage': {},
            'interaction_zones': {},
            'movement_synchronization': {}
        }
        
        # Analyze court coverage by all players
        player_data = tracking_df[tracking_df['class'] == 'player']
        if not player_data.empty:
            all_positions = []
            for _, row in player_data.iterrows():
                x_m = row['center_x'] / 10.0
                y_m = row['center_y'] / 10.0
                all_positions.append((x_m, y_m))
            
            if all_positions:
                patterns['court_coverage'] = self._analyze_movement_zones(all_positions)
        
        # Analyze interaction zones (where players are close to each other)
        patterns['interaction_zones'] = self._analyze_player_interactions(tracking_df)
        
        return patterns

    def _analyze_player_interactions(self, tracking_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Phân tích tương tác giữa các players.
        
        Args:
            tracking_df: DataFrame tracking data
            
        Returns:
            Dictionary player interactions
        """
        interactions = {
            'close_encounters': 0,
            'avg_distance_between_players': 0,
            'interaction_zones': []
        }
        
        player_data = tracking_df[tracking_df['class'] == 'player']
        close_distance_threshold = 2.0  # meters
        
        # Group by frame to analyze interactions
        for frame_id in player_data['frame_id'].unique():
            frame_players = player_data[player_data['frame_id'] == frame_id]
            
            if len(frame_players) >= 2:
                positions = []
                for _, row in frame_players.iterrows():
                    x_m = row['center_x'] / 10.0
                    y_m = row['center_y'] / 10.0
                    positions.append((x_m, y_m))
                
                # Calculate distances between all pairs
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        distance = calculate_distance(positions[i], positions[j])
                        
                        if distance < close_distance_threshold:
                            interactions['close_encounters'] += 1
                            interactions['interaction_zones'].append({
                                'frame_id': frame_id,
                                'distance': distance,
                                'position1': positions[i],
                                'position2': positions[j]
                            })
        
        return interactions

    def export_analysis_results(self, analysis_results: Dict[str, Any], 
                               output_path: str) -> None:
        """
        Export kết quả phân tích thành file JSON.
        
        Args:
            analysis_results: Kết quả phân tích
            output_path: Đường dẫn file output
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(analysis_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Đã export analysis results vào {output_path}")

def main():
    """Test function cho analysis module."""
    from .utils import load_config, setup_logging
    
    # Load config
    config = load_config()
    setup_logging(config)
    
    # Test với sample data
    analyzer = MovementAnalyzer(config)
    
    # Create sample tracking data
    sample_data = pd.DataFrame({
        'frame_id': range(100),
        'timestamp': [i/30.0 for i in range(100)],  # 30 FPS
        'object_id': [1] * 100,  # Single player
        'class': ['player'] * 100,
        'center_x': [100 + i*2 for i in range(100)],  # Moving right
        'center_y': [100 + np.sin(i/10)*50 for i in range(100)],  # Sinusoidal movement
        'confidence': [0.9] * 100
    })
    
    # Analyze
    results = analyzer.analyze_tracking_data(sample_data)
    
    # Print results
    print("Analysis Results:")
    print(f"Player analysis keys: {list(results['player_analysis'].keys())}")
    
    if 'player_1' in results['player_analysis']:
        player_stats = results['player_analysis']['player_1']
        print(f"Total distance: {player_stats.get('total_distance_meters', 0):.2f} m")
        print(f"Average speed: {player_stats.get('avg_speed_kmh', 0):.2f} km/h")

if __name__ == "__main__":
    main()