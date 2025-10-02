"""
Visualization Module

Module này tạo ra:
1. Heatmaps cho movement patterns
2. Trajectory plots
3. Speed/velocity charts
4. Court visualization với annotations
5. Video output với tracking overlays
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .utils import Timer

class PickleballVisualizer:
    """Class chính để tạo các visualization cho pickleball analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Khởi tạo PickleballVisualizer.
        
        Args:
            config: Cấu hình từ config.yaml
        """
        self.config = config
        self.viz_config = config['visualization']
        self.court_config = config['court']
        
        # Court dimensions
        self.court_width_m = self.court_config['width_meters']
        self.court_height_m = self.court_config['height_meters']
        
        # Visualization settings
        self.colors = self.viz_config['colors']
        self.dpi = self.viz_config['charts']['dpi']
        self.figsize = tuple(self.viz_config['charts']['figsize'])
        
        # Setup matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        logger.info("PickleballVisualizer đã được khởi tạo")

    def create_court_background(self, ax: plt.Axes, show_zones: bool = True) -> plt.Axes:
        """
        Tạo background sân pickleball.
        
        Args:
            ax: Matplotlib axes
            show_zones: Hiển thị các zones của sân
            
        Returns:
            Axes với court background
        """
        # Court boundary
        court_rect = patches.Rectangle(
            (0, 0), self.court_width_m, self.court_height_m,
            linewidth=3, edgecolor='black', facecolor='lightgreen', alpha=0.3
        )
        ax.add_patch(court_rect)
        
        # Net
        net_y = self.court_height_m / 2
        ax.plot([0, self.court_width_m], [net_y, net_y], 
               'r-', linewidth=4, label='Net')
        
        if show_zones:
            # Service areas
            service_line_distance = 1.83  # meters from net
            
            # Service lines
            ax.plot([0, self.court_width_m], [net_y - service_line_distance, net_y - service_line_distance], 
                   'b--', alpha=0.7, linewidth=2)
            ax.plot([0, self.court_width_m], [net_y + service_line_distance, net_y + service_line_distance], 
                   'b--', alpha=0.7, linewidth=2)
            
            # Center line
            center_x = self.court_width_m / 2
            ax.plot([center_x, center_x], [0, self.court_height_m], 
                   'b--', alpha=0.7, linewidth=2)
            
            # Non-volley zone (Kitchen)
            kitchen_distance = 2.13  # meters from net
            kitchen1 = patches.Rectangle(
                (0, net_y - kitchen_distance), self.court_width_m, kitchen_distance,
                linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.2
            )
            kitchen2 = patches.Rectangle(
                (0, net_y), self.court_width_m, kitchen_distance,
                linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.2
            )
            ax.add_patch(kitchen1)
            ax.add_patch(kitchen2)
        
        # Labels
        ax.text(self.court_width_m/2, -0.5, 'Pickleball Court', 
               ha='center', va='top', fontsize=14, fontweight='bold')
        
        ax.set_xlim(-1, self.court_width_m + 1)
        ax.set_ylim(-1, self.court_height_m + 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Width (meters)')
        ax.set_ylabel('Height (meters)')
        
        return ax

    def create_heatmap(self, positions: List[Tuple[float, float]], 
                      title: str = "Movement Heatmap",
                      output_path: Optional[str] = None) -> plt.Figure:
        """
        Tạo heatmap cho movement patterns.
        
        Args:
            positions: List vị trí [(x, y), ...]
            title: Tiêu đề heatmap
            output_path: Đường dẫn lưu file
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Tạo heatmap với {len(positions)} điểm")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create court background
        self.create_court_background(ax)
        
        if positions:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Create 2D histogram
            bins = self.viz_config['heatmap']['bins']
            heatmap, xedges, yedges = np.histogram2d(
                x_coords, y_coords, 
                bins=bins,
                range=[[0, self.court_width_m], [0, self.court_height_m]]
            )
            
            # Plot heatmap
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                          cmap=self.viz_config['heatmap']['colormap'],
                          alpha=self.viz_config['heatmap']['alpha'],
                          interpolation='gaussian')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Movement Density', rotation=270, labelpad=20)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Đã lưu heatmap: {output_path}")
        
        return fig

    def create_trajectory_plot(self, trajectories: Dict[str, List[Tuple[float, float]]], 
                              title: str = "Player Trajectories",
                              output_path: Optional[str] = None) -> plt.Figure:
        """
        Tạo trajectory plot cho multiple players.
        
        Args:
            trajectories: Dict {player_id: [(x, y), ...]}
            title: Tiêu đề plot
            output_path: Đường dẫn lưu file
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Tạo trajectory plot cho {len(trajectories)} players")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create court background
        self.create_court_background(ax)
        
        # Colors for different players
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (player_id, positions) in enumerate(trajectories.items()):
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                color = colors[i % len(colors)]
                
                # Plot trajectory
                ax.plot(x_coords, y_coords, color=color, linewidth=2, 
                       alpha=0.8, label=player_id)
                
                # Mark start and end points
                ax.scatter(x_coords[0], y_coords[0], color=color, s=100, 
                          marker='o', edgecolor='black', linewidth=2, 
                          label=f'{player_id} Start')
                ax.scatter(x_coords[-1], y_coords[-1], color=color, s=100, 
                          marker='s', edgecolor='black', linewidth=2,
                          label=f'{player_id} End')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Đã lưu trajectory plot: {output_path}")
        
        return fig

    def create_speed_analysis_chart(self, analysis_data: Dict[str, Any],
                                   output_path: Optional[str] = None) -> plt.Figure:
        """
        Tạo biểu đồ phân tích tốc độ.
        
        Args:
            analysis_data: Dữ liệu phân tích từ MovementAnalyzer
            output_path: Đường dẫn lưu file
            
        Returns:
            Matplotlib figure
        """
        logger.info("Tạo speed analysis chart")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        fig.suptitle('Speed Analysis', fontsize=16, fontweight='bold')
        
        # Extract player data
        player_data = analysis_data.get('player_analysis', {})
        
        if not player_data:
            # Create empty plot if no data
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
            return fig
        
        # Plot 1: Speed over time for all players
        ax1 = axes[0, 0]
        for player_id, data in player_data.items():
            if 'speeds' in data and 'timestamps' in data:
                timestamps = data['timestamps']
                speeds_kmh = [s * 3.6 for s in data['speeds']]  # Convert to km/h
                ax1.plot(timestamps, speeds_kmh, label=player_id, linewidth=2)
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_title('Speed Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speed distribution
        ax2 = axes[0, 1]
        all_speeds = []
        labels = []
        for player_id, data in player_data.items():
            if 'speeds' in data:
                speeds_kmh = [s * 3.6 for s in data['speeds']]
                all_speeds.append(speeds_kmh)
                labels.append(player_id)
        
        if all_speeds:
            ax2.boxplot(all_speeds, labels=labels)
            ax2.set_ylabel('Speed (km/h)')
            ax2.set_title('Speed Distribution by Player')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Running pace comparison
        ax3 = axes[1, 0]
        players = []
        paces = []
        for player_id, data in player_data.items():
            if 'running_pace_min_per_km' in data:
                players.append(player_id)
                paces.append(data['running_pace_min_per_km'])
        
        if players and paces:
            bars = ax3.bar(players, paces, color=sns.color_palette("husl", len(players)))
            ax3.set_ylabel('Pace (min/km)')
            ax3.set_title('Running Pace Comparison')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, pace in zip(bars, paces):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{pace:.1f}', ha='center', va='bottom')
        
        # Plot 4: Distance covered
        ax4 = axes[1, 1]
        players = []
        distances = []
        for player_id, data in player_data.items():
            if 'total_distance_meters' in data:
                players.append(player_id)
                distances.append(data['total_distance_meters'])
        
        if players and distances:
            bars = ax4.bar(players, distances, color=sns.color_palette("viridis", len(players)))
            ax4.set_ylabel('Distance (meters)')
            ax4.set_title('Total Distance Covered')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, distance in zip(bars, distances):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{distance:.0f}m', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Đã lưu speed analysis chart: {output_path}")
        
        return fig

    def create_interactive_dashboard(self, analysis_data: Dict[str, Any],
                                   output_path: Optional[str] = None) -> str:
        """
        Tạo interactive dashboard với Plotly.
        
        Args:
            analysis_data: Dữ liệu phân tích
            output_path: Đường dẫn lưu file HTML
            
        Returns:
            HTML string hoặc đường dẫn file
        """
        logger.info("Tạo interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Court Heatmap', 'Speed Over Time', 
                          'Player Statistics', 'Ball Analysis'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        player_data = analysis_data.get('player_analysis', {})
        
        # Plot 1: Court positions (heatmap-like scatter)
        all_positions = []
        all_player_ids = []
        for player_id, data in player_data.items():
            if 'positions' in data:
                for pos in data['positions']:
                    all_positions.append(pos)
                    all_player_ids.append(player_id)
        
        if all_positions:
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            
            fig.add_trace(
                go.Scattergl(
                    x=x_coords, y=y_coords,
                    mode='markers',
                    marker=dict(size=3, opacity=0.6, color=list(range(len(x_coords))),
                              colorscale='Viridis'),
                    text=all_player_ids,
                    name='Positions'
                ),
                row=1, col=1
            )
        
        # Plot 2: Speed over time
        for i, (player_id, data) in enumerate(player_data.items()):
            if 'speeds' in data and 'timestamps' in data:
                speeds_kmh = [s * 3.6 for s in data['speeds']]
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamps'], y=speeds_kmh,
                        mode='lines',
                        name=f'{player_id} Speed',
                        line=dict(width=2)
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Player statistics bars
        players = []
        distances = []
        speeds = []
        for player_id, data in player_data.items():
            players.append(player_id)
            distances.append(data.get('total_distance_meters', 0))
            speeds.append(data.get('avg_speed_kmh', 0))
        
        if players:
            fig.add_trace(
                go.Bar(
                    x=players, y=distances,
                    name='Distance (m)',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # Plot 4: Ball analysis
        ball_data = analysis_data.get('ball_analysis', {})
        if ball_data:
            ball_speeds = []
            ball_times = []
            for ball_id, data in ball_data.items():
                if 'speeds' in data and 'timestamps' in data:
                    ball_speeds.extend([s * 3.6 for s in data['speeds']])
                    ball_times.extend(data['timestamps'])
            
            if ball_speeds and ball_times:
                fig.add_trace(
                    go.Scatter(
                        x=ball_times, y=ball_speeds,
                        mode='lines+markers',
                        name='Ball Speed',
                        line=dict(color='red', width=2)
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Pickleball Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Court Width (m)", row=1, col=1)
        fig.update_yaxes(title_text="Court Height (m)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Speed (km/h)", row=1, col=2)
        fig.update_xaxes(title_text="Player", row=2, col=1)
        fig.update_yaxes(title_text="Distance (m)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Ball Speed (km/h)", row=2, col=2)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Đã lưu interactive dashboard: {output_path}")
            return output_path
        else:
            return fig.to_html()

    def create_match_summary_report(self, analysis_data: Dict[str, Any],
                                   output_path: Optional[str] = None) -> plt.Figure:
        """
        Tạo báo cáo tổng kết match.
        
        Args:
            analysis_data: Dữ liệu phân tích
            output_path: Đường dẫn lưu file
            
        Returns:
            Matplotlib figure
        """
        logger.info("Tạo match summary report")
        
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Pickleball Match Analysis Report', fontsize=20, fontweight='bold', y=0.95)
        
        # Extract data
        player_data = analysis_data.get('player_analysis', {})
        match_stats = analysis_data.get('match_statistics', {})
        
        # Plot 1: Court heatmap (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        all_positions = []
        for data in player_data.values():
            if 'positions' in data:
                all_positions.extend(data['positions'])
        
        if all_positions:
            x_coords = [pos[0] for pos in all_positions]
            y_coords = [pos[1] for pos in all_positions]
            ax1.hexbin(x_coords, y_coords, gridsize=20, cmap='YlOrRd', alpha=0.7)
        
        self.create_court_background(ax1, show_zones=False)
        ax1.set_title('Movement Heatmap', fontweight='bold')
        
        # Plot 2: Player comparison (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        players = list(player_data.keys())
        distances = [player_data[p].get('total_distance_meters', 0) for p in players]
        speeds = [player_data[p].get('avg_speed_kmh', 0) for p in players]
        
        if players:
            x_pos = np.arange(len(players))
            width = 0.35
            
            ax2_twin = ax2.twinx()
            bars1 = ax2.bar(x_pos - width/2, distances, width, label='Distance (m)', alpha=0.8)
            bars2 = ax2_twin.bar(x_pos + width/2, speeds, width, label='Avg Speed (km/h)', 
                               alpha=0.8, color='orange')
            
            ax2.set_xlabel('Players')
            ax2.set_ylabel('Distance (m)', color='blue')
            ax2_twin.set_ylabel('Speed (km/h)', color='orange')
            ax2.set_title('Player Performance Comparison', fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(players, rotation=45)
        
        # Plot 3: Match statistics (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        stats_text = "Match Statistics\n\n"
        duration = match_stats.get('match_duration', 0)
        stats_text += f"Duration: {duration:.1f} seconds\n"
        stats_text += f"Total Players: {match_stats.get('total_players', 0)}\n"
        stats_text += f"Avg Players on Court: {match_stats.get('avg_players_on_court', 0):.1f}\n"
        stats_text += f"Ball in Play: {match_stats.get('ball_in_play_percentage', 0):.1f}%\n"
        
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Plot 4: Speed timeline (middle, span 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        for player_id, data in player_data.items():
            if 'speeds' in data and 'timestamps' in data:
                speeds_kmh = [s * 3.6 for s in data['speeds']]
                ax4.plot(data['timestamps'], speeds_kmh, label=player_id, linewidth=2)
        
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Speed (km/h)')
        ax4.set_title('Speed Over Time', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Movement zones (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Aggregate movement zones from all players
        zone_totals = {'left_court': 0, 'right_court': 0, 'front_court': 0, 'back_court': 0}
        player_count = 0
        
        for data in player_data.values():
            if 'movement_zones' in data:
                zones = data['movement_zones']
                for zone in zone_totals:
                    if zone in zones:
                        zone_totals[zone] += zones[zone]
                player_count += 1
        
        if player_count > 0:
            zone_averages = {k: v/player_count for k, v in zone_totals.items()}
            zones = list(zone_averages.keys())
            values = list(zone_averages.values())
            
            ax5.pie(values, labels=zones, autopct='%1.1f%%', startangle=90)
            ax5.set_title('Court Zone Usage', fontweight='bold')
        
        # Plot 6: Distance comparison (bottom)
        ax6 = fig.add_subplot(gs[2, :])
        
        if players and distances:
            colors = plt.cm.Set3(np.linspace(0, 1, len(players)))
            bars = ax6.bar(players, distances, color=colors)
            
            ax6.set_ylabel('Total Distance (meters)')
            ax6.set_title('Total Distance Covered by Each Player', fontweight='bold')
            ax6.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, distance in zip(bars, distances):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + max(distances)*0.01,
                        f'{distance:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Đã lưu match summary report: {output_path}")
        
        return fig

    def save_all_visualizations(self, analysis_data: Dict[str, Any], 
                               output_dir: str) -> Dict[str, str]:
        """
        Tạo và lưu tất cả visualizations.
        
        Args:
            analysis_data: Dữ liệu phân tích
            output_dir: Thư mục output
            
        Returns:
            Dictionary chứa đường dẫn các file đã tạo
        """
        logger.info(f"Tạo tất cả visualizations vào {output_dir}")
        
        output_files = {}
        
        with Timer("Creating all visualizations"):
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Individual player heatmaps
            player_data = analysis_data.get('player_analysis', {})
            for player_id, data in player_data.items():
                if 'positions' in data:
                    heatmap_path = os.path.join(output_dir, f'heatmap_{player_id}.png')
                    self.create_heatmap(data['positions'], 
                                      title=f'{player_id} Movement Heatmap',
                                      output_path=heatmap_path)
                    output_files[f'heatmap_{player_id}'] = heatmap_path
            
            # 2. Combined trajectory plot
            trajectories = {}
            for player_id, data in player_data.items():
                if 'positions' in data:
                    trajectories[player_id] = data['positions']
            
            if trajectories:
                trajectory_path = os.path.join(output_dir, 'trajectories.png')
                self.create_trajectory_plot(trajectories, output_path=trajectory_path)
                output_files['trajectories'] = trajectory_path
            
            # 3. Speed analysis chart
            speed_chart_path = os.path.join(output_dir, 'speed_analysis.png')
            self.create_speed_analysis_chart(analysis_data, output_path=speed_chart_path)
            output_files['speed_analysis'] = speed_chart_path
            
            # 4. Interactive dashboard
            dashboard_path = os.path.join(output_dir, 'interactive_dashboard.html')
            self.create_interactive_dashboard(analysis_data, output_path=dashboard_path)
            output_files['dashboard'] = dashboard_path
            
            # 5. Match summary report
            summary_path = os.path.join(output_dir, 'match_summary_report.png')
            self.create_match_summary_report(analysis_data, output_path=summary_path)
            output_files['summary_report'] = summary_path
        
        logger.info(f"Đã tạo {len(output_files)} visualizations")
        return output_files

def main():
    """Test function cho visualization module."""
    from .utils import load_config, setup_logging
    import random
    
    # Load config
    config = load_config()
    setup_logging(config)
    
    # Create visualizer
    visualizer = PickleballVisualizer(config)
    
    # Create sample data
    sample_analysis = {
        'player_analysis': {
            'player_1': {
                'positions': [(random.uniform(0, 13.41), random.uniform(0, 6.1)) for _ in range(100)],
                'speeds': [random.uniform(0, 10) for _ in range(100)],
                'timestamps': [i/30.0 for i in range(100)],
                'total_distance_meters': 150.5,
                'avg_speed_kmh': 8.2,
                'running_pace_min_per_km': 7.3,
                'movement_zones': {'left_court': 60, 'right_court': 40, 'front_court': 55, 'back_court': 45}
            },
            'player_2': {
                'positions': [(random.uniform(0, 13.41), random.uniform(0, 6.1)) for _ in range(80)],
                'speeds': [random.uniform(0, 8) for _ in range(80)],
                'timestamps': [i/30.0 for i in range(80)],
                'total_distance_meters': 120.3,
                'avg_speed_kmh': 6.8,
                'running_pace_min_per_km': 8.8,
                'movement_zones': {'left_court': 45, 'right_court': 55, 'front_court': 50, 'back_court': 50}
            }
        },
        'match_statistics': {
            'match_duration': 180.0,
            'total_players': 2,
            'avg_players_on_court': 1.8,
            'ball_in_play_percentage': 75.0
        }
    }
    
    # Create test visualizations
    output_dir = "output/test_visualizations"
    output_files = visualizer.save_all_visualizations(sample_analysis, output_dir)
    
    print("Created visualizations:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")

if __name__ == "__main__":
    main()