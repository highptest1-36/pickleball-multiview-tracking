# View 3D Analysis Results
# Script để xem kết quả phân tích 3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import glob

def load_and_analyze_results():
    """Load và phân tích kết quả 3D tracking"""
    print("🔍 Đang phân tích kết quả 3D tracking...")
    
    # Load tracking data
    tracking_data = []
    physics_data = []
    
    try:
        with open('tracking_3d_results.pkl', 'rb') as f:
            tracking_data = pickle.load(f)
        print(f"✅ Loaded {len(tracking_data)} tracking frames")
    except:
        print("❌ Không tìm thấy tracking_3d_results.pkl")
    
    try:
        with open('physics_data.pkl', 'rb') as f:
            physics_data = pickle.load(f)
        print(f"✅ Loaded {len(physics_data)} physics records")
    except:
        print("❌ Không tìm thấy physics_data.pkl")
    
    # Check God View images
    god_view_images = glob.glob('god_view_frame_*.png')
    print(f"✅ Tìm thấy {len(god_view_images)} God View images")
    
    return tracking_data, physics_data, god_view_images

def analyze_tracking_stats(tracking_data):
    """Phân tích thống kê tracking"""
    if not tracking_data:
        print("❌ Không có dữ liệu tracking")
        return
    
    print("\n📊 THỐNG KÊ TRACKING:")
    print("=" * 40)
    
    person_counts = []
    ball_counts = []
    
    for frame_data in tracking_data:
        persons = len(frame_data['positions_3d'].get('persons', []))
        balls = len(frame_data['positions_3d'].get('balls', []))
        person_counts.append(persons)
        ball_counts.append(balls)
    
    print(f"📈 Tổng frames: {len(tracking_data)}")
    print(f"🏃 Người chơi:")
    print(f"   - Trung bình: {np.mean(person_counts):.1f} người/frame")
    print(f"   - Tối đa: {np.max(person_counts)} người")
    print(f"   - Tối thiểu: {np.min(person_counts)} người")
    
    print(f"⚽ Bóng:")
    print(f"   - Trung bình: {np.mean(ball_counts):.1f} bóng/frame")
    print(f"   - Tối đa: {np.max(ball_counts)} bóng")
    print(f"   - Frames có bóng: {sum(1 for x in ball_counts if x > 0)}")

def analyze_physics_stats(physics_data):
    """Phân tích thống kê vật lý"""
    if not physics_data:
        print("❌ Không có dữ liệu physics")
        return
    
    print("\n⚡ THỐNG KÊ VẬT LÝ BÓNG:")
    print("=" * 40)
    
    speeds = [p['speed'] for p in physics_data if 'speed' in p]
    heights = [p['height'] for p in physics_data if 'height' in p]
    
    if speeds:
        print(f"🚀 Vận tốc:")
        print(f"   - Trung bình: {np.mean(speeds):.2f} m/s ({np.mean(speeds)*3.6:.2f} km/h)")
        print(f"   - Tối đa: {np.max(speeds):.2f} m/s ({np.max(speeds)*3.6:.2f} km/h)")
        print(f"   - Tối thiểu: {np.min(speeds):.2f} m/s ({np.min(speeds)*3.6:.2f} km/h)")
    
    if heights:
        print(f"📏 Độ cao:")
        print(f"   - Trung bình: {np.mean(heights):.2f} m")
        print(f"   - Tối đa: {np.max(heights):.2f} m")
        print(f"   - Tối thiểu: {np.min(heights):.2f} m")

def create_god_view_summary(tracking_data):
    """Tạo God View tổng hợp"""
    if not tracking_data:
        return
    
    print("\n🎯 Tạo God View tổng hợp...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Court dimensions
    court_length = 6.10  # meters
    court_width = 4.27   # meters
    
    # 2D Top-down view
    ax1.set_xlim(-0.5, 6.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_aspect('equal')
    ax1.set_title('God View - All Player Positions', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Length (m)')
    ax1.set_ylabel('Width (m)')
    
    # Vẽ sân
    court_corners = np.array([[0, 0], [court_length, 0], [court_length, court_width], [0, court_width], [0, 0]])
    ax1.plot(court_corners[:, 0], court_corners[:, 1], 'k-', linewidth=3, label='Court Boundary')
    
    # Vẽ net
    net_x = court_length / 2
    ax1.plot([net_x, net_x], [0, court_width], 'r-', linewidth=4, label='Net')
    
    # Vẽ các khu vực sân
    # Service areas
    ax1.plot([0, court_length], [court_width/4, court_width/4], 'b--', alpha=0.5, linewidth=1)
    ax1.plot([0, court_length], [3*court_width/4, 3*court_width/4], 'b--', alpha=0.5, linewidth=1)
    
    # Non-volley zone (kitchen)
    kitchen_depth = 2.13  # 7 feet
    ax1.fill_between([net_x-kitchen_depth/2, net_x+kitchen_depth/2], 0, court_width, alpha=0.2, color='yellow', label='Non-Volley Zone')
    
    # Collect positions
    all_person_positions = []
    all_ball_positions = []
    
    for frame_data in tracking_data:
        positions_3d = frame_data['positions_3d']
        
        # Persons
        if 'persons' in positions_3d:
            for person in positions_3d['persons']:
                pos = person['position_3d']
                if 0 <= pos[0] <= court_length and 0 <= pos[1] <= court_width:
                    all_person_positions.append([pos[0], pos[1]])
        
        # Balls
        if 'balls' in positions_3d:
            for ball in positions_3d['balls']:
                pos = ball['position_3d']
                if 0 <= pos[0] <= court_length and 0 <= pos[1] <= court_width:
                    all_ball_positions.append([pos[0], pos[1]])
    
    # Plot positions
    if all_person_positions:
        person_pos = np.array(all_person_positions)
        ax1.scatter(person_pos[:, 0], person_pos[:, 1], c='blue', s=15, alpha=0.6, label=f'Players ({len(person_pos)} positions)')
    
    if all_ball_positions:
        ball_pos = np.array(all_ball_positions)
        ax1.scatter(ball_pos[:, 0], ball_pos[:, 1], c='red', s=25, alpha=0.8, label=f'Ball ({len(ball_pos)} positions)')
    
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
    ax1.grid(True, alpha=0.3)
    
    # Heat map
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(-0.5, 4.5)
    ax2.set_aspect('equal')
    ax2.set_title('Player Activity Heat Map', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Length (m)')
    ax2.set_ylabel('Width (m)')
    
    # Vẽ sân
    ax2.plot(court_corners[:, 0], court_corners[:, 1], 'k-', linewidth=3)
    ax2.plot([net_x, net_x], [0, court_width], 'r-', linewidth=4)
    ax2.fill_between([net_x-kitchen_depth/2, net_x+kitchen_depth/2], 0, court_width, alpha=0.2, color='yellow')
    
    # Heat map
    if all_person_positions:
        person_pos = np.array(all_person_positions)
        hb = ax2.hexbin(person_pos[:, 0], person_pos[:, 1], gridsize=15, cmap='Blues', alpha=0.7)
        cb = plt.colorbar(hb, ax=ax2)
        cb.set_label('Player Density')
    
    plt.tight_layout()
    plt.savefig('god_view_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: god_view_analysis.png")
    plt.show()

def create_3d_court_view(tracking_data):
    """Tạo 3D visualization"""
    if not tracking_data:
        return
    
    print("\n📐 Tạo 3D Court Visualization...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Court dimensions
    court_length = 6.10
    court_width = 4.27
    net_height = 0.91
    
    # Vẽ sân 3D
    court_corners = np.array([[0, 0, 0], [court_length, 0, 0], [court_length, court_width, 0], [0, court_width, 0]])
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([court_corners[i, 0], court_corners[j, 0]], 
               [court_corners[i, 1], court_corners[j, 1]], 
               [court_corners[i, 2], court_corners[j, 2]], 'k-', linewidth=2)
    
    # Vẽ net 3D
    net_x = court_length / 2
    ax.plot([net_x, net_x], [0, 0], [0, net_height], 'r-', linewidth=3)
    ax.plot([net_x, net_x], [0, court_width], [net_height, net_height], 'r-', linewidth=3)
    ax.plot([net_x, net_x], [court_width, court_width], [net_height, 0], 'r-', linewidth=3)
    
    # Collect 3D positions
    ball_trajectory = []
    person_positions = []
    
    for frame_data in tracking_data:
        positions_3d = frame_data['positions_3d']
        
        if 'balls' in positions_3d and len(positions_3d['balls']) > 0:
            ball_pos = positions_3d['balls'][0]['position_3d']
            if (0 <= ball_pos[0] <= court_length and 
                0 <= ball_pos[1] <= court_width and 
                0 <= ball_pos[2] <= 3):
                ball_trajectory.append(ball_pos)
        
        if 'persons' in positions_3d:
            for person in positions_3d['persons']:
                pos = person['position_3d']
                if (0 <= pos[0] <= court_length and 0 <= pos[1] <= court_width):
                    person_positions.append(pos)
    
    # Plot 3D data
    if ball_trajectory:
        ball_traj = np.array(ball_trajectory)
        ax.plot(ball_traj[:, 0], ball_traj[:, 1], ball_traj[:, 2], 'r-', linewidth=2, alpha=0.8, label='Ball Trajectory')
        ax.scatter(ball_traj[:, 0], ball_traj[:, 1], ball_traj[:, 2], c='red', s=20, alpha=0.6)
    
    if person_positions:
        person_pos = np.array(person_positions)
        ax.scatter(person_pos[:, 0], person_pos[:, 1], person_pos[:, 2], c='blue', s=8, alpha=0.4, label='Players')
    
    ax.set_xlim(0, court_length)
    ax.set_ylim(0, court_width)
    ax.set_zlim(0, 2.5)
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')
    ax.set_zlabel('Height (m)')
    ax.set_title('3D Pickleball Court Analysis', fontsize=16, fontweight='bold')
    ax.legend()
    
    plt.savefig('3d_court_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: 3d_court_analysis.png")
    plt.show()

def main():
    """Main function"""
    print("🏓 PICKLEBALL 3D ANALYSIS VIEWER")
    print("=" * 50)
    
    # Load results
    tracking_data, physics_data, god_view_images = load_and_analyze_results()
    
    # Analyze stats
    analyze_tracking_stats(tracking_data)
    analyze_physics_stats(physics_data)
    
    # Create visualizations
    if tracking_data:
        create_god_view_summary(tracking_data)
        create_3d_court_view(tracking_data)
    
    # List output files
    print(f"\n📁 OUTPUT FILES:")
    print("=" * 30)
    files_to_check = [
        'god_view_analysis.png',
        '3d_court_analysis.png',
        'tracking_3d_results.pkl',
        'physics_data.pkl'
    ]
    
    for filename in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024  # KB
            print(f"✅ {filename} ({size:.1f} KB)")
        else:
            print(f"❌ {filename} (không tìm thấy)")
    
    print(f"\n🎉 PHÂN TÍCH HOÀN TẤT!")
    print(f"💡 Mở các file .png để xem kết quả visualization")

if __name__ == "__main__":
    main()