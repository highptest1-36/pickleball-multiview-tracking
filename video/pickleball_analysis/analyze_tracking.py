"""
Analyze tracking data để hiểu vấn đề
"""
import pandas as pd

# Load tracking data
df = pd.read_csv('real_demo_output/tracking_data/real_tracking.csv')

print("📊 TRACKING DATA ANALYSIS")
print("=" * 50)
print(f"Total records: {len(df)}")
print(f"Frame range: {df.frame_id.min()} - {df.frame_id.max()}")
print(f"Total frames with data: {len(df.frame_id.unique())}")
print()

# Analyze by class
print("🔍 OBJECTS BY CLASS:")
for class_name in df['class'].unique():
    count = len(df[df['class'] == class_name])
    unique_ids = df[df['class'] == class_name]['object_id'].unique()
    print(f"  {class_name}: {count} detections, IDs: {list(unique_ids)}")

print()

# Analyze frame distribution
frame_counts = df.groupby('frame_id').size()
print(f"📈 FRAME DISTRIBUTION:")
print(f"  Frames with 1-3 detections: {len(frame_counts[frame_counts <= 3])}")
print(f"  Frames with 4-6 detections: {len(frame_counts[(frame_counts > 3) & (frame_counts <= 6)])}")
print(f"  Frames with 7+ detections: {len(frame_counts[frame_counts > 6])}")
print(f"  Max detections in single frame: {frame_counts.max()}")

print()

# Sample frames
print("📋 SAMPLE FRAMES:")
sample_frames = [2, 10, 20, 29]
for frame_id in sample_frames:
    frame_data = df[df['frame_id'] == frame_id]
    if len(frame_data) > 0:
        print(f"  Frame {frame_id}: {len(frame_data)} objects")
        for _, row in frame_data.iterrows():
            print(f"    {row['class']}_{row['object_id']}: ({row['center_x']:.0f}, {row['center_y']:.0f})")
    else:
        print(f"  Frame {frame_id}: NO DATA")

print()
print("🎯 POTENTIAL ISSUES:")
if df.frame_id.max() < 100:
    print("  ❌ Only tracking first few seconds of video")
if len(df[df['class'].str.contains('player|person', na=False)]['object_id'].unique()) < 4:
    print("  ❌ Missing players (expected 4, found fewer)")
if len(df.frame_id.unique()) < 50:
    print("  ❌ Too few frames with tracking data")