import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load new tracking data
with open('tracking_data_san1.json', 'r') as f:
    tracking_data = json.load(f)

print(f"Total tracking records: {len(tracking_data)}")
print(f"Frame range: {min(r['frame'] for r in tracking_data)} - {max(r['frame'] for r in tracking_data)}")

# Group by frame
frames = {}
for record in tracking_data:
    frame_id = record['frame']
    if frame_id not in frames:
        frames[frame_id] = []
    frames[frame_id].append(record)

print(f"Total frames with tracking: {len(frames)}")

# Count detections per frame
detection_counts = {}
ball_detections = 0
person_detections = 0

for frame_id, detections in frames.items():
    person_count = len([d for d in detections if d['class'] == 'person'])
    ball_count = len([d for d in detections if d['class'] == 'sports ball'])
    
    detection_counts[frame_id] = {
        'persons': person_count,
        'balls': ball_count
    }
    
    person_detections += person_count
    ball_detections += ball_count

print(f"\nTotal person detections: {person_detections}")
print(f"Total ball detections: {ball_detections}")

# Show frame range statistics
frames_list = sorted(frames.keys())
print(f"First frame: {frames_list[0]}")
print(f"Last frame: {frames_list[-1]}")
print(f"Frame gaps: {len(range(frames_list[0], frames_list[-1]+1)) - len(frames_list)}")

# Player tracking analysis
players = {}
for record in tracking_data:
    if record['class'] == 'person':
        player_id = record['track_id']
        if player_id not in players:
            players[player_id] = []
        players[player_id].append(record)

print(f"\nUnique players tracked: {len(players)}")
for player_id, detections in players.items():
    print(f"Player {player_id}: {len(detections)} detections, frames {min(d['frame'] for d in detections)} - {max(d['frame'] for d in detections)}")

# Sample of recent data
print(f"\nSample from latest frames:")
recent_frames = sorted(frames.keys())[-5:]
for frame_id in recent_frames:
    detections = frames[frame_id]
    persons = [d for d in detections if d['class'] == 'person']
    balls = [d for d in detections if d['class'] == 'sports ball']
    print(f"Frame {frame_id}: {len(persons)} persons, {len(balls)} balls")
    for person in persons[:3]:  # Show first 3 persons
        print(f"  Player {person['track_id']}: ({person['x']:.1f}, {person['y']:.1f})")