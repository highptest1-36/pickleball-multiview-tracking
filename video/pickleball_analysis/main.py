"""
Pickleball Analysis - Main Entry Point
Phân tích video pickleball với AI tracking và court calibration
"""
import sys
import os
from enhanced_tracking_san4 import EnhancedTrackingSan4

def main():
    print("=" * 60)
    print(" PICKLEBALL ANALYSIS - San4 Video")
    print("=" * 60)
    print()
    
    if not os.path.exists('court_calibration_san4.json'):
        print(" ERROR: court_calibration_san4.json not found!")
        print()
        print(" Setup required:")
        print("   1. python multi_point_selector.py")
        print("   2. python net_selector.py")
        print("   3. python main.py")
        return
    
    print(" Starting tracking...")
    print("  Press 'q' to quit")
    print()
    
    try:
        tracker = EnhancedTrackingSan4()
        tracker.run()
    except KeyboardInterrupt:
        print("\n  Interrupted")
    except Exception as e:
        print(f"\n Error: {e}")
    finally:
        print("\n Complete!")

if __name__ == "__main__":
    main()
