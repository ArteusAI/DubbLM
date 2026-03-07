
import sys
import os
from typing import List, Dict

# Mock the logger
class Logger:
    def info(self, msg): print(f"INFO: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

logger = Logger()

# Verify logic of gap correction
def run_verification():
    print("Running verification for Gap Correction Logic...")
    
    # 1. Setup mock data
    # Scenario:
    # Seg 1: Orig 0-10s. Speed 2.0x -> Output duration 5s.
    # Gap: Orig 10-15s (5s duration).
    # Seg 2: Orig 15-20s. Target Start = 15s.
    # 
    # Without correction:
    # Output time after Seg 1 = 5s.
    # Gap duration (1.0x) = 5s.
    # Start of Seg 2 = 5 + 5 = 10s.
    # Target Start of Seg 2 = 15s.
    # Error = -5s (Seg 2 starts 5s early).
    #
    # With correction:
    # Output time after Seg 1 = 5s.
    # Target for Seg 2 = 15s.
    # Required Gap Duration = 15 - 5 = 10s.
    # Source Gap Duration = 5s.
    # Gap Speed = 5 / 10 = 0.5x.
    
    full_speed_segments = [
        # Segment 1
        {
            "start": 0.0, "end": 10.0, 
            "speed_factor": 2.0, 
            "is_gap": False,
            "video_sync_start": 0.0
        },
        # Gap
        {
            "start": 10.0, "end": 15.0, 
            "speed_factor": 1.0, 
            "is_gap": True
        },
        # Segment 2
        {
            "start": 15.0, "end": 20.0, 
            "speed_factor": 1.0, 
            "is_gap": False,
            "video_sync_start": 15.0  # Synced to original timeline
        }
    ]
    
    video_duration = 20.0
    
    # 2. Run the logic (copied/adapted from the implemention)
    current_output_time = 0.0
    
    print("\n--- Processing Segments ---")
    
    for i, seg in enumerate(full_speed_segments):
        start = seg["start"]
        end = seg["end"]
        speed = seg["speed_factor"]
        
        # --- CORRECTION LOGIC ---
        if seg.get("is_gap"):
            target_gap_end_time = end
            
            if i + 1 < len(full_speed_segments):
                next_seg = full_speed_segments[i+1]
                next_seg_start = next_seg.get('video_sync_start', next_seg.get('start'))
                target_gap_end_time = next_seg_start
            else:
                target_gap_end_time = video_duration
            
            required_gap_duration = target_gap_end_time - current_output_time
            source_gap_duration = end - start
            
            if source_gap_duration > 0.001:
                if required_gap_duration > 0.001:
                    gap_speed_factor = source_gap_duration / required_gap_duration
                    gap_speed_factor = max(0.01, min(100.0, gap_speed_factor))
                    
                    print(f"Gap Correction: source={source_gap_duration}, needed={required_gap_duration} -> speed={gap_speed_factor}")
                    speed = gap_speed_factor
                else:
                    print(f"Gap Skipped (Late)")
                    speed = 1000.0
        # ------------------------
        
        segment_source_duration = end - start
        segment_output_duration = segment_source_duration / speed
        
        print(f"Segment {i}: Type={'GAP' if seg.get('is_gap') else 'SEG'}, "
              f"speed={speed:.2f}, OUT duration={segment_output_duration:.2f}, "
              f"Cumulative End Time={current_output_time + segment_output_duration:.2f}")
        
        current_output_time += segment_output_duration

    # 3. Verify results
    # We expect Seg 2 to start exactly at 15.0s
    # So after segment 0 (end 5.0) and segment 1 (gap), time must be 15.0
    
    # Index 0 is Seg 1
    # Index 1 is Gap
    # Time after index 1 should be 15.0
    
    print("\n--- Verification ---")
    expected_time_at_seg2 = 15.0
    if abs(current_output_time - 20.0) < 0.01: # Total duration should also match correct total
         print("SUCCESS: Final duration matches target (20.0s)")
    else:
         print(f"FAILURE: Final duration {current_output_time} does not match target 20.0s")

if __name__ == "__main__":
    run_verification()
