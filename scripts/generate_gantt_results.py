#!/usr/bin/env python3
import os
import sys
# ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import MARL.common.plot_utils as plot_utils

def main():
    out_dir = os.path.join("result", "qmix", "boatschedule")
    os.makedirs(out_dir, exist_ok=True)
    print("Generating per-robot Gantt (minutes)...")
    p1 = plot_utils.generate_and_save_gantt(out_dir, out_filename="gantt.png", time_unit_minutes=5)
    print("Saved:", p1)
    print("Generating per-job Gantt (minutes)...")
    p2 = plot_utils.generate_and_save_per_job_gantt(out_dir, out_filename="gantt_per_job.png", time_unit_minutes=5)
    print("Saved:", p2)

if __name__ == '__main__':
    main()

