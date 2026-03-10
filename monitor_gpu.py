#!/usr/bin/env python3
"""
IMPROVED monitor_gpu.py - with better signal handling and error reporting
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
import signal
import argparse

class GPUMonitor:
    def __init__(self, output_dir="gpu_stats"):
        self.running = False
        self.output_dir = output_dir
        self.stats = []
        self.run_name = None
        os.makedirs(output_dir, exist_ok=True)
        
    def get_gpu_stats(self):
        """Get all available GPU statistics using nvidia-smi."""
        try:
            query = 'timestamp,index,name,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,fan.speed,compute_mode,driver_version,pstate'
            
            cmd = ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                print(f"[GPU] Error: {result.stderr}", file=sys.stderr)
                return None
                
            stats = result.stdout.strip().split(',')
            
            if len(stats) < 18:
                print(f"[GPU] Unexpected fields: got {len(stats)}, expected 18", file=sys.stderr)
                return None
            
            gpu_stats = {
                'timestamp': datetime.now().isoformat(),
                'index': int(stats[1]),
                'name': stats[2],
                'power_draw': float(stats[3]),
                'power_limit': float(stats[4]),
                'gpu_utilization': float(stats[5]),
                'memory_utilization': float(stats[6]),
                'memory_used': float(stats[7]),
                'memory_total': float(stats[8]),
                'temperature': float(stats[9]),
                'graphics_clock': float(stats[10]),
                'memory_clock': float(stats[11]),
                'max_graphics_clock': float(stats[12]),
                'max_memory_clock': float(stats[13]),
                'fan_speed': float(stats[14]),
                'compute_mode': stats[15],
                'driver_version': stats[16],
                'pstate': stats[17]
            }
            return gpu_stats
            
        except Exception as e:
            print(f"[GPU] Exception: {e}", file=sys.stderr)
            return None

    def start_monitoring(self, run_name, interval=1, duration=None):
        """Start monitoring GPU stats."""
        self.running = True
        self.stats = []
        self.run_name = run_name
        start_time = time.time()
        
        print(f"[GPU] Starting monitoring: {run_name}", file=sys.stderr)
        
        try:
            while self.running:
                stats = self.get_gpu_stats()
                if stats:
                    self.stats.append(stats)
                
                if duration and (time.time() - start_time) >= duration:
                    print(f"[GPU] Duration {duration}s completed", file=sys.stderr)
                    break
                    
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("[GPU] Received interrupt signal", file=sys.stderr)
        except Exception as e:
            print(f"[GPU] Error during monitoring: {e}", file=sys.stderr)
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring and save stats to file."""
        self.running = False
        print(f"[GPU] Stopping... collected {len(self.stats)} samples", file=sys.stderr)
        
        if self.stats and self.run_name:
            output_file = os.path.join(self.output_dir, f"{self.run_name}_gpu.json")
            try:
                with open(output_file, 'w') as f:
                    json.dump(self.stats, f, indent=2)
                print(f"GPU stats saved to {output_file}")
            except Exception as e:
                print(f"[GPU] Error saving file: {e}", file=sys.stderr)
        else:
            print(f"[GPU] No data to save (stats={len(self.stats)}, run_name={self.run_name})", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Monitor GPU power usage and metrics")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="gpu_stats_comparison")
    parser.add_argument("--suffix", type=str, default="")
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(output_dir=args.output_dir)
    
    # Set up signal handler to gracefully stop monitoring
    def signal_handler(signum, frame):
        print(f"[GPU] Received signal {signum}", file=sys.stderr)
        monitor.running = False  # Stop the monitoring loop
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitor.start_monitoring(args.run_name + args.suffix, args.interval, args.duration)

if __name__ == "__main__":
    main()