#!/usr/bin/env python3
"""
IMPROVED monitor_pdu.py - with better signal handling and error reporting
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
import signal
import argparse

class PDUMonitor:
    def __init__(self, output_dir="gpu_stats_comparison"):
        self.running = False
        self.output_dir = output_dir
        self.stats = []
        self.run_name = None
        self.error_count = 0
        os.makedirs(output_dir, exist_ok=True)
        
    def get_pdu_stats(self):
        """Get all available PDU statistics using SNMP."""
        try:
            cmd = ['snmpget', '-v2c', '-c', 'public', '192.168.10.168', 
                   'PowerNet-MIB::ePDUPhaseStatusActivePower.1']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                self.error_count += 1
                if self.error_count <= 3:  # Only print first 3 errors
                    print(f"[PDU] SNMP error: {result.stderr.strip()}", file=sys.stderr)
                return None
                
            response = result.stdout.strip()
            if not response:
                print("[PDU] Empty SNMP response", file=sys.stderr)
                return None
                
            # Parse response: "PowerNet-MIB::ePDUPhaseStatusActivePower.1 = INTEGER: 73"
            value_str = response.split('=')[1].strip()
            power_value = float(value_str.replace('INTEGER:', '').strip())
            
            pdu_info = {
                'timestamp': datetime.now().isoformat(),
                'power_draw': power_value,
            }
            
            # Reset error count on success
            if self.error_count > 0:
                print(f"[PDU] Recovered after {self.error_count} errors", file=sys.stderr)
                self.error_count = 0
            
            return pdu_info
                
        except subprocess.TimeoutExpired:
            self.error_count += 1
            if self.error_count == 1:
                print("[PDU] SNMP timeout - PDU not responding", file=sys.stderr)
            return None
        except Exception as e:
            self.error_count += 1
            if self.error_count <= 3:
                print(f"[PDU] Exception: {e}", file=sys.stderr)
            return None

    def start_monitoring(self, run_name, interval=1, duration=None):
        """Start monitoring PDU stats."""
        self.running = True
        self.stats = []
        self.run_name = run_name
        start_time = time.time()
        
        print(f"[PDU] Starting monitoring: {run_name}", file=sys.stderr)
        
        # Test PDU connectivity first
        test_stats = self.get_pdu_stats()
        if test_stats is None:
            print("[PDU] WARNING: Initial PDU query failed - monitoring will continue but may not collect data", file=sys.stderr)
        else:
            print(f"[PDU] Initial reading: {test_stats['power_draw']}W", file=sys.stderr)
        
        try:
            while self.running:
                stats = self.get_pdu_stats()
                if stats:
                    self.stats.append(stats)
                
                if duration and (time.time() - start_time) >= duration:
                    print(f"[PDU] Duration {duration}s completed", file=sys.stderr)
                    break
                    
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("[PDU] Received interrupt signal", file=sys.stderr)
        except Exception as e:
            print(f"[PDU] Error during monitoring: {e}", file=sys.stderr)
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring and save stats to file."""
        self.running = False
        print(f"[PDU] Stopping... collected {len(self.stats)} samples", file=sys.stderr)
        
        if self.error_count > 0:
            print(f"[PDU] Total SNMP errors: {self.error_count}", file=sys.stderr)
        
        if self.stats and self.run_name:
            output_file = os.path.join(self.output_dir, f"{self.run_name}_pdu.json")
            try:
                with open(output_file, 'w') as f:
                    json.dump(self.stats, f, indent=2)
                print(f"PDU stats saved to {output_file}")
            except Exception as e:
                print(f"[PDU] Error saving file: {e}", file=sys.stderr)
        else:
            print(f"[PDU] No data to save (stats={len(self.stats)}, run_name={self.run_name})", file=sys.stderr)
            if len(self.stats) == 0:
                print("[PDU] HINT: Check if PDU is accessible and SNMP is configured", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Monitor PDU power usage and metrics")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="gpu_stats_comparison")
    parser.add_argument("--suffix", type=str, default="")
    
    args = parser.parse_args()
    
    monitor = PDUMonitor(output_dir=args.output_dir)
    
    # Set up signal handler to gracefully stop monitoring
    def signal_handler(signum, frame):
        print(f"[PDU] Received signal {signum}", file=sys.stderr)
        monitor.running = False  # Stop the monitoring loop
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitor.start_monitoring(args.run_name + args.suffix, args.interval, args.duration)

if __name__ == "__main__":
    main()
