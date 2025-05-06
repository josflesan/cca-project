import psutil
import time
import csv
import argparse
import signal
import sys

# Global file object to close on exit
output_file = None
csv_writer = None

def get_cpu_info():
    # Get overall CPU percentage
    cpu_percent = psutil.cpu_percent(interval=1)
    # Get per-core CPU percentage
    per_cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    
    return {
        'timestamp': time.time(),
        'overall_cpu': cpu_percent,
        'per_cpu': per_cpu_percent,
    }

def signal_handler(sig, frame):
    """Handle CTRL+C and other termination signals"""
    print('\nExiting gracefully...')
    if output_file is not None and not output_file.closed:
        print(f'Closing output file: {output_file.name}')
        output_file.close()
    sys.exit(0)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Monitor CPU utilization and save to CSV file.')
    parser.add_argument('-o', '--output', type=str, default='cpu_stats.csv',
                      help='Output CSV filename (default: cpu_stats.csv)')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                      help='Sampling interval in seconds (default: 1.0)')
    return parser.parse_args()

def main():
    global output_file, csv_writer
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Register signal handlers for graceful exit
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    try:
        output_file = open(args.output, "w", newline='')
        fieldnames = ["timestamp", "overall_cpu", "per_cpu"]
        csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        print(f"Recording CPU statistics to {args.output}")
        print("Press Ctrl+C to stop recording")
        
        while True:
            row = get_cpu_info()
            csv_writer.writerow(row)
            time.sleep(args.interval)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if output_file is not None and not output_file.closed:
            output_file.close()
            print(f"Output file closed: {args.output}")

if __name__ == "__main__":
    main()
