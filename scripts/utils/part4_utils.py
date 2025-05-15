import dataclasses
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import pandas as pd


@dataclasses.dataclass
class Part4PlotData:
    df: pd.DataFrame
    start_events: Dict[str, float]
    end_events: Dict[str, float]
    pause_events: Dict[str, List[float]]
    unpause_events: Dict[str, List[float]]


# Helper function to parse log timestamps and lines
def parse_line(line):
    parts = line.strip().split()
    timestamp = datetime.fromisoformat(parts[0]).replace(tzinfo=timezone.utc)
    event = parts[1]
    job = parts[2] if len(parts) > 2 else None
    args = parts[3] if len(parts) > 3 else None

    return timestamp, event, job, args


# Task 4.1 filename parser
def parse_file_name(filename: str):
    base_name = filename.split(".")[0]
    parts = base_name.split("_")
    thread_part = parts[0]
    thread_number = int(thread_part.replace("thread", ""))
    cpu_part = parts[1]
    cpu_number = int(cpu_part.replace("cpu", ""))

    return (thread_number, cpu_number)


def get_memcached_startend(filename: str) -> Tuple[int, int]:
    # Open the memcached file and read timestamps
    with open(filename, "r") as f:
        lines = f.readlines()

    start_time = int(lines[0].split(":")[1].strip())
    end_time = int(lines[1].split(":")[1].strip())

    return start_time, end_time


def parse_logger_file(filename: str) -> Dict[str, Dict]:
    # Read and process log
    with open(filename) as f:
        lines = f.readlines()

    # Data structures to track state
    job_start = {}  # Start time of job
    job_active_start = {}  # When the job was last unpaused
    job_active_time = defaultdict(lambda: 0)  # Total active runtime in seconds
    paused_jobs = set()

    job_start_times = defaultdict(lambda: 0)
    job_paused_times = defaultdict(list)
    job_unpaused_times = defaultdict(list)
    job_end_times = defaultdict(lambda: 0)
    memcached_cores = {1: [], 2: []}

    for line in lines:
        timestamp, event, job, args = parse_line(line)

        if event == "start":
            job_start[job] = timestamp
            job_active_start[job] = timestamp  # It starts in unpaused state

            # Log the time that the event was started at
            job_start_times[job] = timestamp

        elif event == "pause":
            if job in job_active_start:
                # Add active duration
                active_duration = (timestamp - job_active_start[job]).total_seconds()
                job_active_time[job] += active_duration
                paused_jobs.add(job)

            # Log the time that the event was paused at
            job_paused_times[job].append(timestamp)

        elif event == "unpause":
            job_active_start[job] = timestamp
            paused_jobs.discard(job)

            # Log the time that the event was unpaused at
            job_unpaused_times[job].append(timestamp)

        elif event == "end":
            if job in job_active_start:
                # Add remaining active duration until end
                active_duration = (timestamp - job_active_start[job]).total_seconds()
                job_active_time[job] += active_duration
                job_active_start.pop(job, None)

            # Log the time that the event ended at
            job_end_times[job] = timestamp

        elif event == "update_cores" and job == "memcached":
            if args == "[0]":
                memcached_cores[1].append(timestamp)
            else:
                memcached_cores[2].append(timestamp)

    return {
        "runtimes": job_active_time,
        "starts": job_start_times,
        "ends": job_end_times,
        "paused": job_paused_times,
        "unpaused": job_unpaused_times,
        "memcached_cores": memcached_cores,
    }


def prepare_part4_plot_data(
    parsed_logfile: Dict[str, Dict],
    memcached_results: pd.DataFrame,
    qps_interval: int,
    question: int,
    run: int,
) -> Part4PlotData:
    plot_df = pd.DataFrame()

    # Get the start and end timestamps
    start, end = get_memcached_startend(
        f"logs/part4/Q{question}/memcached_RUN{run}.txt"
    )
    start_ts = datetime.fromtimestamp(start / 1000, tz=timezone.utc)
    end_ts = datetime.fromtimestamp(end / 1000, tz=timezone.utc)

    # Create x (seconds) and y1 (QPS) axis data
    num_cycles = int((end_ts - start_ts).total_seconds() // qps_interval)
    seconds = [qps_interval * cycles for cycles in range(0, num_cycles)]
    ticks = [start_ts + timedelta(seconds=qps_interval * i) for i in range(num_cycles)]

    plot_df["seconds"] = seconds
    plot_df["qps"] = memcached_results["QPS"]
    plot_df["p95"] = memcached_results["p95"] / 1000  # Convert to ms

    # Combine switch events into list of tuples
    switches = []
    for cores, timestamps in parsed_logfile["memcached_cores"].items():
        for ts in timestamps:
            switches.append((ts, cores))
    switches.sort()

    # Create list of cores assigned to memcached for each tick
    memcached_cores = [1]
    current_cores = 1
    switch_idx = 0

    for tick in ticks[1:]:
        while switch_idx < len(switches) and switches[switch_idx][0] <= tick:
            current_cores = switches[switch_idx][1]
            switch_idx += 1

        memcached_cores.append(current_cores)

    plot_df["cores"] = memcached_cores

    # Convert time from timestamps to seconds since start
    start_events = {
        job: max((benchmark_start - start_ts).total_seconds(), 0)
        for job, benchmark_start in parsed_logfile["starts"].items()
    }

    end_events = {
        job: max((benchmark_end - start_ts).total_seconds(), 0)
        for job, benchmark_end in parsed_logfile["ends"].items()
    }

    pause_events = {
        job: [(pause_ts - start_ts).total_seconds() for pause_ts in bench_pauses]
        for job, bench_pauses in parsed_logfile["paused"].items()
    }

    unpause_events = {
        job: [(unpause_ts - start_ts).total_seconds() for unpause_ts in bench_unpauses]
        for job, bench_unpauses in parsed_logfile["unpaused"].items()
    }

    return Part4PlotData(plot_df, start_events, end_events, pause_events, unpause_events)
