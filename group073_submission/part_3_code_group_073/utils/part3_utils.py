import argparse
import json
import sys
import numpy as np
import pandas as pd
import dataclasses
from .convert_to_csv import convert_to_csv
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from pathlib import Path


@dataclasses.dataclass
class Part3PlotData:
    df: pd.DataFrame
    start_events: Dict[str, float]
    end_events: Dict[str, float]

def prepare_part3_plot_data(run: int):
    """
    Create 3 bar plots (one for each run) of memcached p95 latency (y-axis) over time (x-axis),
    with annotations showing when each batch job started and ended, also indicating the machine
    each of them is running on. Using the augmented version of mcperf, you get two additional
    columns in the output: ts start and ts end. Use them to determine the width of each bar
    in the bar plot, while the height should represent the p95 latency. 
    
    Align the x axis so that x = 0 coincides with the starting time of the first container. 
    """

    # Get the pod json file and the memcached log file
    pod_json_file = f"part3_results_group_073/pods_{run}.json"
    memcached_log_file = f"part3_results_group_073/mcperf_{run}.txt"

    # Get the start times and end times for all of the containers
    start_times, end_times = get_start_end_containers(pod_json_file)

    # Get the start time for the earliest container
    earliest = min(start_times)

    # Convert the memcached log file into a pandas DataFrame
    df = convert_to_csv(memcached_log_file, f"data/task3/mcperf_{run}.csv")

    # Trim x-axis

    # Create plot data



def get_start_end_containers(filename: str) -> Tuple[List[int], List[int]]:
    time_format = '%Y-%m-%dT%H:%M:%SZ'
    file = open(filename, 'r')
    json_file = json.load(file)

    start_times = []
    completion_times = []
    for item in json_file['items']:
        name = item['status']['containerStatuses'][0]['name']
        print("Job: ", str(name))
        if str(name) != "memcached":
            try:
                start_time = datetime.strptime(
                        item['status']['containerStatuses'][0]['state']['terminated']['startedAt'],
                        time_format)
                completion_time = datetime.strptime(
                        item['status']['containerStatuses'][0]['state']['terminated']['finishedAt'],
                        time_format)
                print("Job time: ", completion_time - start_time)
                start_times.append(start_time)
                completion_times.append(completion_time)
            except KeyError:
                print("Job {0} has not completed....".format(name))
                sys.exit(0)

    if len(start_times) != 7 and len(completion_times) != 7:
        print("You haven't run all the PARSEC jobs. Exiting...")
        sys.exit(0)

    print("Total time: {0}".format(max(completion_times) - min(start_times)))
    file.close()

    # Convert start and end times to timestamps
    start_times = [int(tim.timestamp()) for tim in start_times]
    completion_times = [int(tim.timestamp()) for tim in completion_times]

    return start_times, completion_times

def get_time(filename: str) -> Tuple[List[int], List[int]]:
    time_format = '%Y-%m-%dT%H:%M:%SZ'
    file = open(filename, 'r')
    json_file = json.load(file)

    start_times = []
    completion_times = []
    for item in json_file['items']:
        name = item['status']['containerStatuses'][0]['name']
        print("Job: ", str(name))
        if str(name) != "memcached":
            try:
                start_time = datetime.strptime(
                        item['status']['containerStatuses'][0]['state']['terminated']['startedAt'],
                        time_format)
                completion_time = datetime.strptime(
                        item['status']['containerStatuses'][0]['state']['terminated']['finishedAt'],
                        time_format)
                print("Job time: ", completion_time - start_time)
                start_times.append(start_time)
                completion_times.append(completion_time)
            except KeyError:
                print("Job {0} has not completed....".format(name))
                sys.exit(0)

    if len(start_times) != 7 and len(completion_times) != 7:
        print("You haven't run all the PARSEC jobs. Exiting...")
        sys.exit(0)

    print("Total time: {0}".format(max(completion_times) - min(start_times)))
    file.close()

def compute_slo_violation(df: pd.DataFrame, memcached_file: str):
    pass



def compute_mean_std(filename: str):
    benchmark_runs = defaultdict(list)
    current_job = ""

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Job:"):
            current_job = line.split(":")[1].strip()

        if line.startswith("Job time:") and current_job != "":
            current_time = line.split(":", maxsplit=1)[1].strip()
            h, m, s = map(int, current_time.split(":"))
            total_seconds = timedelta(hours=h, minutes=m, seconds=s).total_seconds()

            benchmark_runs[current_job].append(total_seconds)  # Append as seconds

    # Compute the mean and standard deviation for each benchmark
    for bench, runtimes in benchmark_runs.items():
        mean_time = np.mean(runtimes)
        std_time = np.std(runtimes)

        print(f"Benchmark: {bench}")
        print(f"Mean Runtime (s): {mean_time}")
        print(f"Std Dev Runtime (s): {std_time}")
        print("\n"*2)
    
    # Compute the mean and standard deviation for the whole runs
    for run in range(0, 3):
        run_runtimes = [timings[run] for bench, timings in benchmark_runs.items() if bench != "total"]
        total_runtime = max(run_runtimes) - min(run_runtimes)
        benchmark_runs["total"].append(total_runtime)

    total_mean_time = np.mean(benchmark_runs["total"])
    total_std_time = np.std(benchmark_runs["total"])

    print("Total:")
    print(f"Mean Runtime (s): {total_mean_time}")
    print(f"Std Dev Runtime (s): {total_std_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to the timings for part 3")
    args = parser.parse_args()
    compute_mean_std(args.file)
    compute_mean_std(args.file)
