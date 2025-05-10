import argparse
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from utils.convert_to_csv import convert_to_csv
from utils.part4_utils import get_memcached_startend, parse_logger_file


def compute_violation_ratio(df: pd.DataFrame, qps_interval: int, memcached_file: str, log_file: str) -> float:
    # Get start and end timestamp for memcached job
    start_mc, _ = get_memcached_startend(memcached_file)
    start_ts = datetime.fromtimestamp(start_mc / 1000, tz=timezone.utc)

    # Filter number of rows to only consider batch jobs running
    parsed_logs = parse_logger_file(log_file)
    final_endtime = max(parsed_logs["ends"].values())
    final_endtime_sec = (final_endtime - start_ts).total_seconds()

    num_rows = int(final_endtime_sec // qps_interval)
    df = df[:num_rows]

    mask = df["p95"] > 800
    print(mask.sum())
    return mask.sum() / float(len(df))

def runtime_mean_std(question: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Get all of the relevant logfiles
    run1 = f"logs/part4/Q{question}/log_RUN1.txt"
    run2 = f"logs/part4/Q{question}/log_RUN2.txt"
    run3 = f"logs/part4/Q{question}/log_RUN3.txt"

    # Parse the files and get the runtimes
    runtimes1 = parse_logger_file(run1)["runtimes"]
    runtimes2 = parse_logger_file(run2)["runtimes"]
    runtimes3 = parse_logger_file(run3)["runtimes"]

    # Prepare the mean and standard deviation results
    means = defaultdict(lambda: 0)
    stds = defaultdict(lambda: 0)
    for job in runtimes1.keys():
        job_runtimes = [runtimes1[job], runtimes2[job], runtimes3[job]]  # Runtimes in seconds

        # Compute mean and std deviation
        means[job] = np.mean(job_runtimes)
        stds[job] = np.std(job_runtimes, ddof=1)
        
    return means, stds

def main():
    # Convert the log file into a CSV and load it as a dataframe
    filename = args.mc_file.split("/")[-1].split(".")[0].strip()
    convert_to_csv(args.mc_file, f"{args.out}/{filename}.csv")
    df = pd.read_csv(f"{args.out}/{filename}.csv")

    # Compute violation ratio
    violation_ratio = compute_violation_ratio(df, args.qps_interval, args.mc_file, args.log_file)
    print(f"Violation Ratio: {violation_ratio:.4f}")

    # Compute runtime mean and standard deviation for each job
    means, stds = runtime_mean_std(args.question)
    for job in means.keys():
        print(f"{job} Mean Runtime: {means[job]:.3f} s")
        print(f"{job} Std Runtime: {stds[job]:.3f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_file", type=str, help="File path to the memcached log")
    parser.add_argument("--log_file", type=str, help="File path to the logger output file")
    parser.add_argument("--qps_interval", type=int, help="QPS interval used for this experiment")
    parser.add_argument("--question", type=int, help="Part 4 question we are computing", default=3)
    parser.add_argument(
        "--out",
        type=str,
        help="File path to the output folder where the converted CSV should live",
    )
    args = parser.parse_args()

    main()
