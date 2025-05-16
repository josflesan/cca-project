import argparse
from pathlib import Path
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import numpy as np
from utils.convert_to_csv import convert_to_csv
import statistics

COLOUR_PALETTE = {
    "blackscholes": "#CCA000",
    "canneal": "#CCCCAA",
    "dedup": "#CCACCA",
    "ferret": "#AACCCA",
    "freqmine": "#0CCA00",
    "radix": "#00CCA0",
    "vips": "#CC0A00",
}
nodes = {
    'blackscholes': 'node-b-2core',
    'canneal': 'node-c-4core',
    'dedup': 'node-a-2core',
    'ferret': 'node-c-4core',
    'freqmine': 'node-d-4core',
    'radix': 'node-d-4core',
    'vips': 'node-a-2core'
}
colours = sns.color_palette("husl", 2)

def timestamp_to_milliseconds(timestamp_str):
    # Parse the ISO format timestamp
    d = dt.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    
    # Convert to seconds since epoch as a float
    seconds_since_epoch = d.timestamp()
    
    # Convert to milliseconds (multiply by 1000) and truncate to integer
    milliseconds = int(seconds_since_epoch * 1000)
    
    return milliseconds


def plot_barchart(run: int):
    with open (f"../part_3_results_group_073/pods_{run}.json") as f:
        data = json.load(f)
    times = {}
    for item in data["items"][:-1]:
        for c in item["status"]["containerStatuses"]:
            times[c['name']] = {"started_at": c["state"]["terminated"]["startedAt"],"ended_at": c["state"]["terminated"]["finishedAt"]}
    begin = timestamp_to_milliseconds(times["parsec-blackscholes"]["started_at"])
    processed = {}
    for name, time in times.items():
        _, name = name.split("-")
        ts = (timestamp_to_milliseconds(time["started_at"]) - begin, timestamp_to_milliseconds(time["ended_at"])- begin)
        processed[name] = ts
    df_parsec = pd.DataFrame([(app, start, end) for app, (start, end) in processed.items()],
                  columns=['application', 'start', 'end'])
    convert_to_csv(f"../part_3_results_group_073/mcperf_{run}.txt", f"../part_3_results_group_073/mcperf_{run}.csv")
    df = pd.read_csv(f"../part_3_results_group_073/mcperf_{run}.csv")
    df_parsec["node"] = df_parsec["application"].map(nodes)
    fig, (ax1, ax2)  = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    df = df[df["ts_end"] > begin]
    df["ts_start"] -= begin
    df["ts_end"] -= begin
    df["ts_start"] /= 1000
    df["ts_end"] /= 1000
    df['p95'] /= 1000
    for i, row in df.iterrows():
        width = row['ts_end'] - row['ts_start']

        ax1.bar(row['ts_start'] + width/2,  # position the bar at the middle of its time range
            row['p95'],              # height of the bar is the p95 value
            width=width,             # width is the duration
            alpha=0.7,color="blue")
    ax1.set_xlabel('Time[s]')
    ax1.set_ylabel('P95 Value[ms]')
    ax1.set_title('P95 Values by Execution Timeframe')

    # Format x-axis with commas for thousands
    ax1.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Add grid for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Set y-axis to start from 0
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(0, 180)

    y_positions = np.arange(len(df_parsec))
    df_parsec['duration'] = df_parsec['end'] - df_parsec['start']

    for i, (_, row) in enumerate(df_parsec.iterrows()):
        ax2.barh(i, row['duration'] / 1000.0, left=row['start'] / 1000.0, height=0.5, 
                color=COLOUR_PALETTE[row["application"]], alpha=0.8, edgecolor='black')
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(df_parsec["node"] + " | " + df_parsec['application'])
    plt.tight_layout()
    fig.savefig(f"../plots/task3/q1_run{run}.png", format="png")


def timestamp_other(timestamp_str):
    # Parse the ISO format timestamp
    d = dt.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    
    # Convert to seconds since epoch as a float
    seconds_since_epoch = d.timestamp()
    
    return float(seconds_since_epoch)


def get_runtimes(run: int):
    with open (f"../part_3_results_group_073/pods_{run}.json") as f:
        data = json.load(f)
    times = {}
    for item in data["items"][:-1]:
        for c in item["status"]["containerStatuses"]:
            times[c['name']] = {"started_at": c["state"]["terminated"]["startedAt"],"ended_at": c["state"]["terminated"]["finishedAt"]}
    processed = {}
    for name, time in times.items():
        _, name = name.split("-")
        ts = (timestamp_other(time["started_at"]), timestamp_other(time["ended_at"]))
        processed[name] = ts
    df_parsec = pd.DataFrame([(app, start, end) for app, (start, end) in processed.items()],
                  columns=['application', 'start', 'end'])
    df_parsec["runtime"] = (df_parsec["end"] - df_parsec["start"]).astype(np.float64)
    df_parsec["run"] = run
    total_runtime = (df_parsec["end"].max() - df_parsec["start"].min())
    return df_parsec, total_runtime

def main():
    res = [get_runtimes(i) for i in range(1,4,1)]
    final_df = pd.concat(df for df, _ in res)
    final_df = final_df.groupby("application")["runtime"].agg(["mean", "std"]).reset_index()
    mean_runtime = statistics.mean([rt for _, rt in res])
    std_runtime = statistics.stdev([rt for _, rt in res])
    print(f"MEAN RUNTIME:\t {mean_runtime}")
    print(f"STD RUNTIME:\t {std_runtime}")
    print(final_df)


if __name__ == "__main__":
    main()
