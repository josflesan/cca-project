from pathlib import Path

import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import seaborn as sns
from utils.part4_utils import (
    Part4PlotData,
    parse_file_name,
    parse_logger_file,
    prepare_part4_plot_data,
)

COLOUR_PALETTE = {
    "blackscholes": "#CCA000",
    "canneal": "#CCCCAA",
    "dedup": "#CCACCA",
    "ferret": "#AACCCA",
    "freqmine": "#0CCA00",
    "radix": "#00CCA0",
    "vips": "#CC0A00",
}

colours = sns.color_palette("husl", 2)


def plot_lines_41d(latency_file: Path, cpu_file: Path, cores: int = 1, **kwargs):
    colours = sns.color_palette("husl", 2)

    latency_df = pd.read_csv(latency_file)
    cpu_df = pd.read_csv(cpu_file)

    # def func(x):
    #     if cores == 1:
    #         return x[0]
    #     else:
    #         return x[0] + x[1]

    # cpu_df["per_cpu"] = cpu_df["per_cpu"].apply(
    #     lambda x: eval(x)
    # )  # Convert string to list
    # cpu_df["utilization"] = cpu_df["per_cpu"].apply(func)

    # Concatenate the dataframes
    df = concat_files(latency_df, cpu_df)
    plt.figure(figsize=(10, 6), dpi=150)
    sns.set_style("darkgrid")

    ax1 = plt.gca()
    ax1.plot(
        df["QPS"] / 1000,
        df["p95"] / 1000,
        marker="o",
        linestyle="-",
        color=colours[0],
        linewidth=2,
        markersize=8,
        label="95th Percentile Latency",
    )

    # slo guarantee limit
    ax1.axhline(
        y=0.8, color="red", linestyle=":", linewidth=2, label="0.8ms Latency SLO"
    )

    ax1.tick_params(axis="y", colors=colours[0])
    for tick in ax1.get_yticklabels():
        tick.set_fontweight("bold")
    ax1.set_xlabel("QPS (K)", fontsize=12, family="Arial", weight="bold")
    ax1.set_ylabel(
        "P95 Latency [ms]",
        loc="top",
        fontsize=10,
        rotation=0,
        family="Arial",
        weight="bold",
        color=colours[0]
    )
    ax1.yaxis.set_label_coords(0.13, 1.02)
    ax1.set_xlim(0, 230)
    ax1.set_ylim(0, None)

    ax2 = ax1.twinx()
    ax2.plot(
        df["QPS"] / 1000,
        df["utilization"],
        marker="s",
        linestyle="--",
        color=colours[1],
        linewidth=2,
        markersize=6,
        label="CPU Utilization",
    )

    # Set CPU utilization y-axis
    ax2.tick_params(axis="y", colors=colours[1])
    for tick in ax2.get_yticklabels():
        tick.set_fontweight("bold")
    ax2.set_ylabel(
        "CPU Utilization [%]",
        loc="top",
        rotation=0,
        fontsize=10,
        weight="bold",
        family="Arial",
        color=colours[1]
    )
    ax2.yaxis.set_label_coords(0.995, 1.04)
    ax2.set_ylim(0, 100)

    plt.title(
        kwargs["title"],
        loc="left",
        fontweight="bold", 
        fontsize=14, 
        family="Arial",
        pad=22,
    )
    ax1.grid(axis="x")
    ax2.grid(None)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    output_dir = Path("plots/task4")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"plots-41d_c{cores}.png", format="png", dpi=300)

def concat_files(latencies, cpu_utils):
    # Convert and sort timestamps
    latencies = latencies.sort_values("ts_start").copy()
    latencies["ts_start"] = latencies["ts_start"].astype("float64")
    latencies["ts_end"] = latencies["ts_end"].astype("float64")

    cpu_utils = cpu_utils.copy()
    cpu_utils["timestamp"] = cpu_utils["timestamp"].astype("float64") * 1000  # Convert to ms
    cpu_utils = cpu_utils.sort_values("timestamp")

    # Collect average CPU utilization in each request window
    cpu_utils_list = []
    for _, row in latencies.iterrows():
        in_window = cpu_utils[
            (cpu_utils["timestamp"] >= row["ts_start"]) &
            (cpu_utils["timestamp"] <= row["ts_end"])
        ]
        if not in_window.empty:
            avg_util = in_window["utilization"].mean()
        else:
            avg_util = float("nan")  # or 0.0 if you prefer
        cpu_utils_list.append(avg_util)

    latencies["utilization"] = cpu_utils_list
    return latencies

def plot_lines_41a(files: list[Path], out: str = "", **kwargs) -> None:
    colours = sns.color_palette("husl", len(files))
    for colour, program_file in zip(colours, files):
        threads, cores = parse_file_name(program_file.name)
        df = pd.read_csv(program_file)
        plt.plot(
            df["QPS"] / 1000,
            df["p95"] / 1000,
            marker="o",
            linestyle="-",
            color=colour,
            linewidth=2,
            markersize=8,
            label=f"{cores} Cores, {threads} Threads ",
        )

    title = kwargs.get(
        "title", "QPS vs P95 Latency for various thread/core configurations"
    )
    plt.title(title, loc="left", fontweight="bold", fontsize=12, family="Arial", pad=22)
    plt.ylabel("P95 Latency [ms]", loc="top", rotation=0, fontsize=10, family="Arial")

    plt.gca().yaxis.set_label_coords(0.115, 1.02)
    plt.xlabel("QPS (mean)", fontsize=10, family="Arial")
    # plt.xlim(0, 80)
    # plt.ylim(0, 2)
    plt.grid(axis="x")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/task4/plot{out}.png", format="png", dpi=300)


def plot_A(plot_data: Part4PlotData, question: int, run: int):
    """Part 4 Plot with QPS in right y-axis and P95 Latency in left y-axis"""

    data = plot_data.df

    # Set up plot style
    sns.set_style("darkgrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        dpi=150,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.01},  # Latency/QPS gets 3x height of bar chart
        sharex=True
    )

    # Create main plots
    ax1_lat = ax1  # Primary axis for P95 latency 
    ax1_lat.plot(
        data["seconds"],
        data["p95"],
        marker="o",
        linestyle="-",
        color=colours[0],
        linewidth=2,
        markersize=6,
        label="P95 Latency",
    )
    ax1_lat.set_ylabel(
        "P95 Latency [ms]",
        loc="top",
        rotation=0,
        fontsize=10,
        weight="bold",
        family="Arial",
        color=colours[0],
    )
    ax1_lat.tick_params(axis="y", colors=colours[0], pad=10)
    for tick in ax1_lat.get_yticklabels():
        tick.set_fontweight("bold")
    ax1_lat.yaxis.set_label_coords(0.095, 1.02)
    ax1_lat.set_xlim(0, None)
    ax1_lat.set_ylim(0, None)
    ax1_lat.grid(axis="y", visible=False)
    ax1_lat.grid(axis="x", linewidth=2)

    ax1_qps = ax1_lat.twinx()
    ax1_qps.plot(
        data["seconds"],
        data["qps"] / 1000,
        marker="s",
        linestyle="-",
        color=colours[1],
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="QPS",
    )
    ax1_qps.set_ylabel(
        "QPS [K]",
        loc="top",
        rotation=0,
        fontsize=10,
        weight="bold",
        family="Arial",
        color=colours[1],
    )
    ax1_qps.tick_params(axis="y", colors=colours[1])
    for tick in ax1_qps.get_yticklabels():
        tick.set_fontweight("bold")
    ax1_qps.yaxis.set_label_coords(1, 1.03)
    ax1_qps.set_ylim(0, None)
    ax1_qps.grid(None)

    # Set legend for lineplot
    lines_1, labels_1 = ax1_lat.get_legend_handles_labels()
    lines_2, labels_2 = ax1_qps.get_legend_handles_labels()
    ax1_lat.legend((lines_1 + lines_2), (labels_1 + labels_2), loc="upper right")

    # Bottom subplot (Bar Chart)
    # Annotate with runtime bars
    all_jobs = []
    for idx, (job, time) in enumerate(plot_data.start_events.items()):
        if job == "scheduler":
            continue

        all_jobs.append(job)
        end_time = plot_data.end_events[job]

        # Add bar plot from the start of the job to the end
        ax2.barh(
            idx - 1,
            end_time - time,
            label=job,
            left=time,
            height=0.75,
            color=COLOUR_PALETTE[job],
            alpha=0.9
        )

    # Annotate with pause bars
    for job_idx, (job, times) in enumerate(plot_data.pause_events.items()):
        if job == "scheduler":
            continue

        rgb = mcolors.to_rgb(COLOUR_PALETTE[job])
        hue, light, sat = colorsys.rgb_to_hls(*rgb)
        desat_color = colorsys.hls_to_rgb(hue, light, sat * 0.5)  # Reduce saturation by 50%

        # Draw a bar for each of the pause times
        for time_idx, pause in enumerate(times):
            unpause_time = plot_data.unpause_events[job][time_idx]

            # Add bar plot from the pause time to the unpause time
            ax2.barh(
                job_idx - 1,
                unpause_time - pause,
                left=pause,
                height=0.75,
                color=desat_color,
                alpha=1
            )

    ax2.legend(loc="upper left", ncols=4)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", linewidth=2)
    ax2.set_xlabel("Time Elapsed (seconds)", fontsize=10, family="Arial", labelpad=10)
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)

    # Configure titles
    plt.title(
        "P95 Latency and QPS vs Time for Scheduling Policy",
        loc="left",
        fontweight="bold",
        fontsize=12,
        family="Arial",
        pad=22,
    )

    # Output
    plt.tight_layout()
    plt.savefig(f"plots/task4/Q{question}_run{run}_plotA.png", format="png", dpi=300)
    plt.close()


def plot_B(plot_data: Part4PlotData, question: int, run: int):
    """Part 4 Plot with QPS in right y-axis and Cores Used for memcached in left y-axis"""

    data = plot_data.df

    # Set up plot style
    sns.set_style("darkgrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(15, 8),
        dpi=150,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.01},  # Latency/QPS gets 3x height of bar chart
        sharex=True
    )

    # Create main plots
    ax1_lat = ax1  # Primary axis for P95 latency
    ax1_lat.step(
        data["seconds"],
        data["cores"],
        marker="o",
        linestyle="-",
        color=colours[0],
        linewidth=2,
        markersize=6,
        label="Cores",
    )
    ax1_lat.set_ylabel(
        "P95 Latency [ms]",
        loc="top",
        rotation=0,
        fontsize=10,
        weight="bold",
        family="Arial",
        color=colours[0],
    )
    ax1_lat.tick_params(axis="y", colors=colours[0], pad=10)
    for tick in ax1_lat.get_yticklabels():
        tick.set_fontweight("bold")
    ax1_lat.yaxis.set_label_coords(0.097, 1.02)
    ax1_lat.set_xlabel("Time Elapsed (seconds)", fontsize=10, family="Arial", labelpad=10)
    ax1_lat.set_xlim(0, None)
    ax1_lat.set_ylim(0, 2.2)
    ax1_lat.grid(axis="y", visible=False)
    ax1_lat.grid(axis="x", linewidth=2)

    ax1_qps = ax1_lat.twinx()
    ax1_qps.plot(
        data["seconds"],
        data["qps"] / 1000,
        marker="s",
        linestyle="-",
        color=colours[1],
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="QPS",
    )
    ax1_qps.set_ylabel(
        "QPS [K]",
        loc="top",
        rotation=0,
        fontsize=10,
        weight="bold",
        family="Arial",
        color=colours[1],
    )
    ax1_qps.tick_params(axis="y", colors=colours[1])
    for tick in ax1_qps.get_yticklabels():
        tick.set_fontweight("bold")
    ax1_qps.yaxis.set_label_coords(0.995, 1.03)
    ax1_qps.set_ylim(0, None)
    ax1_qps.grid(None)

    # Set legend for lineplot
    lines_1, labels_1 = ax1_lat.get_legend_handles_labels()
    lines_2, labels_2 = ax1_qps.get_legend_handles_labels()
    ax1_lat.legend((lines_1 + lines_2), (labels_1 + labels_2), loc="upper right")


    # Bottom subplot (Bar Chart)
    # Annotate with runtime bars
    all_jobs = []
    for idx, (job, time) in enumerate(plot_data.start_events.items()):
        if job == "scheduler":
            continue

        all_jobs.append(job)
        end_time = plot_data.end_events[job]

        # Add bar plot from the start of the job to the end
        ax2.barh(
            idx - 1,
            end_time - time,
            label=job,
            left=time,
            height=0.75,
            color=COLOUR_PALETTE[job],
            alpha=0.9
        )

    # Annotate with pause bars
    for job_idx, (job, times) in enumerate(plot_data.pause_events.items()):
        if job == "scheduler":
            continue

        rgb = mcolors.to_rgb(COLOUR_PALETTE[job])
        hue, light, sat = colorsys.rgb_to_hls(*rgb)
        desat_color = colorsys.hls_to_rgb(hue, light, sat * 0.5)  # Reduce saturation by 50%

        # Draw a bar for each of the pause times
        for time_idx, pause in enumerate(times):
            unpause_time = plot_data.unpause_events[job][time_idx]

            # Add bar plot from the pause time to the unpause time
            ax2.barh(
                job_idx - 1,
                unpause_time - pause,
                left=pause,
                height=0.75,
                color=desat_color,
                alpha=1
            )

    ax2.legend(loc="upper left", ncols=4)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", linewidth=2)
    ax2.set_xlabel("Time Elapsed (seconds)", fontsize=10, family="Arial", labelpad=10)
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)


    # Configure titles
    plt.title(
        "Memcached Cores Used and QPS vs Time for Scheduling Policy",
        loc="left",
        fontweight="bold",
        fontsize=12,
        family="Arial",
        pad=22,
    )

    # Output
    plt.tight_layout()
    plt.savefig(f"plots/task4/Q{question}_run{run}_plotB.png", format="png", dpi=300)
    plt.close()


def plot_scheduler(question: int, run: int):
    # Define log files and CSV paths
    log_file = f"logs/part4/Q{question}/log_RUN{run}.txt"
    memcached_df = pd.read_csv(f"data/task4/Q{question}/memcached_RUN{run}.csv")
    parsed_logfile = parse_logger_file(log_file)

    # Prepare plot data
    plot_data = prepare_part4_plot_data(
        parsed_logfile, memcached_df, qps_interval=10, question=3, run=1
    )

    # Plot A type plots and B type plots
    plot_A(plot_data, question, run)
    plot_B(plot_data, question, run)


def main():
    # part4_1a_dir = Path("data/task4/Q1A")
    # plot_lines_41a(list(part4_1a_dir.iterdir()), out="4.1a")

    # plot_lines_41d(Path("data/task4/Q1D/cpu1_memcached.csv"), Path("data/task4/Q1D/cpu1_stats.csv"), cores=1, title="QPS vs P95 Latency and CPU Utilization (C = 1, T = 1)")
    # plot_lines_41d(Path("data/task4/Q1D/cpu2_memcached.csv"), Path("data/task4/Q1D/cpu2_stats.csv"), cores=2, title="QPS vs P95 Latency and CPU Utilization (C = 2, T = 1)")

    # Scheduler plots
    plot_scheduler(question=3, run=1)


if __name__ == "__main__":
    main()
