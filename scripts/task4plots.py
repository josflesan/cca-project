from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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

    def func(x):
        if cores == 1:
            return x[0]
        else:
            return x[0] + x[1]

    cpu_df["per_cpu"] = cpu_df["per_cpu"].apply(
        lambda x: eval(x)
    )  # Convert string to list
    cpu_df["utilization"] = cpu_df["per_cpu"].apply(func)

    # Concatenate the dataframes
    df = concat_files(latency_df, cpu_df)
    plt.figure(figsize=(10, 6))
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

    ax1.set_xlabel("QPS (K)", fontsize=12, family="Arial")
    ax1.set_ylabel("95th Percentile Latency (ms)", fontsize=12, family="Arial")
    ax1.set_xlim(0, 230)
    ax1.set_ylim(0, 2)

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
    ax2.set_ylabel("CPU Utilization (%)", fontsize=12, family="Arial")
    plt.title(
        kwargs["title"], loc="center", fontweight="bold", fontsize=14, family="Arial"
    )
    ax1.grid(axis="x")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    output_dir = Path("plots/task4")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"plots-41d_c{cores}.png", format="png", dpi=300)


def concat_files(latencies, cpu_utils):
    # Make sure timestamps are sorted
    latencies = latencies.sort_values("ts_start")
    latencies["ts_start"] = latencies["ts_start"].astype("float64")
    cpu_utils_df = cpu_utils.sort_values("timestamp")
    cpu_utils_df["timestamp"] = cpu_utils_df["timestamp"] * 1000

    # Create a merged dataframe where ts >= ts_start
    merged = pd.merge_asof(
        latencies,
        cpu_utils_df,
        left_on="ts_start",
        right_on="timestamp",
        direction="forward",
    )

    result = merged[merged["timestamp"] <= merged["ts_end"]]
    utilizations = result.groupby("ts_start")["utilization"].mean().reset_index()
    res = pd.merge(latencies, utilizations, on="ts_start")
    return res


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
    plt.figure(figsize=(15, 6), dpi=150)
    sns.set_style("darkgrid")

    # Create main plots
    ax1 = plt.gca()  # Primary axis for P95 latency
    ax1.plot(
        data["seconds"],
        data["p95"],
        marker="o",
        linestyle="-",
        color=colours[0],
        linewidth=2,
        markersize=8,
        label="P95 Latency",
    )
    ax1.set_ylabel(
        "P95 Latency [ms]",
        loc="top",
        rotation=0,
        fontsize=10,
        weight="bold",
        family="Arial",
        color=colours[0],
    )
    ax1.tick_params(axis="y", colors=colours[0], pad=10)
    for tick in ax1.get_yticklabels():
        tick.set_fontweight("bold")
    ax1.yaxis.set_label_coords(0.085, 1.02)
    ax1.set_xlabel("Time Elapsed (seconds)", fontsize=10, family="Arial", labelpad=10)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    ax1.grid(None)

    ax2 = ax1.twinx()
    ax2.plot(
        data["seconds"],
        data["qps"],
        marker="s",
        linestyle="-",
        color=colours[1],
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="QPS",
    )
    ax2.set_ylabel(
        "QPS",
        loc="top",
        rotation=0,
        fontsize=10,
        weight="bold",
        family="Arial",
        color=colours[1],
    )
    ax2.tick_params(axis="y", colors=colours[1])
    for tick in ax2.get_yticklabels():
        tick.set_fontweight("bold")
    ax2.yaxis.set_label_coords(0.995, 1.03)
    ax2.set_ylim(0, None)
    ax2.grid(None)

    # Annotate with start, pause and unpause vertical lines
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.grid(axis="y")

    markers = {"pause": "^", "unpause": "v"}

    start_positions = []

    for job, time in plot_data.start_events.items():
        if job == "scheduler":
            continue

        # Add vertical line at the event position
        ax1.axvline(
            x=time,
            color=COLOUR_PALETTE[job],
            linestyle="--",
            linewidth=1.5,
            label="Start",
        )

        start_positions.append(time)

    for job, times in plot_data.pause_events.items():
        if job == "scheduler":
            continue

        for time in times:
            ax1.scatter(
                time,
                0.04,
                marker=markers["pause"],
                color=COLOUR_PALETTE[job],
                s=100,
                label="Pause",
            )

    for job, times in plot_data.unpause_events.items():
        if job == "scheduler":
            continue

        for time in times:
            ax1.scatter(
                time,
                1.74,
                marker=markers["unpause"],
                color=COLOUR_PALETTE[job],
                s=100,
                label="Unpause",
            )

    ax3.set_xticklabels([])  # Remove the labels (empty list)
    ax3.tick_params(axis="x", direction="in", length=0)  # Remove the ticks lines

    ax1.tick_params(axis="x", rotation=90)
    ax1.set_xticks(start_positions)

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
    plt.figure(figsize=(15, 6), dpi=150)
    sns.set_style("darkgrid")

    # Create main plots
    ax1 = plt.gca()  # Primary axis for P95 latency
    ax1.step(
        data["seconds"],
        data["cores"],
        marker="o",
        linestyle="-",
        color=colours[0],
        linewidth=2,
        markersize=8,
        label="Cores",
    )
    ax1.set_ylabel(
        "P95 Latency [ms]",
        loc="top",
        rotation=0,
        fontsize=10,
        weight="bold",
        family="Arial",
        color=colours[0],
    )
    ax1.tick_params(axis="y", colors=colours[0], pad=10)
    for tick in ax1.get_yticklabels():
        tick.set_fontweight("bold")
    ax1.yaxis.set_label_coords(0.085, 1.02)
    ax1.set_xlabel("Time Elapsed (seconds)", fontsize=10, family="Arial", labelpad=10)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, 2.2)
    ax1.grid(None)

    ax2 = ax1.twinx()
    ax2.plot(
        data["seconds"],
        data["qps"],
        marker="s",
        linestyle="-",
        color=colours[1],
        linewidth=2,
        markersize=6,
        alpha=0.7,
        label="QPS",
    )
    ax2.set_ylabel(
        "QPS",
        loc="top",
        rotation=0,
        fontsize=10,
        weight="bold",
        family="Arial",
        color=colours[1],
    )
    ax2.tick_params(axis="y", colors=colours[1])
    for tick in ax2.get_yticklabels():
        tick.set_fontweight("bold")
    ax2.yaxis.set_label_coords(0.995, 1.03)
    ax2.set_ylim(0, None)
    ax2.grid(None)

    # Annotate with start, pause and unpause vertical lines
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.grid(axis="y")

    markers = {"pause": "^", "unpause": "v"}

    start_positions = []

    for job, time in plot_data.start_events.items():
        if job == "scheduler":
            continue

        # Add vertical line at the event position
        ax1.axvline(
            x=time,
            color=COLOUR_PALETTE[job],
            linestyle="--",
            linewidth=1.5,
            label="Start",
        )

        start_positions.append(time)

    for job, times in plot_data.pause_events.items():
        if job == "scheduler":
            continue

        for time in times:
            ax1.scatter(
                time,
                0.04,
                marker=markers["pause"],
                color=COLOUR_PALETTE[job],
                s=100,
                label="Pause",
            )

    for job, times in plot_data.unpause_events.items():
        if job == "scheduler":
            continue

        for time in times:
            ax1.scatter(
                time,
                2.15,
                marker=markers["unpause"],
                color=COLOUR_PALETTE[job],
                s=100,
                label="Unpause",
            )

    ax3.set_xticklabels([])  # Remove the labels (empty list)
    ax3.tick_params(axis="x", direction="in", length=0)  # Remove the ticks lines

    ax1.tick_params(axis="x", rotation=90)
    ax1.set_xticks(start_positions)

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

    # plot_lines_41d(Path("data/task4/Q1D/cpu1_measurement.csv"), Path("data/task4/Q1D/cpu1_stats.csv"), cores=1, title="QPS vs P95 Latency and CPU Utilization (C = 1, T = 1)")
    # plot_lines_41d(Path("data/task4/Q1D/cpu2_measurement.csv"), Path("data/task4/Q1D/cpu2_stats.csv"), cores=2, title="QPS vs P95 Latency and CPU Utilization (C = 2, T = 1)")

    # Scheduler plots
    plot_scheduler(question=3, run=1)


if __name__ == "__main__":
    main()
