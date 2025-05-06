from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_file_name(filename: str):
    base_name = filename.split('.')[0]
    parts = base_name.split('_')
    thread_part = parts[0]
    thread_number = int(thread_part.replace('thread', ''))
    cpu_part = parts[1]
    cpu_number = int(cpu_part.replace('cpu', ''))
    
    return (thread_number, cpu_number)


def plot_lines_41d(latency_file: Path, cpu_file: Path, cores: int=1, **kwargs):
    colours = sns.color_palette("husl", 2)
    
    latency_df = pd.read_csv(latency_file)
    cpu_df = pd.read_csv(cpu_file)
    def func(x):
        if cores == 1:
            return x[0]
        else:
            return x[0] + x[1]
    
    cpu_df["per_cpu"] = cpu_df["per_cpu"].apply(lambda x: eval(x))  # Convert string to list
    cpu_df["utilization"] = cpu_df["per_cpu"].apply(func)    
    
    # Concatenate the dataframes
    df = concat_files(latency_df, cpu_df)
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(df['QPS'] / 1000, df['p95'] / 1000,
            marker='o',
            linestyle='-',
            color=colours[0],
            linewidth=2,
            markersize=8,
            label='95th Percentile Latency')
    
    # slo guarantee limit
    ax1.axhline(y=0.8, color='red', linestyle=':', linewidth=2, label='0.8ms Latency SLO')
    
    ax1.set_xlabel("QPS (K)", fontsize=12, family='Arial')
    ax1.set_ylabel("95th Percentile Latency (ms)", fontsize=12, family="Arial")
    ax1.set_xlim(0, 230)  
    ax1.set_ylim(0, 2)
    
    ax2 = ax1.twinx()
    ax2.plot(df['QPS'] / 1000, df['utilization'],
            marker='s',
            linestyle='--',
            color=colours[1],
            linewidth=2,
            markersize=6,
            label='CPU Utilization')
    
    # Set CPU utilization y-axis
    ax2.set_ylabel("CPU Utilization (%)", fontsize=12, family="Arial")
    # title = 
    plt.title(kwargs['title'], loc='center', fontweight='bold', fontsize=14, family="Arial")
    ax1.grid(axis='x')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    output_dir = Path("plots/task4")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"plots-41d_c{cores}.png", format="png", dpi=300)



def concat_files(latencies, cpu_utils):
    
    # Make sure timestamps are sorted
    latencies = latencies.sort_values('ts_start')
    latencies["ts_start"] = latencies["ts_start"].astype("float64")
    cpu_utils_df = cpu_utils.sort_values('timestamp')
    cpu_utils_df["timestamp"] = cpu_utils_df["timestamp"] * 1000

    # print(latencies["ts_start"].astype())
    # print(cpu_utils_df["timestamp"].dtype)
    
    # Create a merged dataframe where ts >= ts_start
    merged = pd.merge_asof(
        latencies,
        cpu_utils_df,
        left_on='ts_start',
        right_on='timestamp',
        direction='forward'
    )
    
    result = merged[merged['timestamp'] <= merged['ts_end']]
    utilizations = result.groupby("ts_start")["utilization"].mean().reset_index()
    res = pd.merge(latencies, utilizations, on="ts_start")
    return res




def plot_lines_41a(files: list[Path], out: str="", **kwargs) -> None:
    colours = sns.color_palette("husl", len(files))
    for colour, program_file in zip(colours,files):
        threads, cores = parse_file_name(program_file.name)
        df = pd.read_csv(program_file)
        plt.plot(df['QPS'] / 1000, df['p95'] / 1000,
                marker='o',
                linestyle='-',
                color=colour,
                linewidth=2,
                markersize=8,
                label=f'{cores} Cores, {threads} Threads ')

    title = kwargs.get("title","QPS vs P95 Latency for various thread/core configurations")
    plt.title(title,loc='left', fontweight='bold', fontsize=12, family="Arial", pad=22)
    plt.ylabel("P95 Latency [ms]", loc='top', rotation=0, fontsize=10, family="Arial")
    
    plt.gca().yaxis.set_label_coords(0.115, 1.02)
    plt.xlabel("QPS (mean)", fontsize=10, family='Arial')
    # plt.xlim(0, 80)
    # plt.ylim(0, 2)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/task4/plot{out}.png", format="png", dpi=300)


def plot_A():
    pass

def plot_B():
    pass

def plot_lines_43(run: int):
    plot_A()
    plot_B()

def main():
    # part4_1a_dir = Path("data/task4/Q1A")
    # plot_lines_41a(list(part4_1a_dir.iterdir()), out="4.1a")

    plot_lines_41d(Path("data/task4/Q1D/cpu1_measurement.csv"), Path("data/task4/Q1D/cpu1_stats.csv"), cores=1, title="QPS vs P95 Latency and CPU Utilization (C = 1, T = 1)")
    plot_lines_41d(Path("data/task4/Q1D/cpu2_measurement.csv"), Path("data/task4/Q1D/cpu2_stats.csv"), cores=2, title="QPS vs P95 Latency and CPU Utilization (C = 2, T = 1)")


if __name__ == "__main__":
    main()




