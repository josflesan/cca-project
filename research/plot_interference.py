import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Argument Parsing Logic
parser = argparse.ArgumentParser(description="Interference Plot Arguments")
parser.add_argument("--benchmark", type=str, choices=['cpu', 'l1i', 'l1d', 'l2', 'llc', 'mem', 'bl', 'all'], default='plot', help='Benchmark to be plotted')

args = parser.parse_args()

# Define colour palette (7 line plots)
colours = sns.color_palette("husl", 7)

def convert_units(data):
    data['p95_mean'] = data['p95_mean'] / 1000
    data['p95_err'] = data['p95_err'] / 1000
    data['qps_mean'] = data['qps_mean'] / 1000
    data['qps_err'] = data['qps_err'] / 1000

def process_benchmark(benchmark: str):
    df1 = pd.read_csv(f"./data/{benchmark}_run1_int.csv")
    df1["run"] = 1
    df2 = pd.read_csv(f"./data/{benchmark}_run2_int.csv")
    df2["run"] = 2
    df3 = pd.read_csv(f"./data/{benchmark}_run3_int.csv")
    df2["run"] = 3
    df = pd.concat([df1, df2, df3])
    data = df[['QPS', 'target', 'p95']].groupby("target")[['QPS', 'p95']].agg(qps_mean=("QPS", "mean"), p95_mean=('p95', "mean"), p95_err=('p95', "std"), qps_err=('QPS', "std")).reset_index()

    return data

def plot_single_benchmark(df):
    data = df.copy()

    # Convert units
    data['p95_mean'] = data['p95_mean'] / 1000
    data['p95_err'] = data['p95_err'] / 1000
    data['qps_mean'] = data['qps_mean'] / 1000
    data['qps_err'] = data['qps_err'] / 1000

    # Set up plot style
    plt.figure(figsize=(10, 6), dpi=150)
    sns.set_style("darkgrid")
    # sns.set_context("notebook", font_scale=1.2)

    # Create main plot
    plt.plot(data['qps_mean'], data['p95_mean'],
             marker='o',
             linestyle='-',
             color=colours[0],
             linewidth=2,
             markersize=8,
             label=f'{args.benchmark.upper()} Interference')

    # Add error bars
    plt.errorbar(x=data['qps_mean'],
                 y=data['p95_mean'],
                 yerr=data['p95_err'],
                 xerr=data['qps_err'],
                 fmt='none',
                 color=colours[1],
                 alpha=0.5,
                 capsize=5,
                 capthick=1.5)
    
    # Configure titles and axes
    plt.title(f"{args.benchmark.upper()} Interference - QPS vs P95 Latency", loc='left', fontweight='bold', fontsize=12, family="Arial", pad=22)
    plt.ylabel("P95 Latency [ms]", loc='top', rotation=0, fontsize=10, family="Arial")
    
    plt.gca().yaxis.set_label_coords(0.115, 1.02)
    plt.xlabel("QPS (mean)", fontsize=10, family='Arial')
    plt.xlim(0, None)  # Not adding upper limit for individual plots so we can see any strange trends
    plt.ylim(0, None)  # Not adding upper limit for individual plots so we can see any strange trends
    plt.grid(axis='x')

    # Add legend
    plt.legend()

    # Output
    plt.tight_layout()
    plt.savefig(f"plots/{args.benchmark}.png", format="png", dpi=300)

def plot_all_benchmarks():
    baseline_df = process_benchmark("bl")
    cpu_df = process_benchmark("cpu")
    l1i_df = process_benchmark("l1i")
    l1d_df = process_benchmark("l1d")
    l2_df = process_benchmark("l2")
    llc_df = process_benchmark("llc")
    mem_df = process_benchmark("mem")

    # Convert units
    convert_units(baseline_df)
    convert_units(cpu_df)
    convert_units(l1i_df)
    convert_units(l1d_df)
    convert_units(l2_df)
    convert_units(llc_df)
    convert_units(mem_df)

    interference_groups = [baseline_df, cpu_df, l1i_df, l1d_df, l2_df, llc_df, mem_df]
    interference_labels = ["baseline", "ibench-cpu", "ibench-l1i", "ibench-l1d", "ibench-l2", "ibench-llc", "ibench-membw"]

    # Set up plot style
    plt.figure(figsize=(10, 6), dpi=150)
    sns.set_style("darkgrid")

    for idx, group in enumerate(interference_groups):
        # Create main plot        
        plt.plot(group['qps_mean'], group['p95_mean'],
                marker='o',
                linestyle='-',
                color=colours[idx],
                linewidth=2,
                markersize=8,
                label=f'{interference_labels[idx]}')

        # Add error bars
        plt.errorbar(x=group['qps_mean'],
                    y=group['p95_mean'],
                    yerr=group['p95_err'],
                    xerr=group['qps_err'],
                    fmt='none',
                    color=colours[idx],
                    alpha=0.6,
                    capsize=5,
                    capthick=1.5)
    
    # Configure titles and axes
    plt.title("QPS vs P95 Latency for Different Interference Types", loc='left', fontweight='bold', fontsize=12, family="Arial", pad=22)
    plt.ylabel("P95 Latency [ms]", loc='top', rotation=0, fontsize=10, family="Arial")
    
    plt.gca().yaxis.set_label_coords(0.115, 1.02)
    plt.xlabel("QPS (mean)", fontsize=10, family='Arial')
    plt.xlim(0, 80)
    plt.ylim(0, 6)
    plt.grid(axis='x')

    # Add legend
    plt.legend()

    # Output
    plt.tight_layout()
    plt.savefig("plots/all.png", format="png", dpi=300)

if __name__ == '__main__':

    if args.benchmark == "all":
        plot_all_benchmarks()
    else:
        df = process_benchmark(args.benchmark)
        plot_single_benchmark(df)
