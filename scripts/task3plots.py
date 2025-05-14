import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

def plot_barchart(run: int):
    # Get the part 3 plot data

    # Get the files needed to create the plot


    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, choices=[1, 2, 3], help="Run to plot")
    args = parser.parse_args()

    plot_barchart(args.run)

if __name__ == "__main__":
    main()
