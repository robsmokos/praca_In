import argparse
import glob
from itertools import cycle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import seaborn as sns

sns.set(
    style="darkgrid",
    rc={
        "figure.figsize": (7.2, 4.45),
        "text.usetex": True,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "font.size": 15,
        "figure.autolayout": True,
        "axes.titlesize": 16,
        "axes.labelsize": 17,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.fontsize": 15,
    },
)
colors = sns.color_palette("colorblind", 4)
dashes_styles = cycle(["solid", "dash", "dashdot", "dot"])
sns.set_palette(colors)
colors = cycle(colors)

# Definicja palety kolorów
color_palette = [
    "rgb(0, 123, 213)",  # blue
    "rgb(255, 150, 0)",  # orange
    "rgb(128, 186, 0)",  # green
    "rgb(202, 0, 32)",  # red
]


def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


def plot_df(df, color, xaxis, yaxis, ma=1, label=""):
    df[yaxis] = pd.to_numeric(df[yaxis], errors="coerce")

    mean = df.groupby(xaxis).mean()[yaxis]
    std = df.groupby(xaxis).std()[yaxis]
    if ma > 1:
        mean = moving_average(mean, ma)
        std = moving_average(std, ma)

    x = df.groupby(xaxis)[xaxis].mean().keys().values

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line=dict(color=color_palette[color], dash=next(dashes_styles)),
            name=label,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean + std,
            mode="lines",
            line=dict(color=color_palette[color], width=0),
            marker=dict(color=color_palette[color]),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean - std,
            mode="lines",
            line=dict(color=color_palette[color], width=0),
            marker=dict(color=color_palette[color]),
            fill="tonexty",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Traffic Signal Metrics" if label == "" else label,
        xaxis_title="Time step (seconds)",
        yaxis_title="Total waiting time (s)",
    )

    return fig


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Plot Traffic Signal Metrics""",
    )
    prs.add_argument("-f", nargs="+", required=True, help="Measures files\n")
    prs.add_argument("-l", nargs="+", default=None, help="File's legends\n")
    prs.add_argument("-t", type=str, default="", help="Plot title\n")
    prs.add_argument(
        "-yaxis", type=str, default="system_total_waiting_time", help="The column to plot.\n"
    )
    prs.add_argument("-xaxis", type=str, default="step", help="The x axis.\n")
    prs.add_argument("-ma", type=int, default=1, help="Moving Average Window.\n")
    prs.add_argument(
        "-sep", type=str, default=",", help="Values separator on file.\n"
    )
    prs.add_argument(
        "-xlabel", type=str, default="Time step (seconds)", help="X axis label.\n"
    )
    prs.add_argument(
        "-ylabel", type=str, default="Total waiting time (s)", help="Y axis label.\n"
    )
    prs.add_argument("-output", type=str, default=None, help="HTML output filename.\n")

    args = prs.parse_args()
    labels = cycle(args.l) if args.l is not None else cycle([str(i) for i in range(len(args.f))])

    fig = go.Figure()

    for idx, file in enumerate(args.f):
        main_df = pd.DataFrame()
        for f in glob.glob(file + "*"):
            df = pd.read_csv(f, sep=args.sep)
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))

        fig = plot_df(
            main_df,
            xaxis=args.xaxis,
            yaxis=args.yaxis,
            label=next(labels),
            color=idx,
            ma=args.ma,
        )

    if args.output is not None:
        pio.write_html(fig, args.output + ".html")

    fig.show()
