import pandas as pd
import matplotlib.pyplot as plt

# User-provided lists
colors = ["red", "green", "blue", "purple", "cyan", "yellow", "black", "orange"]
labels = [0.5, 1.0, 1.3, 1.5, 1.7, 2.1, 2.6, 2.9]

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)

plt.figure(figsize=(8, 6))

for color, label in zip(colors, labels):
    filename = f"ref_Figueiredo_2016/{color}_dataset.csv"

    # Load CSV with ; separator and , decimal
    df = pd.read_csv(filename, sep=";", decimal=",", header=None, names=["x", "y"])

    df["x"] = pd.concat([pd.Series([1]), df["x"]], ignore_index=True)
    df["y"] = pd.concat([pd.Series([df["y"].iloc[0]]), df["y"]], ignore_index=True)

    # Plot curve
    plt.plot(df["x"], df["y"], color=color, linestyle="--", alpha=0.6)

# Plot formatting
plt.xlabel("$r$ [1]")
plt.ylabel(r"$h(r,t\to\infty)$ [1]")
plt.xlim(1, 2)
plt.grid(True)
plt.tight_layout()

import numpy as np

data = np.load("my_data/surface_shapes_all_a0.0_SV43.npz", allow_pickle=True)
records = data["records"]

for record, color in zip(records, colors):
    xs = record["xs"]
    vals = record["vals"]
    RPS = record["RPS"]

    plt.plot(
        xs,
        vals,
        color=color,
        label=f"$\omega$ = {RPS}",
        marker="d",
        markevery=4,
        markersize=2,
    )

plt.legend()
plt.savefig("oldroyd_B_comparison.pdf", dpi=300)
