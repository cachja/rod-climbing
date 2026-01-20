import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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

data_B = np.load("my_data/surface_shapes_all_1.0_JS_B.npz", allow_pickle=True)
data_A = np.load("my_data/surface_shapes_all_1.0_JS_A.npz", allow_pickle=True)
recorda_B = data_B["records"]
records_A = data_A["records"]

plt.figure(figsize=(8, 6))

for record, color in zip(recorda_B, colors):
    xs = record["xs"]
    vals = record["vals"]
    a = record["a"]

    plt.plot(
        xs,
        vals,
        color=color,
        label=f"a = {np.round(a,1)}",
        marker="d",
        markevery=2,
        markersize=2,
    )

# Plot formatting
plt.ylabel(r"$h(r,t\to\infty)$ [1]")
plt.xlabel("$r$ [1]")
plt.xlim(1, 2)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("JS_lower_derivative_modelB.pdf", dpi=300)
plt.close()


plt.figure(figsize=(8, 6))

for record, color in zip(records_A, colors):
    xs = record["xs"]
    vals = record["vals"]
    a = record["a"]

    plt.plot(
        xs,
        vals,
        color=color,
        label=f"a = {np.round(a,1)}",
        marker="d",
        markevery=2,
        markersize=2,
    )

# Plot formatting
plt.xlabel("$r$ [1]")
plt.ylabel(r"$h(r,t\to\infty)$ [1]")
plt.xlim(1, 2)
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig("JS_lower_derivative_modelA.pdf", dpi=300)
