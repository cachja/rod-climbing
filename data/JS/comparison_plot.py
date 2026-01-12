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
        markevery=4,
        markersize=2,
    )

# Plot formatting
plt.ylabel(r"$h(r,t\to\infty)$ [1]")
plt.xlabel("$r$ [1]")
plt.xlim(1, 2)
plt.ylim(-0.3, 0.15)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("JS_modelB.pdf", dpi=300)
plt.close()


plt.figure(figsize=(8, 6))
ax = plt.gca()

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
        markevery=4,
        markersize=2,
    )

# Plot formatting
plt.xlabel("$r$ [1]")
plt.ylabel(r"$h(r,t\to\infty)$ [1]")
plt.xlim(1, 2)
plt.ylim(-0.3, 0.15)
plt.grid(True)
plt.legend()

ax.legend()

# Inset plot (magnifying glass)
axins = inset_axes(
    ax,
    width="75%",
    height="50%",
    loc="lower left",
    bbox_to_anchor=(0.075, 0.0475, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)

for record, color in zip(records_A, colors):
    xs = record["xs"]
    vals = record["vals"]
    axins.plot(xs, vals, color=color, marker="x", markevery=4)

# Zoom region
x1, x2 = 1.00, 1.5
axins.set_xlim(x1, x2)

# Automatically fit y-limits to data in zoom region
all_vals = []
for record in records_A:
    mask = (record["xs"] >= x1) & (record["xs"] <= x2)
    all_vals.extend(record["vals"][mask])

axins.set_ylim(-0.015, 0.005)

axins.grid(True)
yticks = axins.get_yticks()
axins.set_yticks(yticks[::2])
axins.set_yticklabels([f"{y:.3f}" for y in yticks[::2]])
axins.tick_params(labelsize=8)

# Draw connecting lines
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="black")


plt.tight_layout()

plt.savefig("JS_modelA.pdf", dpi=300)
