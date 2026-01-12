from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup plot formating
colors = ["red", "green", "blue", "purple", "cyan", "orange"]
labels = [0.5, 1.0, 1.3, 1.5, 1.7, 2.1, 2.6, 2.9]

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)

############################################# Obtain L2 error #############################################
data = np.load("my_data/surface_shapes_all_convergence_data.npz", allow_pickle=True)
records = data["records"]  # numpy array of dicts

data_ref = np.load(
    "my_data/surface_shapes_all_convergence_data_1.5M.npz", allow_pickle=True
)
records_ref = data_ref["records"]  # numpy array of dicts

n_points = 10 * 2 * 689  # scale up number of dofs for interpolation evaluation
r_eval = np.linspace(1.0, 23.0, n_points)
ref_vals_interp = interp1d(
    # 4th order pol. not available but using more points we should eliminate interp. error
    records_ref[-1]["xs"],
    records_ref[-1]["vals"],
    kind="quadratic",
)

L2norms = []

for record, color in zip(records, colors):
    xs = record["xs"]
    vals = record["vals"]
    mesh_id = record["mesh_id"]
    k_degree = record["k_degree"]

    # Interpolate to the common r_eval points
    vals_interp = interp1d(xs, vals, kind="quadratic")(r_eval)
    ref_interp = ref_vals_interp(r_eval)

    # Compute L2 error
    L2_error = np.sqrt(np.trapz((vals_interp - ref_interp) ** 2, r_eval))
    L2_rel = L2_error / np.sqrt(np.trapz(ref_interp**2, r_eval))
    print(
        f"Mesh {record['mesh_id']}, k={k_degree}: L2 error = {L2_error:.4e}, relative = {L2_rel:.2%}"
    )
    L2norms.append(L2_error)

############################################# Profile L2 error plotting #############################################
DoFs = [10568, 26010, 68842, 133612, 100584, 395463]
xs_plot = []
ys_plot = []
plt.figure(figsize=(8, 6))
plt.grid(True, which="both", alpha=0.5)
for val, DoF, color, record in zip(L2norms, DoFs, colors, records):
    xs_plot.append(np.sqrt(DoF))
    ys_plot.append(val)
    print(np.sqrt(DoF))
    mesh_id = record["mesh_id"]
    k_degree = record["k_degree"]
    plt.scatter(np.sqrt(DoF), val, color=color, label=f"M{mesh_id}, $p={k_degree}$")
plt.yscale("log")
plt.xscale("log")

eoc_label_added = False
pairs = [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5)]
for i, j in pairs:
    x1, x2 = xs_plot[i], xs_plot[j]
    y1, y2 = ys_plot[i], ys_plot[j]

    # Draw line between the two points
    if not eoc_label_added:
        plt.plot(
            [x1, x2], [y1, y2], color="black", linestyle="--", linewidth=1, label="EOC"
        )
        eoc_label_added = True
    else:
        plt.plot([x1, x2], [y1, y2], color="black", linestyle="--", linewidth=1)

    # Compute convergence order
    order = np.log(y2 / y1) / np.log(x2 / x1)

    # Midpoint for annotation (geometric mean works best in log–log)
    xm = np.sqrt(x1 * x2)
    ym = np.sqrt(y1 * y2)

    plt.text(
        xm,
        ym,
        f"{order:.2f}",
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )


# Plot formatting
plt.xlabel(r"$\sqrt{DoF}$")
plt.ylabel(
    r"$\|h(r,t\to\infty) - h_\text{fine}(r,t\to\infty)\|_{L^2(\Gamma_\chi^\text{free})}$"
)
plt.legend()
plt.tight_layout()
plt.savefig("oldroyd_B_convergence_plot_formal.pdf", dpi=300)


############################################# Profile plotting #############################################
plt.close()

plt.figure(figsize=(8, 6))
ax = plt.gca()

# Main plot formatting
ax.set_xlabel("$r$ [1]")
ax.set_ylabel(r"$h(r,t\to\infty)$ [1]")
ax.set_xlim(1, 2)
ax.set_ylim(0, 0.9)
ax.grid(True)

# Main curves
for record, color in zip(records, colors):
    xs = record["xs"]
    vals = record["vals"]
    mesh_id = record["mesh_id"]
    k_degree = record["k_degree"]
    ax.plot(
        xs,
        vals,
        color=color,
        label=f"M{mesh_id}, $p={k_degree}$",
        marker="x",
        markevery=k_degree,
    )

for record, color in zip(records_ref, ["black"]):
    xs = record["xs"]
    vals = record["vals"]
    mesh_id = record["mesh_id"]
    k_degree = record["k_degree"]
    ax.plot(xs, vals, color=color, label=f"M{mesh_id}, $p={k_degree}$")

ax.legend()

# Inset plot (magnifying glass)
axins = inset_axes(
    ax,
    width="50%",
    height="50%",
    loc="lower left",
    bbox_to_anchor=(0.075, 0.05, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)

for record, color in zip(records, colors):
    xs = record["xs"]
    vals = record["vals"]
    axins.plot(xs, vals, color=color, marker="x", markevery=k_degree)

for record, color in zip(records_ref, ["black"]):
    xs = record["xs"]
    vals = record["vals"]
    axins.plot(xs, vals, color=color, marker="x", markevery=k_degree)

# Zoom region
x1, x2 = 1.00, 1.05
axins.set_xlim(x1, x2)

# Automatically fit y-limits to data in zoom region
all_vals = []
for record in records:
    mask = (record["xs"] >= x1) & (record["xs"] <= x2)
    all_vals.extend(record["vals"][mask])

axins.set_ylim(min(all_vals), max(all_vals))

axins.grid(True)
axins.tick_params(labelsize=8)

# Draw connecting lines
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="black")

plt.tight_layout()
plt.savefig("oldroyd_B_convergence_plot.pdf", dpi=300)
