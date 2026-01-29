import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from scipy.special import k0
from scipy.integrate import quad

name = "my_data"
ABCD = ["We0.2G0.07", "We0.2G2", "We0.04G70", "We0.01G140"]
abcd_list = ["A", "B", "C", "D"]

for folder_name, case_name in zip(ABCD, abcd_list):
    interval_list = ["0.1-0.5", "1-2", "3-4", "5.0-7.5", "10-15", "20-25"]
    Bond_all = []
    Height_all = []

    for interval in interval_list:
        data = np.load(f"{name}/{folder_name}/heatmap_data_{interval}.npz")
        Bond_all.append(data["Bond"])  # 1D
        Height_all.append(data["HoverR"])  # 2D (Lambda × Bond)

    Lambda_vals = data["Lambda"]  # same for all files
    Bond = np.concatenate(Bond_all)  # shape (nBond_total,)
    Height = np.concatenate(Height_all, axis=1)  # shape (nLambda, nBond_total)

    sort_idx = np.argsort(Bond)
    Bond_list = Bond[sort_idx]
    Height = Height[:, sort_idx]

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )

    fig, ax = plt.subplots()
    vmax = np.nanmax(Height)
    vmin = np.nanmin(Height)

    # TwoSlopeNorm: zero-centered
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # Plot heatmap
    pcm = ax.pcolormesh(
        Bond_list, Lambda_vals, Height, shading="auto", cmap="seismic", norm=norm
    )

    # Colorbar: piecewise-linear ticks aligned to TwoSlopeNorm
    num_neg_ticks = 4
    num_pos_ticks = 4

    # Generate normalized positions (0 = bottom, 0.5 = zero, 1 = max)
    neg_ticks_norm = np.linspace(0, 0.5, num_neg_ticks, endpoint=False)
    pos_ticks_norm = np.linspace(0.5, 1, num_pos_ticks + 1)

    # Map back to data values using norm.inverse
    neg_ticks = norm.inverse(neg_ticks_norm)
    pos_ticks = norm.inverse(pos_ticks_norm)
    ticks = np.concatenate([neg_ticks, pos_ticks])

    # Create colorbar with ticks
    cbar = fig.colorbar(pcm, ax=ax, ticks=ticks)
    cbar.set_label("$H$ [1]")

    ax.set_xlim(0, 25)
    ax.set_ylim(None, 2.16)

    # Plot analytical result
    B_vals = np.linspace(0.2, 25, 200)

    def zero_contour(B):
        f1 = lambda R: R ** (-1) * k0(np.sqrt(B) * R)
        f2 = lambda R: R ** (-3) * k0(np.sqrt(B) * R)
        Int1, _ = quad(f1, 1, np.inf)
        Int2, _ = quad(f2, 1, np.inf)
        return Int1 / Int2

    ZeroContour = np.array([zero_contour(B) for B in B_vals])

    (line1,) = ax.plot(
        B_vals,
        ZeroContour,
        color="black",
        linewidth=5.0,
        linestyle="--",
        label="Ruangkriengsin et al. 2025",
    )

    # Plot our zero contour
    cs = ax.contour(
        Bond_list,
        Lambda_vals,
        Height,
        levels=[0],
        colors="black",
        linewidths=5,
    )
    ax.clabel(cs, fmt="$H = 0$", inline=True)

    # Create a proxy Line2D for the contour for the legend
    import matplotlib.lines as mlines

    contour_proxy = mlines.Line2D(
        [], [], color="black", linewidth=5, label="Current work"
    )

    # Combine into single legend
    ax.legend(handles=[line1, contour_proxy], handlelength=4.5)

    ax.set_xlabel("$B$ [1]")
    ax.set_ylabel(r"$\Lambda = 4 \beta_p(1-2\alpha)We/Re$ [1]")
    plt.savefig(f"giesekus_heatmap_{case_name}.pdf")
