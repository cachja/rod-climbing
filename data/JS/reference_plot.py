import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# User-provided lists
colors = ["green", "orange"]
labels = [1.7, 2.1, 2.6, 2.9]
names_exp = ["1.7_Debb_exp", "2.1_BJ_exp", "2.6_BJ_exp", "2.9_BJ_exp"]
xis = [0.284, 0.318]
our_as = [0.83, 0.815, 0.805, 0.805]

for omega, name_exp, our_a in zip(labels, names_exp, our_as):
    plt.figure(figsize=(8, 6))

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )

    ############## Expreriment data
    filename = f"JS_ref_data/{name_exp}.csv"
    # Load CSV with ; separator and , decimal
    df = pd.read_csv(filename, sep=";", decimal=",", header=None, names=["x", "y"])

    # Prepend 0 to x, and duplicate first y for y
    df["x"] = pd.concat([pd.Series([1]), df["x"]], ignore_index=True)
    df["y"] = pd.concat([pd.Series([df["y"].iloc[0]]), df["y"]], ignore_index=True)

    # Plot curve
    if "Debb" in names_exp:
        plt.scatter(df["x"], df["y"], color="black", label=f"Debbaut and Hocq (1992)")
    else:
        plt.scatter(df["x"], df["y"], color="black", label=f"Beavers and Joseph (1975)")

    ######## My model B data
    data_B = np.load(
        f"my_data/surface_shapes_all_{omega}_a0.682.npz", allow_pickle=True
    )
    records_B = data_B["records"]

    for record, color in zip(records_B, colors):
        xs = record["xs"]
        vals = record["vals"]
        a = record["a"]

        plt.plot(
            xs,
            vals,
            color=color,
            label=rf"Present work, model B, with $a$ = {np.round(a,3)}",
            marker="d",
            markevery=4,
            markersize=2,
        )

    ########### Luo data
    if omega != 2.9:
        for xi in xis:
            filename = f"JS_ref_data/{omega}_Luo_{xi}.csv"
            # Load CSV with ; separator and , decimal
            df = pd.read_csv(
                filename, sep=";", decimal=",", header=None, names=["x", "y"]
            )

            # Prepend 0 to x, and duplicate first y for y
            df["x"] = pd.concat([pd.Series([1]), df["x"]], ignore_index=True)
            df["y"] = pd.concat(
                [pd.Series([df["y"].iloc[0]]), df["y"]], ignore_index=True
            )

            # Plot curve
            if xi == 0.284:
                plt.plot(
                    df["x"],
                    df["y"],
                    color="red",
                    label=rf"Luo (1999), with $a$ = {np.round(1-xi,3)}",
                    linestyle="dashdot",
                )
            else:
                plt.plot(
                    df["x"],
                    df["y"],
                    color="blue",
                    label=rf"Luo (1999), with $a$ = {np.round(1-xi,3)}",
                    linestyle=(0, (3, 5, 1, 5)),
                )

    ########### Figueiredo data
    for xi in xis:
        filename = f"JS_ref_data/{omega}_Figuo_{xi}.csv"
        # Load CSV with ; separator and , decimal
        df = pd.read_csv(filename, sep=";", decimal=",", header=None, names=["x", "y"])

        # Prepend 0 to x, and duplicate first y for y
        df["x"] = pd.concat([pd.Series([1]), df["x"]], ignore_index=True)
        df["y"] = pd.concat([pd.Series([df["y"].iloc[0]]), df["y"]], ignore_index=True)

        # Plot curve
        if xi == 0.284:
            plt.plot(
                df["x"],
                df["y"],
                color="black",
                label=rf"Figueiredo et al. (2016), with $a$ = {np.round(1-xi,3)}",
                linestyle="--",
            )
        else:
            plt.plot(
                df["x"],
                df["y"],
                color="black",
                label=rf"igueiredo et al. (2016), with $a$ = {np.round(1-xi,3)}",
                linestyle=":",
            )

    ######## My model B data
    data_A = np.load(
        f"my_data/surface_shapes_all_{omega}_a{our_a}.npz", allow_pickle=True
    )
    records_A = data_A["records"]

    for record, color in zip(records_A, ["cyan"]):
        xs = record["xs"]
        vals = record["vals"]
        a = record["a"]

        plt.plot(
            xs,
            vals,
            color=color,
            label=rf"Present work, model A, with $a$ = {a:.3f}",
            marker="d",
            markevery=4,
            markersize=2,
        )

    # Plot formatting
    plt.ylabel(r"$h(r,t\to\infty)$ [1]")
    plt.xlabel("$r$ [1]")
    plt.xlim(1, 2)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"JSref_omega{omega}.pdf", dpi=300)
    plt.close()
