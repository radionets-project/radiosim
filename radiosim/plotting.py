import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_distribution(img):
    plt.figure()
    plt.imshow(img, cmap="inferno")
    plt.colorbar(label="brightness distribution / a.u.")
    plt.ylabel("pixels")
    plt.xlabel("pixels")


def plot_positions(source_list):
    jet_l = source_list[source_list[:, -1] == 0, :]
    point_l = source_list[source_list[:, -1] == 1, :]
    if jet_l.size > 0:
        plt.plot(
            jet_l[:, 1],
            jet_l[:, 2],
            linestyle="none",
            marker="x",
            markersize=4,
            markeredgewidth=1.3,
            label="jet component",
        )
    if point_l.size > 0:
        plt.plot(
            point_l[:, 1],
            point_l[:, 2],
            linestyle="none",
            color="#FFD358",
            marker="x",
            markersize=4,
            markeredgewidth=1.3,
            markeredgecolor="#FF9119",
            label="point source",
        )


def plot_legend():
    legend = plt.legend(edgecolor="white")
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 1, 0))
    plt.setp(legend.get_texts(), color="w")


def plot_single_example(img, source_list):
    plt.figure()
    plot_distribution(img)
    plot_positions(source_list)
    plt.ylabel("pixels")
    plt.xlabel("pixels")
    plot_legend()
    plt.tight_layout()


def plot_overview(f):
    num_exp = len(f) / 3
    i = np.random.choice(np.arange(num_exp, dtype="int"), 6, replace=False)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(f["sky" + str(i[0])])
    ax1.axis("off")
    ax2.imshow(f["sky" + str(i[1])])
    ax2.axis("off")
    ax3.imshow(f["sky" + str(i[2])])
    ax3.axis("off")
    ax4.imshow(f["sky" + str(i[3])])
    ax4.axis("off")
    ax5.imshow(f["sky" + str(i[4])])
    ax5.axis("off")
    ax6.imshow(f["sky" + str(i[5])])
    ax6.axis("off")

    plt.tight_layout()


def create_simulation_overview(sim_conf):
    path = Path(sim_conf["outpath"])
    bundles = np.array([x for x in path.iterdir() if re.findall(".h5", x.name)])
    bundle = np.random.choice(bundles)
    f = h5py.File(bundle, "r")

    img = f["sky0"]
    s_list = f["list0"][()]
    plot_single_example(img, s_list)
    path_overview = path / "overview"
    path_overview.mkdir(exist_ok=True)
    plt.savefig(path_overview / "single_source.pdf")
    plot_overview(f)
    plt.savefig(path_overview / "overview.pdf")
