import matplotlib.pyplot as plt


def plot_distribution(img):
    plt.figure()
    plt.imshow(img, cmap="inferno")
    plt.colorbar(label="brightness distribution / a.u.")
    plt.ylabel("pixels")
    plt.xlabel("pixels")


def plot_positions(source_list):
    jet_l = source_list[source_list[:, -1] == 0, :]
    point_l = source_list[source_list[:, -1] == 1, :]
    plt.plot(
        jet_l[:, 1],
        jet_l[:, 2],
        linestyle="none",
        marker="x",
        markersize=4,
        markeredgewidth=1.3,
        label="jet component",
    )
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
