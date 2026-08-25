import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from radiosim.ppdisks.plotting.utils import configure_axes, configure_colorbar, get_norm


def fit_ellipse(
    image: np.ndarray, cut_ratio: float = 0.01, show_progress: bool = False
) -> tuple[dict, dict]:
    """
    Fit an ellipse and return its parameters by performing a PCA.
    The input image is built as a pixel-based point representation
    which contains an amount of points based on the pixel's values.

    Parameters
    ----------

    image: np.ndarray
        The input image to perform the analysis on. Should contain a structure
        to which fitting an ellipse is sensible.
        This requires the ellipse to be the brightest object in the image.
        Otherwise this fit will not output representative values as the selection
        of the ellipse's pixels is intensity-based.

    cut_ratio: float, optional
        The multiple of the maximum image value which will be used to select the
        pixels to consider for the analysis. The higher the value, the more pixels
        will we selected and thus the analysis will become more complex.
        This should be chosen depending on the input image. Depending on the intensity
        gradients it might be necessary to select more values to clearly identify the
        ellipse. Default is ``0.01``.

    show_progress: bool, optional
        Whether to show a progress bar to indicate the point model building process.
        Default is ``False``.

    Returns
    -------

    tuple[dict, dict]:
        The output contains the ellipse parameters in the first dictionary.
        It contains the keys
            - ``semi_min`` -> length of semi-minor axis
            - ``semi_maj`` -> length of the semi-major axis
            - ``incl`` -> inclination
            - ``posang`` -> position angle

        The second dictionary contains the output of the PCA and the raw and transformed
        point representations:
            - ``center`` -> pixel coordinates of the ellipse center
            - ``pca_obj`` -> ``sklearn.decomposition.PCA`` object for the image
            - ``image_cut`` -> image with applied intensity cut
            - ``X`` -> point representation of the input image
            - ``X_transform`` -> point representation of the input image in principal
                                 component space.

    """

    normalization_limit = image.max() * cut_ratio
    image /= normalization_limit
    image = np.floor(image).astype(np.int64)

    total = np.sum(image)
    X = np.empty((total, 2))

    pix = np.argwhere(image > 0)
    current_row = 0

    for i in tqdm(np.arange(pix.shape[0])):
        vec = pix[i]
        num_reps = image[vec[0], vec[1]]
        X[current_row : current_row + num_reps] = np.tile(vec, (num_reps, 1))
        current_row += num_reps

    pca = PCA()
    pca.fit(X)

    center = np.unravel_index(image.argmax(), image.shape)

    X_transform = pca.transform(X)
    b_min = np.abs(X_transform[:, 1].max() - X_transform[:, 1].min()) / 2
    b_maj = np.abs(X_transform[:, 0].max() - X_transform[:, 0].min()) / 2

    # Angle relations from: https://doi.org/10.3847/2041-8213/aaf740 p. 2
    incl = np.arccos(b_min / b_maj)
    posang = np.pi / 2 - np.arctan2(*pca.components_[0])

    ellipse_parameters = {
        "semi_min": b_min,
        "semi_maj": b_maj,
        "incl": incl,
        "posang": posang,
    }
    pca_output = {
        "center": center,
        "pca_obj": pca,
        "image_cut": image,
        "X": X,
        "X_transform": X_transform,
    }

    return ellipse_parameters, pca_output


def plot_ellipse_fit(
    ellipse_parameters: dict,
    pca_output: dict,
    show_transformed: bool = False,
    image_cmap: str | matplotlib.colors.Colormap = "inferno",
    image_norm: str | matplotlib.colors.Normalize | None = None,
    line_cmap: str = "viridis",
    point_color: str = "royalblue",
    point_size: float = 0.0003,
    show_points: bool = True,
    show_pcs: bool = True,
    show_image: bool = True,
    show_ellipse_axes: bool = True,
    show_ellipse: bool = True,
    legend_loc: str = "best",
    patch_parameters: dict | None = None,
    arrow_args: dict | None = None,
    show_legend: bool = True,
    fig: matplotlib.figure.Figure | None = None,
    fig_args: dict | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    arrow_args = {"width": 1} if arrow_args is None else arrow_args
    patch_parameters = (
        {
            "fill": False,
            "ec": "white" if not show_transformed else "black",
        }
        if patch_parameters is None
        else patch_parameters
    )

    colors = plt.colormaps.get_cmap(line_cmap)(np.linspace(0.2, 0.8, 2))
    fig, ax = configure_axes(fig=fig, ax=ax, fig_args=fig_args)

    if not show_transformed:
        if show_pcs:
            pca = pca_output["pca_obj"]
            center = pca_output["center"]
            pc1_vec = (
                pca.components_[0]
                * pca.explained_variance_ratio_[0]
                * ellipse_parameters["semi_maj"]
            )
            pc2_vec = (
                pca.components_[1]
                * pca.explained_variance_ratio_[1]
                * ellipse_parameters["semi_maj"]
            )

            ax.arrow(
                x=center[1],
                y=center[0],
                dx=pc1_vec[1],
                dy=pc1_vec[0],
                color=colors[0],
                zorder=10,
                label="PC1",
                **arrow_args,
            )
            ax.arrow(
                x=center[1],
                y=center[0],
                dx=pc2_vec[1],
                dy=pc2_vec[0],
                color=colors[1],
                zorder=10,
                label="PC2",
                **arrow_args,
            )

        if show_ellipse_axes:
            semi_min = ellipse_parameters["semi_min"]
            semi_maj = ellipse_parameters["semi_maj"]

            pca = pca_output["pca_obj"]
            center = pca_output["center"]
            pc1_vec = (
                pca.components_[0]
                / np.linalg.norm(pca.components_[0])
                * ellipse_parameters["semi_maj"]
            )
            pc2_vec = (
                pca.components_[1]
                / np.linalg.norm(pca.components_[1])
                * ellipse_parameters["semi_min"]
            )

            ax.arrow(
                x=center[1],
                y=center[0],
                dx=pc1_vec[1],
                dy=pc1_vec[0],
                color=colors[0],
                zorder=10,
                label="Semi-Major axis",
                head_width=0,
                head_length=0,
            )
            ax.arrow(
                x=center[1],
                y=center[0],
                dx=pc2_vec[1],
                dy=pc2_vec[0],
                color=colors[1],
                zorder=10,
                label="Semi-Minor axis",
                head_width=0,
                head_length=0,
            )

        X = pca_output["X"]
        if show_points:
            ax.scatter(X[:, 1], X[:, 0], s=point_size, color=point_color)

        if show_image:
            im = ax.imshow(
                pca_output["image_cut"],
                origin="lower",
                norm=get_norm(norm=image_norm),
                cmap=image_cmap,
            )
            configure_colorbar(mappable=im, ax=ax, fig=fig, label="Points / Pixel")

        ax.set_xlim(X[:, 1].min(), X[:, 1].max())
        ax.set_ylim(X[:, 0].min(), X[:, 0].max())

        if show_ellipse:
            ax.add_patch(
                Ellipse(
                    xy=pca_output["center"][::-1],
                    width=ellipse_parameters["semi_maj"] * 2,
                    height=ellipse_parameters["semi_min"] * 2,
                    angle=np.rad2deg(ellipse_parameters["posang"]),
                    **patch_parameters,
                )
            )

    else:
        X_transform = pca_output["X_transform"]
        ax.scatter(
            X_transform[:, 0], X_transform[:, 1], s=point_size, color=point_color
        )

        if show_ellipse_axes:
            semi_min = ellipse_parameters["semi_min"]
            semi_maj = ellipse_parameters["semi_maj"]

            ax.plot([0, semi_maj], [0, 0], color=colors[0], label="Semi-Major axis")
            ax.plot([0, 0], [0, semi_min], color=colors[1], label="Semi-Minor axis")

        if show_ellipse:
            ax.add_patch(
                Ellipse(
                    xy=(0, 0),
                    width=ellipse_parameters["semi_maj"] * 2,
                    height=ellipse_parameters["semi_min"] * 2,
                    **patch_parameters,
                )
            )

    ax.set_aspect("equal")

    if show_legend:
        ax.legend(loc=legend_loc)

    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")

    return fig, ax
