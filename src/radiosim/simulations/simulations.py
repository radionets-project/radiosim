import numpy as np
from numpy.random import default_rng
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from radiosim.io import Config
from radiosim.jets import create_jet
from radiosim.mojave import create_mojave
from radiosim.survey import create_survey
from radiosim.utils import (
    add_noise,
    create_grid,
    save_sky_distribution_bundle,
    setup_logger,
)
from radiosim.utils.utils import _save_mojave_bundle

__all__ = ["create_sky_distribution", "simulate_sky_distributions"]

LOGGER = setup_logger(__name__)

OVERALL_PROGRESS = Progress(
    TimeElapsedColumn(),
    TextColumn("{task.description}"),
)

SIMULATION_PROGRESS = Progress(
    TextColumn(
        "[bold blue]Progress for dataset {task.fields[name]}: {task.percentage:.0f}%"
    ),
    BarColumn(),
    TextColumn("({task.completed} of {task.total} steps done)"),
)
PROGRESS_GROUP = Group(
    OVERALL_PROGRESS,
    SIMULATION_PROGRESS,
)


def simulate_sky_distributions(conf, mode):
    seed = conf.general.seed
    if not seed:
        rng = default_rng()
    elif isinstance(seed, int):
        rng = default_rng(seed)

    bundles = [
        conf.dataset.bundles_train,
        conf.dataset.bundles_valid,
        conf.dataset.bundles_test,
    ]

    # Add task_id for overall progress, i.e. train, valid,
    # and test dataset tracking for a total of 3.
    overall_task_id = OVERALL_PROGRESS.add_task("", total=3)

    with Live(PROGRESS_GROUP):
        for idx, (opt, num_bundles) in enumerate(
            zip(["train", "valid", "test"], bundles)
        ):
            simulation_task_id = SIMULATION_PROGRESS.add_task(
                "", total=num_bundles, name=opt
            )
            create_sky_distribution(
                conf=conf,
                num_bundles=num_bundles,
                mode=mode,
                opt=opt,
                simulation_task_id=simulation_task_id,
                rng=rng,
            )
            top_descr = f"[bold #AAAAAA]({idx + 1} out of {3} datasets simulated)"
            OVERALL_PROGRESS.update(overall_task_id, description=top_descr)


def create_sky_distribution(
    conf: Config,
    num_bundles: int,
    mode: str,
    opt: str,
    simulation_task_id: int,
    rng=None,
) -> None:
    for _ in range(num_bundles):
        path = conf.paths.outpath / f"data_{mode}_{opt}.h5"
        grid = create_grid(conf.dataset.img_size, conf.dataset.bundle_size)

        if mode == "jet":
            sky, target = create_jet(grid, conf)
        elif mode == "survey":
            sky, target = create_survey(grid, conf)
        elif mode == "mojave":
            data, data_name = create_mojave(conf, rng)
            sky_bundle = data[0].copy()
            if conf.dataset.noise:
                sky_bundle = np.squeeze(
                    add_noise(
                        np.expand_dims(sky_bundle, axis=1),
                        conf.dataset.noise_level,
                    ),
                    axis=1,
                )

            _save_mojave_bundle(path, data=[sky_bundle, *data[1:]], data_name=data_name)
            SIMULATION_PROGRESS.update(simulation_task_id, advance=1)
            continue
        else:
            raise ValueError(
                f"Mode {mode} not found. Choose 'survey', "
                "'jet' or 'mojave' in config file"
            )

        sky_bundle = sky.copy()
        target_bundle = target.copy()
        if conf.dataset.noise and conf.dataset.noise_level > 0:
            sky_bundle = add_noise(sky_bundle, conf.dataset.noise_level)
            for img in sky_bundle:
                img -= img.min()
                img /= img.max()

        save_sky_distribution_bundle(path, sky_bundle, target_bundle)
        SIMULATION_PROGRESS.update(simulation_task_id, advance=1)
