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

from radiosim.jet import create_jet
from radiosim.mojave import create_mojave
from radiosim.survey import create_survey
from radiosim.utils import (
    add_noise,
    adjust_outpath,
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


def simulate_sky_distributions(conf):
    if conf["mode"] == "mojave":
        seed = conf["seed"]
        if seed == "none":
            rng = default_rng()
        elif isinstance(seed, int):
            rng = default_rng(seed)
        else:
            raise TypeError('seed has to be int or "none"')
    else:
        rng = None

    # Add task_id for overall progress, i.e. train, valid,
    # and test dataset tracking for a total of 3.
    overall_task_id = OVERALL_PROGRESS.add_task("", total=3)

    with Live(PROGRESS_GROUP):
        for idx, opt in enumerate(["train", "valid", "test"]):
            simulation_task_id = SIMULATION_PROGRESS.add_task(
                "", total=conf["bundles_" + opt], name=opt
            )
            create_sky_distribution(
                conf=conf,
                opt=opt,
                simulation_task_id=simulation_task_id,
                rng=rng,
            )
            top_descr = f"[bold #AAAAAA]({idx + 1} out of {3} datasets simulated)"
            OVERALL_PROGRESS.update(overall_task_id, description=top_descr)


def create_sky_distribution(conf, opt: str, simulation_task_id: int, rng=None) -> None:
    for _ in range(conf["bundles_" + opt]):
        path = adjust_outpath(conf["outpath"], f"/data_{conf['mode']}_" + opt)
        grid = create_grid(conf["img_size"], conf["bundle_size"])

        if conf["mode"] == "jet":
            sky, target = create_jet(grid, conf)
        elif conf["mode"] == "survey":
            sky, target = create_survey(grid, conf)
        elif conf["mode"] == "mojave":
            data, data_name = create_mojave(conf, rng)
            sky_bundle = data[0].copy()
            if conf["noise"]:
                sky_bundle = np.squeeze(
                    add_noise(np.expand_dims(sky_bundle, axis=1), conf["noise_level"]),
                    axis=1,
                )
                # for img in data:
                #     img -= img.min()
                #     img /= img.max()
            _save_mojave_bundle(path, data=[sky_bundle, *data[1:]], data_name=data_name)
            SIMULATION_PROGRESS.update(simulation_task_id, advance=1)
            continue
        else:
            LOGGER.warning(
                "Given mode not found. Choose 'survey', "
                "'jet' or 'mojave' in config file"
            )

        sky_bundle = sky.copy()
        target_bundle = target.copy()
        if conf["noise"] and conf["noise_level"] > 0:
            sky_bundle = add_noise(sky_bundle, conf["noise_level"])
            for img in sky_bundle:
                img -= img.min()
                img /= img.max()

        save_sky_distribution_bundle(path, sky_bundle, target_bundle)
        SIMULATION_PROGRESS.update(simulation_task_id, advance=1)
