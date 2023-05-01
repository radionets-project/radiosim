import click
import multiprocessing
from tqdm import tqdm

from radiosim.jet import create_jet
from radiosim.survey import create_survey
from radiosim.utils import (
    add_noise,
    create_grid,
    save_sky_distribution_bundle,
)


def simulate_sky_distributions(conf):
    for opt in ["train", "valid", "test"]:
        csd = create_sky_distribution(
            conf=conf,
            opt=opt,
        )
        csd()


class create_sky_distribution:
    def __init__(self, conf, opt):
        self.conf = conf
        self.opt = opt

    def __call__(self):
        n_bundels = self.conf["bundles_" + self.opt]
        n_cores = int(multiprocessing.cpu_count() * 0.5)  # use 50% of available cores
        if n_cores == 1 or not self.conf["multiprocessing"]:
            for i in tqdm(range(n_bundels)):
                self.sky_distribution(i)
        else:
            with multiprocessing.Pool(n_cores) as p:
                _ = list(
                    tqdm(
                        p.imap(self.sky_distribution, range(n_bundels)), total=n_bundels
                    )
                )

    def sky_distribution(self, i: int):
        """Create and save the sky distribution

        Parameters
        ----------
        i: int
            n-th sky distribution to be saved
        """
        grid = create_grid(self.conf["img_size"], self.conf["bundle_size"])
        if self.conf["mode"] == "jet":
            sky, target = create_jet(grid, self.conf)
        elif self.conf["mode"] == "survey":
            sky, target = create_survey(grid, self.conf)
        else:
            click.echo("Given mode not found. Choose 'survey' or 'jet' in config file")

        sky_bundle = sky.copy()
        target_bundle = target.copy()
        if self.conf["noise"] and self.conf["noise_level"] > 0:
            sky_bundle = add_noise(sky_bundle, self.conf["noise_level"])
            for img in sky_bundle:
                img -= img.min()
                img /= img.max()
        path = self.conf["outpath"] + "/samp_" + self.opt + "_" + str(i) + ".h5"
        save_sky_distribution_bundle(path, sky_bundle, target_bundle)
