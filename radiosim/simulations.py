import click
import multiprocessing
from tqdm import tqdm
from radiosim.utils import (
    create_grid,
    add_noise,
    adjust_outpath,
    save_sky_distribution_bundle,
)
from radiosim.jet import create_jet
from radiosim.survey import create_survey


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
        print("Number of cpu cores:", n_cores)

        if n_cores == 1:
            for _ in tqdm(range(n_bundels)):
                self.multiprocessing_func(0)
        else:
            print()
            with multiprocessing.Pool(n_cores) as p:
                # r = list(tqdm(p.imap(self.multiprocessing_func, range(n_bundels)), total=n_bundels))  # sometimes leads to error in tqdm: BlockingIOError: [Errno 11] Unable to create file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')
                for _ in p.imap(self.multiprocessing_func, range(n_bundels)):
                    continue

    def multiprocessing_func(self, _):
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
        path = adjust_outpath(self.conf["outpath"], "/samp_" + self.opt)
        save_sky_distribution_bundle(path, sky_bundle, target_bundle)
