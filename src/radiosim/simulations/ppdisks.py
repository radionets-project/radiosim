from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from radiosim.ppdisks import generate_proto_set


def simulate_ppdisks(config) -> None:
    metadata_params = config.metadata
    dataset_params = config.dataset

    rng = np.random.default_rng(seed=config.ppdisk.general.seed)

    outpath = Path(config.paths.outpath)

    if not outpath.is_dir():
        outpath.mkdir()

    file_prefix = dataset_params.file_prefix

    device = config.general.device
    verbose = config.general.verbose

    batch_num = dataset_params.batches
    batch_size = dataset_params.batch_size
    batches = {
        "train": int(batch_num * dataset_params.batches_train_ratio),
        "valid": int(batch_num * dataset_params.batches_valid_ratio),
        "test": int(batch_num * dataset_params.batches_test_ratio),
    }

    # resolution and pointings taken from FITS headers of DSHARP dataset
    # https://almascience.eso.org/almadata/lp/DSHARP/
    resolution = 0.00299999999999988
    pointings = [
        (252.1901583333, -14.27663888889),
        (242.8806334167, -18.64062026556),
        (246.6002916667, -24.27041666667),
        (269.088663625, -21.9562676325),
        (236.6862890833, -34.51002105694),
        (246.912515125, -23.97199338639),
        (246.6875916667, -24.38563138889),
        (239.1667003167, -22.02789646667),
        (246.484, -24.34672222222),
        (246.5986583334, -24.72051666667),
        (246.578625, -24.47213611111),
        (246.9142916667, -24.65431666667),
        (239.653748625, -22.95433409167),
        (252.313727625, -14.36918004583),
        (239.0382866667, -37.93514649278),
        (236.3035304583, -34.29194894083),
        (239.1762312125, -37.82110333889),
        (239.8185583333, -41.95297045306),
        (242.2576419583, -39.08689564417),
        (240.1854510899, -41.92538824062),
    ]

    for batch_type, num in batches.items():
        for i in tqdm(np.arange(num), desc=f"Generating {batch_type}"):
            protos, params = generate_proto_set(
                img_size=metadata_params.img_size,
                size=batch_size,
                alpha_range=metadata_params.alpha_range,
                ratio_range=metadata_params.ratio_range,
                size_ratio_range=metadata_params.size_ratio_range,
                device=device,
                seed=rng.integers(low=0, high=2**32 - 1),
                verbose=verbose,
            )
            metadata = []
            for j in np.arange(batch_size):
                pointing = rng.choice(pointings)
                metadata.append(
                    dict(
                        index=j,
                        cell_size=resolution,
                        src_ra=pointing[0],
                        src_dec=pointing[1],
                    )
                )

            with h5py.File(outpath / f"{file_prefix}_{batch_type}_{i}.h5", "w") as hf:
                hf.create_dataset("y", data=protos)
                hf.create_dataset(
                    "metadata", data=str(metadata), dtype=h5py.string_dtype()
                )
                hf.create_dataset("params", data=str(params), dtype=h5py.string_dtype())
