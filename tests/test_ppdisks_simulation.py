def test_generate_proto_set():
    import h5py
    import numpy as np
    import torch

    from radiosim.ppdisks import generate_proto_set

    sizes = [1, 2]
    img_sizes = [128, 512]
    size_ratio_ranges = [(0.1, 1), (0.1, 0.5), (0.5, 1)]
    seed = 42

    images = dict()

    for img_size in img_sizes:
        images[str(img_size)] = []

    for size in sizes:
        for img_size in img_sizes:
            for size_ratio_range in size_ratio_ranges:
                protos, params = generate_proto_set(
                    img_size=img_size,
                    size=size,
                    size_ratio_range=size_ratio_range,
                    seed=seed,
                )
                images[str(img_size)].extend(protos)

    for img_size in img_sizes:
        with h5py.File(f"./tests/ppdisks_{img_size}.h5", "r") as hf:
            assert torch.allclose(
                torch.from_numpy(hf["y"][()]),
                torch.from_numpy(np.array(images[str(img_size)])),
            )


def test_generate_proto_no_seed():
    from radiosim.ppdisks import create_proto

    img_sizes = [128, 512]
    alphas = [0.0, 180.0]
    size_ratios = [0.1, 1.0]
    ratios = [3, 15]

    for img_size in img_sizes:
        for alpha in alphas:
            for ratio in ratios:
                for size_ratio in size_ratios:
                    create_proto(
                        img_size=img_size,
                        alpha=alpha,
                        ratio=ratio,
                        size_ratio=size_ratio,
                        seed=None,
                    )
