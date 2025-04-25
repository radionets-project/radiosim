def test_generate_proto_set():
    import h5py
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
                torch.from_numpy(hf["y"][()]), torch.tensor(images[str(img_size)])
            )
