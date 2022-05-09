from click.testing import CliRunner


def test_import():
    from radiosim.utils import load_data

    assert 1 == 1


def test_simulation():
    """
    Testing:
        Pipeline of simulation runs without error
    """
    from radiosim.scripts.start_simulation import main

    runner = CliRunner()
    result = runner.invoke(main, "./tests/simulate.toml")
    assert result.exit_code == 0


def test_image():
    """
    Testing:
        Image is scaled between 0 and 1
        Image has the shape as defined in the config
    """
    import toml
    import numpy as np
    from radiosim.utils import load_data

    config = toml.load("./tests/simulate.toml")
    image = load_data("./tests/simulate.toml")

    img_size = config["image_options"]["img_size"]
    bundle_size = config["image_options"]["bundle_size"]
    assert np.min(image) == 0
    assert np.max(image) == 1
    assert np.shape(image) == (bundle_size, 1, img_size, img_size)


def test_noise():
    """
    Testing:
        Noise has the same shape as input image
        Noise is scaled as defined in the config
    """
    import toml
    import numpy as np
    from radiosim.utils import load_data
    from radiosim.utils import add_noise

    config = toml.load("./tests/simulate.toml")
    image = load_data("./tests/simulate.toml")

    image_shape = np.shape(image)
    noise_level = config["image_options"]["noise_level"]
    noise = add_noise(np.zeros(image_shape), noise_level)
    assert noise.shape == image_shape
    assert np.isclose(np.abs(noise).max(), noise_level / 100)
