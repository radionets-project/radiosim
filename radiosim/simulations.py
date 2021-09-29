from tqdm import tqdm
from radiosim.utils import create_grid
from radiosim.jet import create_jet


def simulate_sky_distributions(sim_conf):
    for opt in ["train", "valid", "test"]:
        create_sky_distribution(
            img_size=sim_conf["img_size"],
            bundle_size=sim_conf["bundle_size"],
            num_bundles=sim_conf["bundles_" + str(opt)],
            path=sim_conf["outpath"],
            option=opt,
        )


def create_sky_distribution(num_bundles, img_size, bundle_size):
    for i in tqdm(range(num_bundles)):
        grid = create_grid(img_size, bundle_size)
        jet = create_jet(grid)
        print(jet.shape)

        # images = bundle.copy()

        # if noise:
        #     images = add_noise(images, noise_level)

        # bundle_fft = np.array([np.fft.fftshift(np.fft.fft2(img)) for img in images])
        # bundle_fft = add_white_noise(bundle_fft)
        # path = adjust_outpath(data_path, "/fft_" + option)
        # save_fft_pair(path, bundle_fft, bundle, list_sources)
