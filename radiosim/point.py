import numpy as np
from radiosim.gauss import create_gauss


def create_point_source_img(image, num_point_sources):
    if len(image.shape) == 3:
        image = image[None]
    for img in image:
        num_point_sources = np.random.randint(num_point_sources[0], num_point_sources[1])

        g, p_point, s_point = create_gauss(
            image, num_point_sources, image.shape[-1]
        )
        print(g.shape)

                # # crop image size
                # # mask = (
                # #     (comps[0] >= 0)
                # #     & (comps[0] <= img_size - 1)
                # #     & (comps[1] >= 0)
                # #     & (comps[1] <= img_size - 1)
                # # )
                # list_x = comps[0][mask]
                # list_y = comps[1][mask]
                # list_sx = comps[2][mask]
                # list_sy = comps[3][mask]
                # list_tag = comps[4][mask]
                # assert (
                #     list_x.shape
                #     == list_y.shape
                #     == list_sx.shape
                #     == list_sy.shape
                #     == list_tag.shape
                # )

                # source_list = np.array([list_x, list_y, list_sx, list_sy, list_tag])
                # g_fft = np.array(np.fft.fftshift(np.fft.fft2(g.copy())))

    return image, points, point_list
