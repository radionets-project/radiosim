import numpy as np
from radiosim.gauss import create_gauss


def create_point_source_img(image, num_point_sources):
    if image.shape[1] == 3:
        image = image[:, 0]

    points_imgs = []
    points_single = []
    points_lists = []
    for img in image:
        num_points = np.random.randint(num_point_sources[0], num_point_sources[1])
        point_img, point_s, point_l = create_gauss(img, num_points, image.shape[-1])

        points_imgs.append(point_img)
        points_single.append(point_s)
        points_lists.append(point_l)

    return (
        points_imgs,
        np.array(points_single, dtype=object),
        np.array(points_lists, dtype=object),
    )
