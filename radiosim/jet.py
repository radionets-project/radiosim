import numpy as np
from scipy.spatial.transform import Rotation as R
from radiosim.utils import get_exp
from radiosim.gauss import gauss


def create_jet(image, num_comps, train_type):
    if len(image.shape) == 3:
        image = image[None]

    jets = []
    jet_comps = []
    source_lists = []
    jet_counts = []
    for img in image:
        img_size = img.shape[-1]
        center = img_size // 2
        comps = np.random.randint(num_comps[0], num_comps[1] + 1)

        coord = []
        x = np.zeros(num_comps[1])
        y = np.zeros(num_comps[1])
        z = np.zeros(num_comps[1])
        amp = np.zeros(num_comps[1])
        sx = np.zeros(num_comps[1])
        sy = np.zeros(num_comps[1])
        #base_amp = np.random.randint(50, 100)

        Ry = R.from_euler("y", np.random.uniform(0, 90), degrees=True).as_matrix()
        Rz = R.from_euler("z", np.random.uniform(0, 90), degrees=True).as_matrix()
        x_curve = np.zeros(num_comps[1])
        y_curve = np.zeros(num_comps[1])
        for i in range(comps):
            coord.append(
                np.array(
                    [
                        2 * i * img_size * 0.04
                        + np.random.uniform(-0.01 * img_size, 0.01 * img_size),
                        0,
                        0,
                    ]
                )
            )
            if i != 0:
                x_curve[i] = np.random.uniform(
                    x_curve[i - 1] + 0.03 * img_size, x_curve[i - 1] + 0.045 * img_size
                )
                y_curve[i] = np.random.uniform(
                    y_curve[i - 1] + 0.03 * img_size, y_curve[i - 1] + 0.045 * img_size
                )
                curve = np.array([x_curve[i], y_curve[i], 0])
                coord[i] += curve
            x[i], y[i], z[i] = coord[i] @ Ry @ Rz
            amp[i] = np.exp(-i * np.random.normal(1.3, 0.4))
            #amp[i] = base_amp / (0.5 * i ** (np.random.normal(1, 0.2)) + 1)  # 1.09
            sx[i] = np.random.uniform((img_size ** 2) / 1500, (img_size ** 2) / 500) * (
                0.5 * i + 1
            )
            sy[i] = np.random.uniform((img_size ** 2) / 1500, (img_size ** 2) / 500) * (
                0.5 * i + 1
            )

        alpha = get_exp()
        # comps += comps - 1
        amp = np.append(amp, amp[1:] * alpha)
        x = np.append(x, -x[1:]) + np.random.normal(center, 1)
        y = np.append(y, -y[1:]) + np.random.normal(center, 1)
        sx = np.append(sx, sx[1:])
        sy = np.append(sy, sy[1:])

        jet_img = img[0]
        jet_comp = []
        for i in range(2 * num_comps[1] - 1):
            comp_dropout = np.random.uniform() < 0.3
            if comp_dropout and i != 0:
                # amp[i] = x[i] = y[i] = sx[i] = sy[i] = 0
                amp[i] = 0
                jet_comp += [np.zeros((img_size, img_size))]
            else:
                if amp[i] == 0:
                    jet_comp += [np.zeros((img_size, img_size))]
                    # x[i] = y[i] = sx[i] = sy[i] = 0
                else:
                    g = gauss(
                        img_size,
                        x[i],
                        y[i],
                        sx[i],
                        sy[i],
                        amp[i],
                    )
                    jet_comp += [g]
                    jet_img += g
        jet_img_norm = jet_img / jet_img.max()
        jet_comp_norm = jet_comp / jet_img.max()
        for i in range(num_comps[1] - 1):
            jet_comp_norm[i+1] += jet_comp_norm[num_comps[1]]
            jet_comp_norm = np.delete(jet_comp_norm, num_comps[1], axis=0)
        jet_comp_norm = np.concatenate((jet_comp_norm, (1 - jet_img_norm)[None, :, :]))
        jets.append(jet_img_norm)
        jet_comps.append(jet_comp_norm)
        
        # scale the parameters between 0 and 1
        # amp /= jet_img.max()
        x /= img_size
        y /= img_size
        sx /= img_size**2 / 500 * (0.5 * (num_comps[1] - 1) + 1)
        sy /= img_size**2 / 500 * (0.5 * (num_comps[1] - 1) + 1)
        
        source_list = np.array(
            [
                amp,
                x,
                y,
                sx,
                sy
            ]
        ).T

        if train_type == 'list':
            source_list = source_list[source_list[:, 0].argsort()]
        elif train_type == 'counts':
            jet_counts.append(np.sum(amp > 0.05) / (2 * num_comps[1] - 1))
        source_lists.append(source_list)
    
    if train_type == 'counts':
        jet_comps = jet_counts
    return (
        np.array(jets)[:, None, :, :],
        np.array(jet_comps),
        np.array(source_lists),
    )
