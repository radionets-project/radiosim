import numpy as np
from scipy.spatial.transform import Rotation as R
from radiosim.utils import get_exp
from radiosim.gauss import gauss


def create_jet(img, bboxes=False):
    img_size = img.shape[2]
    center = img_size // 2
    comps = np.random.randint(3, 6)

    coord = []
    x = np.zeros(comps)
    y = np.zeros(comps)
    z = np.zeros(comps)
    amp = np.zeros(comps)
    sx = np.zeros(comps)
    sy = np.zeros(comps)
    base_amp = np.random.randint(50, 100)

    Ry = R.from_euler("y", np.random.uniform(0, 90), degrees=True).as_matrix()
    Rz = R.from_euler("z", np.random.uniform(0, 90), degrees=True).as_matrix()
    x_curve = np.zeros(comps)
    y_curve = np.zeros(comps)
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
                x_curve[i - 1], 0.05 * img_size + x_curve[i - 1]
            )
            y_curve[i] = np.random.uniform(
                y_curve[i - 1], 0.05 * img_size + y_curve[i - 1]
            )
            curve = np.array([x_curve[i], y_curve[i], 0])
            coord[i] += curve
        x[i], y[i], z[i] = coord[i] @ Ry @ Rz
        amp[i] = base_amp / (0.5 * i ** (np.random.normal(1, 0.2)) + 1)  # 1.09
        sx[i] = np.random.uniform((img_size ** 2) / 720, (img_size ** 2) / 360) * (
            0.5 * i + 1
        )
        sy[i] = np.random.uniform((img_size ** 2) / 720, (img_size ** 2) / 360) * (
            0.5 * i + 1
        )

    alpha = get_exp()
    comps += comps - 1
    amp = np.append(amp, amp[1:] * alpha)
    x = np.append(x, -x[1:])
    y = np.append(y, -y[1:])
    sx = np.append(sx, sx[1:])
    sy = np.append(sy, sy[1:])

    rand_center = np.random.normal(center, 1)

    jet_img = img[0, 0]
    for i in range(comps):
        g = gauss(
            img_size, x[i] + rand_center, y[i] + rand_center, sx[i], sy[i], amp[i]
        )
        jet_img += g
    jet_img_norm = jet_img / jet_img.max()
    return jet_img_norm