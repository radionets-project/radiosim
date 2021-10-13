import numpy as np


def gauss(img_size, mx, my, sx, sy, amp=0.01):
    x = np.arange(img_size)[None].astype(np.float)
    y = x.T
    g = amp * np.exp(-((y - my) ** 2) / sy).dot(np.exp(-((x - mx) ** 2) / sx))
    return g


def create_gauss(img, num_sources, img_size=63, sym=True):
    mx = np.random.randint(1, img_size, size=(num_sources))
    my = np.random.randint(1, img_size, size=(num_sources))
    rng = np.random.default_rng()
    amp = rng.uniform(1e-2, 1, num_sources)
    sx = np.random.randint(
        round(1 / 8 * (img_size ** 2) / 720),
        1 / 2 * (img_size ** 2) / 360,
        size=(num_sources),
    )
    if sym:
        sy = sx
    else:
        sy = np.random.randint(
            round(1 / 8 * (img_size ** 2) / 720),
            1 / 2 * (img_size ** 2) / 360,
            size=(num_sources),
        )

    idx = []
    point_s = []
    for n in range(num_sources):
        if img[mx[n], my[n]] <= 5e-10:
            g = gauss(img_size, mx[n], my[n], sx[n], sy[n], amp[n])
            img += g
            point_s += [g]
        else:
            idx.append(n)
    img_norm = img / img.max()
    point_s_norm = point_s / img.max()
    amp = np.delete(amp, idx)
    mx = np.delete(mx, idx)
    my = np.delete(my, idx)
    sx = np.delete(sx, idx)
    sy = np.delete(sy, idx)
    point_list = np.array(
        [
            amp / img.max(),
            mx,
            my,
            np.sqrt(sx),
            np.sqrt(sy),
            np.ones(len(mx)),
        ],
    ).T
    return img_norm, point_s_norm, point_list
