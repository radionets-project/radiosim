import numpy as np


def gauss(img_size, mx, my, sx, sy, amp=0.01):
    x = np.arange(img_size)[None].astype(np.float)
    y = x.T
    g = amp * np.exp(-((y - my) ** 2) / sy).dot(np.exp(-((x - mx) ** 2) / sx))
    return g
