import numpy as np
from radiosim.utils import get_exp, pol2cart
from radiosim.gauss import twodgaussian, gauss


def create_jet(grid, num_comps, train_type):
    """
    Creates the clean jets with all its components written in a list. Dependend on the
    'train_type' the components will be seperated or summed up.
    
    Parameters
    ----------
    grid: 4darray
        input grid of shape [n, 1, img_size, img_size]
    num_comps: list
        list of two number: min number of components and max number of components 
    train_type: str
        determines the purpose of the simulations. Can be 'gauss', 'list' or 'clean'

    Returns
    -------
    jets: ndarray
        image of the full jet, sum over all components, shape: [n, 1, img_size, img_size]
    jet_comps: ndarray
        images of each component and background, shape: [n, c*2, img_size, img_size]
        with c being the max number of components. A jet without counter jet has c 
        components. A jet with counter jet has c*2-1 components, since the center
        appears only once. Adding one channel for the backgound gives c*2 channels.
    source_lists: ndarray
        array which stores all (six) properties of each component, shape: [n, c*2-1, 6]
    """
    if len(grid.shape) == 3:
        grid = grid[None]

    img_size = grid.shape[-1]
    jets = []
    jet_comps = []
    source_lists = []
    for img in grid:
        center = img_size // 2
        comps = np.random.randint(num_comps[0], num_comps[1] + 1)

        amp = np.zeros(num_comps[1])
        x = np.zeros(num_comps[1])
        y = np.zeros(num_comps[1])
        sx = np.zeros(num_comps[1])
        sy = np.zeros(num_comps[1])
        rotation = np.zeros(num_comps[1])

        jet_angle = np.random.uniform(0, 2 * np.pi)

        for i in range(comps):
            # amplitude decreases for more distant components, empirical
            amp[i] = np.exp(-np.sqrt(i) * np.random.normal(1.3, 0.4))

            # curving the jet, empirical
            jet_angle += np.random.normal(0, np.pi / 18)

            # distance between components, r_factor to fill the corners, empirical
            jet_angle_cos = np.abs(np.cos(jet_angle))
            jet_angle_sin = np.abs(np.sin(jet_angle))
            if jet_angle_cos < jet_angle_sin:
                r_factor = 1 / jet_angle_sin
            elif jet_angle_cos > jet_angle_sin:
                r_factor = 1 / jet_angle_cos
            else:
                r_factor = np.sqrt(2)
            #print(i, comps)
            r = i / (comps - 1) * img_size / 2 * r_factor * np.random.uniform(0.8, 0.9)
            #print(jet_angle, r_factor, r)

            # get the cartesian coordinates
            x[i], y[i] = np.array(pol2cart(r, jet_angle)) + center

            # width of gaussian, empirical
            sx[i], sy[i] = r_factor * np.sqrt(i + 1) * np.random.uniform(
                img_size / (7 * comps),
                img_size / (5 * comps),
                size=2,
                )

            # rotation, random or align with the jet angle, empirical
            rotation[i] = np.random.uniform(0, np.pi)
            # rotation[i] = jet_angle + np.random.normal(0, 20)

        # mirror the data for the counter jet
        amp = np.append(amp, amp[1:] * get_exp())
        x = np.append(x, img_size - x[1:])
        y = np.append(y, img_size - y[1:])
        sx = np.append(sx, sx[1:])
        sy = np.append(sy, sy[1:])
        rotation = np.append(rotation, rotation[1:])

        # creation of the image
        jet_img = img[0]
        jet_comp = []
        for i in range(2 * num_comps[1] - 1):
            if amp[i] == 0:
                jet_comp += [np.zeros((img_size, img_size))]
            else:
                g = twodgaussian(
                    [amp[i], x[i], y[i], sx[i], sy[i], rotation[i]],
                    img_size,
                )
                jet_comp += [g]
                jet_img += g

        # normalisation
        jet_max = jet_img.max()
        jet_img /= jet_max
        jet_comp /= jet_max
        amp /= jet_max

        # sum over the 'symmetric' components
        for i in range(num_comps[1] - 1):
            jet_comp[i+1] += jet_comp[num_comps[1]]
            jet_comp = np.delete(jet_comp, num_comps[1], axis=0)
        
        # '1 - normalised' gives the background strength
        jet_comp = np.concatenate((jet_comp, (1 - jet_img)[None, :, :]))
        jets.append(jet_img)
        if train_type == 'clean':
            jet_sum = np.sum(jet_comp[0:-1], axis=0, keepdims=True)
            jet_comps.append(np.concatenate((jet_sum, jet_comp[-1:None]), axis=0))
        else:
            jet_comps.append(jet_comp)
        
        if train_type == 'list':
            # scale the parameters between 0 and 1
            x /= img_size
            y /= img_size
            sx /= np.sqrt(2 * (num_comps[0])) * img_size / (5 * num_comps[0])
            sy /= np.sqrt(2 * (num_comps[0])) * img_size / (5 * num_comps[0])
            rotation /= 2 * np.pi
        
        source_list = np.array(
            [
                amp,
                x,
                y,
                sx,
                sy,
                rotation,
            ]
        ).T
        
        if train_type == 'list':
            source_list = source_list[source_list[:, 0].argsort()]

        source_lists.append(source_list)
    return (
        np.array(jets)[:, None, :, :],
        np.array(jet_comps),
        np.array(source_lists),
    )
