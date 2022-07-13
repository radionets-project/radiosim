import numpy as np
from radiosim.utils import relativistic_boosting, pol2cart
from radiosim.gauss import twodgaussian


def create_jet(grid, num_comps, train_type):
    """
    Creates the clean jets with all its components written in a list. Dependend on the
    'train_type' the components will be seperated or summed up.

    Parameters
    ----------
    grid: ndarray
        input grid of shape [n, 1, img_size, img_size] or [1, img_size, img_size]
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
    center = img_size // 2
    jets = []
    targets = []
    for img in grid:
        comps = np.random.randint(num_comps[0], num_comps[1] + 1)

        amp = np.zeros(num_comps[1])
        x = np.zeros(num_comps[1])
        y = np.zeros(num_comps[1])
        sx = np.zeros(num_comps[1])
        sy = np.zeros(num_comps[1])
        rotation = np.zeros(num_comps[1])

        # velocity in units of c_0, initialise velocity of first component
        beta = np.zeros(num_comps[1]) 
        beta[1] = np.random.uniform(0, 1)
        y_rotation = np.random.uniform(0, np.pi)
        z_rotation = np.random.uniform(0, np.pi / 2)

        for i in range(comps):
            # amplitude decreases for more distant components, empirical
            amp[i] = np.exp(-np.sqrt(i) * np.random.normal(1.3, 0.2))

            # velocity decreases for more distant components, empirical
            if i >= 2:
                beta[i] = beta[1] * np.exp(-np.sqrt(i - 1) * np.random.normal(0.5, 0.1))

            # curving the jet, empirical
            y_rotation += np.random.normal(0, np.pi / 24)

            # distance between components, r_factor to fill the corners
            jet_angle_cos = np.abs(np.cos(y_rotation))
            jet_angle_sin = np.abs(np.sin(y_rotation))
            if jet_angle_cos < jet_angle_sin:
                r_factor = 1 / jet_angle_sin
            elif jet_angle_cos > jet_angle_sin:
                r_factor = 1 / jet_angle_cos
            else:
                r_factor = np.sqrt(2)

            # *0.8 so the component center is not on the edge
            r = i / (comps - 1) * img_size / 2 * r_factor * np.sin(z_rotation) * 0.8

            # get the cartesian coordinates
            x[i], y[i] = np.array(pol2cart(r, y_rotation)) + center

            # width of gaussian, empirical
            sx[i], sy[i] = (
                r_factor
                * np.sqrt(i + 1)
                * np.random.uniform(
                    img_size / (8 * comps), img_size / (6 * comps), size=2,
                )
            )

            # rotation aligned with the jet angle, empirical
            rotation[i] = y_rotation + np.random.normal(0, np.pi / 18)

        # print('Velocity of the jet:', beta)
        boost_app, boost_rec = relativistic_boosting(z_rotation, beta)

        # mirror the data for the counter jet
        amp = np.concatenate((amp * boost_app, amp[1:] * boost_rec[1:]))
        x = np.append(x, img_size - x[1:])
        y = np.append(y, img_size - y[1:])
        sx = np.append(sx, sx[1:])
        sy = np.append(sy, sy[1:])
        rotation = np.append(rotation, rotation[1:])
        z_rotation = np.repeat(z_rotation, 2 * num_comps[1] - 1)
        beta = np.append(beta, beta[1:])

        # creation of the image
        jet_comp = component_from_list(img_size, amp, x, y, sx, sy, rotation)
        jet_img = np.sum(jet_comp, axis=0)

        # normalisation
        jet_max = jet_img.max()
        jet_img /= jet_max
        jet_comp /= jet_max
        amp /= jet_max

        jets.append(jet_img)

        jet_comp = np.concatenate((jet_comp, (1 - jet_img)[None, :, :]))
        source_list = np.array([amp, x, y, sx, sy, rotation, z_rotation, beta]).T

        target = apply_train_type(train_type, jet_img, jet_comp, source_list)

        targets.append(target)

    jets = np.array(jets)[:, None, :, :]
    targets = np.array(targets)

    return jets, targets


def apply_train_type(train_type, jet_img, jet_comp, source_list):
    """
    Creating the y-data dependent on the training type.

    Parameters
    ----------
    train_type: str
        'list': returns the components attributes only
        'gauss': returns all components, background and list
        'clean': returns sum of components and background (usage for softmax)
    jet_comps: ndarray
        simulated jet components as an image
    source_list:
        attributes of jet components

    Returns
    -------
    y: ndarray
        output data
    """
    if train_type == 'list':
        y = source_list
    if train_type == 'gauss':
        size = jet_comp.shape[-1]
        list_to_add = np.empty((1, size, size))
        list_to_add[:] = np.nan
        list_to_add[:, 0:source_list.shape[0], 0:source_list.shape[1]] = source_list
        y = np.concatenate((jet_comp, list_to_add))
    if train_type == 'clean':
        y = np.concatenate((jet_img[None], jet_comp[-1:None]))
    return y


def component_from_list(size, amp, x, y, sx, sy, rotation):
    """
    Creating jet components from a list of attributes.

    Parameters
    ----------
    size: int
        shape of the output image will be (size, size)
    attributes: list or array
        [amp, x, y, sx, sy, rotation]

    Returns
    -------
    jet_comp: list of ndarray
        all jet components stored in one list
    """
    jet_comp = []
    for i in range(len(amp)):
        if amp[i] == 0:
            jet_comp += [np.zeros((size, size))]
        else:
            g = twodgaussian(
                [amp[i], x[i], y[i], sx[i], sy[i], rotation[i]], size,
            )
            jet_comp += [g]
    return jet_comp