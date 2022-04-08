import numpy as np
from radiosim.utils import get_exp, pol2cart
from radiosim.gauss import twodgaussian


def create_jet(image, num_comps, train_type):
    if len(image.shape) == 3:
        image = image[None]

    img_size = image.shape[-1]
    jets = []
    jet_comps = []
    source_lists = []

    for img in image:
        center = img_size // 2
        comps = np.random.randint(num_comps[0], num_comps[1] + 1)

        height = np.zeros(num_comps[1])
        amp = np.zeros(num_comps[1])
        x = np.zeros(num_comps[1])
        y = np.zeros(num_comps[1])
        sx = np.zeros(num_comps[1])
        sy = np.zeros(num_comps[1])
        rotation = np.zeros(num_comps[1])

        jet_angle = np.random.uniform(0, 360)

        for i in range(comps):
            # without background height for now, empirical
            # height[i] = 0

            # amplitude decreases for more distant components, empirical
            amp[i] = np.exp(-np.sqrt(i) * np.random.normal(1.3, 0.4))

            # distance between components, r_factor to fill the corners, empirical
            r_factor =  np.abs(np.sin(jet_angle)) + np.abs(np.cos(jet_angle))
            r = r_factor * i * np.random.uniform(
                center / comps * 0.8,
                center / comps * 0.9,
                )

            # curving the jet, empirical
            jet_angle += np.random.normal(0, 5)
            
            # get the cartesian coordinates
            x[i], y[i] = np.array(pol2cart(r, jet_angle)) + center

            # width of gaussian, empirical
            sx[i], sy[i] = r_factor * np.sqrt(i + 1) * np.random.uniform(
                img_size / (8 * comps),
                img_size / (6 * comps),
                size=2,
                )

            # rotation, random or align with the jet angle, empirical
            rotation[i] = np.random.uniform(0, 360)
            # rotation[i] = jet_angle + np.random.normal(0, 20)

        # mirror the data for the counter jet
        height = np.append(height, height[1:])
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
                    [height[i], amp[i], x[i], y[i], sx[i], sy[i], rotation[i]],
                    (img_size, img_size)
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
            # height /= height.max()
            x /= img_size
            y /= img_size
            sx /= np.sqrt(2 * (num_comps[0])) * img_size / (6 * num_comps[0])
            sy /= np.sqrt(2 * (num_comps[0])) * img_size / (6 * num_comps[0])
            rotation /= 360
        
        source_list = np.array(
            [
                height,
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
