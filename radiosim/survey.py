import numpy as np
from radiosim.gauss import twodgaussian
from radiosim.jet import create_jet


def create_survey(grid, num_sources, class_distribution, scale_sources=False):
    """
    Creates a clean survey with all its components written in a list. It contains
    serveral classes:
    - jets
    - gaussian
    - pointsources

    Parameters
    ----------
    grid: 4darray
        input grid of shape [n, 3, img_size, img_size]
    num_sources: int
        number of total sources in the image (a jet counts as one source)
    class_distribution: list
        list determining the proportion between the classes, at least one element must
        be > 0, e.g. [1, 0, 3.1415]
    scale_sources: boolean
        scaling the sources with the image size, default size is 1024

    Returns
    -------
    survey: ndarray
        image of the survey, sum over all sources, shape: [n, 1, img_size, img_size]
    class_survey: ndarray
        images of each class and background class, shape: [n, c + 1, img_size, img_size]
        with c being the max number of classes.
    source_lists: ndarray
        array which stores all properties of each component, shape: [n, 1, 6]
        for each image the class label and its properties.
    """
    if len(grid.shape) == 3:
        grid = grid[None]

    img_size = grid.shape[-1]
    survey = np.zeros((grid.shape[0], 1, img_size, img_size))
    survey_comps = np.zeros(
        (grid.shape[0], len(class_distribution) + 1, img_size, img_size)
    )
    source_list = []
    for i_img, img in enumerate(grid):
        for i_source in range(num_sources):
            rand_class = np.random.uniform(0, sum(class_distribution))
            # create first class (jet)
            if rand_class < class_distribution[0]:
                if scale_sources:
                    jet_size = np.random.randint(img_size / 10.24, img_size / 5.12)
                else:
                    jet_size = np.random.randint(100, 200)
                num_comps = [4, 7]
                x, y = np.random.rand(2) * img_size
                jet, _, jet_list = create_jet(
                    img[:, 0:jet_size, 0:jet_size], num_comps, "gauss"
                )
                posx_min = int(np.floor(x - jet_size / 2))
                posx_max = int(np.floor(x + jet_size / 2))
                posy_min = int(np.floor(y - jet_size / 2))
                posy_max = int(np.floor(y + jet_size / 2))
                jet = np.squeeze(jet)
                if posx_min < 0:
                    jet = jet[-posx_min:jet_size, :]
                    posx_min = 0
                if posy_min < 0:
                    jet = jet[:, -posy_min:jet_size]
                    posy_min = 0
                if posx_max > img_size:
                    jet = jet[0 : img_size - posx_max + jet_size, :]
                if posy_max > img_size:
                    jet = jet[:, 0 : img_size - posy_max + jet_size]
                survey_comps[i_img, 0, posx_min:posx_max, posy_min:posy_max] += jet
                source_list.append(jet_list)

            # create second class (gaussian)
            elif rand_class < class_distribution[0] + class_distribution[1]:
                if scale_sources:
                    sx = img_size * np.random.uniform(1 / 40.96, 1 / 20.48)
                    sy = sx + np.random.normal(scale=img_size / 204.8)
                else:
                    sx = np.random.uniform(25, 50)
                    sy = sx + np.random.normal(scale=5)
                x, y = np.random.rand(2) * img_size
                amp = np.random.rand()
                rot = np.random.uniform(0, np.pi)
                gauss = twodgaussian([amp, x, y, sx, sy, rot], img_size)
                survey_comps[i_img, 1] += gauss
                source_list.append([amp, x, y, sx, sy, rot])

            # create third class (pointsources)
            else:
                if scale_sources:
                    sx = img_size * np.random.uniform(1 / 204.8, 1 / 102.4)
                    sy = sx + np.random.normal(scale=img_size / 1024)
                else:
                    sx = np.random.uniform(5, 10)
                    sy = sx + np.random.normal(scale=1)
                x, y = np.random.rand(2) * img_size
                amp = np.random.rand()
                rot = np.random.uniform(0, np.pi)
                gauss = twodgaussian([amp, x, y, sx, sy, rot], img_size)
                survey_comps[i_img, 2] += gauss
                source_list.append([amp, x, y, sx, sy, rot])

        survey[i_img] = np.sum(survey_comps[i_img], axis=0)

        # normalisation
        survey_max = survey[i_img].max()
        survey[i_img] /= survey_max
        survey_comps[i_img] /= survey_max
        for source in source_list:
            if isinstance(source, list):
                source[0] /= survey_max
            if isinstance(source, np.ndarray):
                source[0, :, 0] /= survey_max

        # '1 - normalised' gives the background strength
        survey_comps[i_img, -1] = 1 - survey[i_img]

    # set source list to 0, because the saving of the list with variable lengths is
    # not supported
    source_list = 0

    return (
        survey,
        survey_comps,
        source_list,
    )
