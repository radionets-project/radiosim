import numpy as np
from numpy.typing import ArrayLike
from radiosim.gauss import skewed_gauss, twodgaussian
from radiosim.utils import _gen_date, _gen_vlba_obs_position
from scipy.stats import skewnorm, expon
from skimage.transform import swirl, rotate
from joblib import Parallel, delayed


def create_mojave(conf, rng):
    """
    Create MOJAVE like sources.

    Parameters:
    -----------
    conf :
        loaded conf file

    Returns:
    --------
    glx : ArrayLike
        generated sources
    """

    threads = conf["threads"]
    if threads == "none":
        threads = 1
    elif not isinstance(threads, int):
        raise ValueError('threads has to be int >0 or "none"')
    elif threads <= 0:
        raise ValueError('threads has to be int >0 or "none"')

    size = conf["img_size"]

    # calculate amount of each class to generate per bundle
    bundle_size = conf["bundle_size"]
    ratio = np.array(conf["class_ratio"])

    # amount = np.array([n_comact, n_one_jet, n_two_jet]])
    amount = np.array((ratio / ratio.sum()) * bundle_size).astype(int)
    while amount.sum() < bundle_size:
        amount[amount.argmax()] += 1

    jets = []
    compact = Parallel(n_jobs=threads)(
        delayed(gen_compact)(rng=child, size=size) for child in rng.spawn(amount[0])
    )
    one_jet = Parallel(n_jobs=threads)(
        delayed(gen_one_jet)(rng=child, size=size) for child in rng.spawn(amount[1])
    )
    two_jet = Parallel(n_jobs=threads)(
        delayed(gen_two_jet)(rng=child, size=size) for child in rng.spawn(amount[2])
    )
    jets = np.array([*compact, *one_jet, *two_jet])

    glx_class = []
    for glx_type, amt in zip([0, 1, 2], amount):
        glx_class.extend([glx_type] * amt)
    glx_class = np.array(glx_class)

    shuffler = np.arange(bundle_size)
    rng.shuffle(shuffler)

    ra, dec = _gen_vlba_obs_position(rng=rng, size=bundle_size)
    obs_dates = _gen_date(rng=rng, start_date="1995-01-01", size=bundle_size)

    data = [jets[shuffler], glx_class[shuffler], ra[shuffler], dec[shuffler], obs_dates]
    data_name = ["galaxies", "galaxy_classes", "RA", "DEC", "date"]

    return data, data_name


def gen_jet(
    size: int, amplitude: float, width: float, length: float, a: float
) -> ArrayLike:
    """
    Generate jet from skewed 2d normal distributiuon.

    Parameters
    ----------
    size : int
        length of the square image
    amplitude : float
        maximal amplitude of the distribution
    width : float
        width of the distribution (perpendicular to the skewed function)
    length : float
        length of the distribution
    a : float
        skewness parameter

    Returns
    -------
    jet : ArrayLike
        jet for source
    """

    jet = skewed_gauss(
        size=size,
        x=size / 2,
        y=size / 2,
        amp=amplitude,
        width=width,
        length=length,
        a=a,
    )

    jet[np.isclose(jet, 0)] = 0

    return jet


def gen_shocks(
    jet: ArrayLike,
    shock_comp: int,
    length: float,
    width: float,
    sx: float,
    rng: "np.Generator",
) -> tuple[ArrayLike, tuple, int]:
    """
    Generate equally spaced shocks in jet.

    Parameters
    ----------
    jet : ArrayLike
        generated jet without shock
    shock_comp : int
        max amount of shocks
    length : float
        length of jet
    width : float
        width of jet
    sx : float
        sx component of main gaussian
    rng : np.Generator
        numpy random generator

    Returns
    -------
    jet : ArrayLike
        jet with shock components
    printout_information : tuple
        information about each component
    skipped : int
        amount of skipped shock components
    """
    ps_orientation = []
    ps_amp = []
    ps_x = []
    ps_y = []
    ps_sx = []
    ps_sy = []
    ps_rot = []
    ps_dist = []
    ps_a = []

    mask = np.copy(jet)
    mask[np.isclose(mask, 0)] = 0
    mask[mask > 0] = 1
    mask = mask.astype(bool)

    size = len(jet)

    s_length = 2 * length - sx
    s_dist = s_length / shock_comp
    s_dist0 = s_dist
    skipped = 0
    for i in range(shock_comp):
        # skip randomly
        if rng.uniform(0, 1) > 2 / 3:
            skipped += 1
            continue

        s_x = size / 2 + s_dist
        s_y = size / 2
        s_amp = jet[int(s_y), int(s_x)] * rng.uniform(0.03, 0.6)

        s_sx = rng.uniform(*np.sort([2, width]))
        s_sy = rng.uniform(*np.sort([s_sx, length / shock_comp]))

        s_rot = 0

        s_a = rng.uniform(5, 9)

        jet += skewed_gauss(size, int(s_x), int(s_y), s_amp, s_sx, s_sy, s_a)

        ps_amp.append(s_amp)
        ps_x.append(int(s_x))
        ps_y.append(int(s_y))
        ps_sx.append(s_sx)
        ps_sy.append(s_sy)
        ps_rot.append(s_rot)
        ps_dist.append(s_dist)
        ps_a.append(s_a)
        ps_orientation.append("r")

        s_dist += s_dist0

    return (
        jet,
        (ps_orientation, ps_amp, ps_x, ps_y, ps_sx, ps_sy, ps_rot, ps_dist, ps_a),
        skipped,
    )


def add_swirl(
    jet: ArrayLike, rng: "np.Generator", first_jet_params: tuple | None = None
) -> tuple[ArrayLike, tuple]:
    """
    Add swirl distortion to the jet.

    Parameters
    ----------
    jet : ArrayLike
        generated jet
    rng : np.Generator
        numpy random generator
    first_jet_params : tuple | None, default: None
        Use when generate two jet sources.
        Tuple of parameters to generate similar swirl for second jet.
        Contains the returned parameters from the first jet.

    Returns
    -------
    swirled_jet : ArrayLike
        swirl distorted input jet
    parameters : tuple
        parameters of the aplied swirl distortion
    """
    size = len(jet)
    if not first_jet_params:
        strength = rng.normal(loc=0, scale=0.2)
        radius = rng.uniform(50, 200)
        center = np.array([size / 2, size / 2])
    else:
        deviation = 1 + rng.normal(loc=0, scale=0.1, size=2)
        strength = first_jet_params[0] * deviation[0]
        radius = first_jet_params[1] * deviation[1]
        center = first_jet_params[2]

    swirled_jet = swirl(jet, center=center, strength=strength, radius=radius)

    return swirled_jet, (strength, radius, center)


def gen_two_jet(rng: int, size: int = 1024, printout: bool = False) -> ArrayLike:
    """
    Generate a two jet galaxy.

    Parameters
    ----------
    rng : np.Generator | int
    numpy random generator or seed to generate random number generator
    size : int, default: 1024
        size of the galaxy to be genrated
    printout : bool, default: False
        print out log information

    Returns
    -------
    glx : ArrayLike
        simulated two jet source
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    amp_params = (1.715e-4, 1.985e-2)
    width_params = (4.37, 22.71, 16.85)
    length_params = (8.62, 69.60, 54.73)
    left_deviation = 1 + rng.normal(
        loc=0, scale=0.1, size=5
    )  # amp, length, width, strength, radius

    amp = expon.rvs(*amp_params, random_state=rng)

    r_width = skewnorm.rvs(*width_params, random_state=rng) / 10  # 10
    r_length = skewnorm.rvs(*length_params, random_state=rng) / 4  # 10
    r_jet_amp = amp * rng.power(0.7)

    l_width = r_width * left_deviation[2]
    l_length = r_length * left_deviation[1]
    l_jet_amp = r_jet_amp * left_deviation[0]

    dimensions = np.sort([r_width / 2, r_length / 5, l_width / 2, l_length / 5])

    sx = rng.uniform(dimensions.min(), dimensions.max()) * 1.5  # too large
    sy = rng.uniform(dimensions.min(), dimensions.max()) * 1.5

    # redraw if gaussian is too elliptical
    while np.any([sx / sy < 1 / 2, sx / sy > 2]):
        sy = rng.uniform(dimensions.min(), dimensions.max()) * 1.5
    rot = rng.uniform(0, 2 * np.pi)

    a = 4

    r_jet = gen_jet(size, r_jet_amp, r_width, r_length, a)
    l_jet = gen_jet(size, l_jet_amp, l_width, l_length, a)

    shock_comp_r = int(np.round(rng.power(0.5), decimals=1) * 10)
    shock_comp_l = int(np.round(rng.power(0.5), decimals=1) * 10)

    if shock_comp_r > 0:
        r_jet, r_printout, r_skipped = gen_shocks(
            r_jet, shock_comp_r, r_length, r_width, sx, rng
        )

    if shock_comp_l > 0:
        l_jet, l_printout, l_skipped = gen_shocks(
            l_jet, shock_comp_l, l_length, l_width, sx, rng
        )

    r_jet, r_swirl_comp = add_swirl(r_jet, rng=rng)
    l_jet, l_swirl_comp = add_swirl(l_jet, rng=rng, first_jet_params=r_swirl_comp)

    glx = r_jet + np.flip(l_jet)
    x = size / 2
    y = size / 2

    gauss = twodgaussian([amp, size / 2, size / 2, sx, sy, rot], size)
    glx += gauss

    angle = rng.uniform(0, 360)

    glx = rotate(glx, angle=angle)

    glx[np.isclose(glx, 0)] = 0

    if printout:
        print("left Jet:")
        print(
            f"width = {l_width:.3f}, length = {l_length:.3f}, \
                jet_amp = {l_jet_amp:.3f}, a = {a:.3f}"
        )
        print("right Jet:")
        print(
            f"width = {r_width:.3f}, length = {r_length:.3f}, \
                jet_amp = {r_jet_amp:.3f}, a = {a:.3f}"
        )
        print("Source:")
        print(
            f"amp = {amp:.3f}, x = {x:.0f}, y = {y:.0f}, sx = {sx:.3f}, \
                sy = {sy:.3f}, rot = {rot:.3f}"
        )
        if shock_comp_r > 0:
            print(
                f"Right Shock, {shock_comp_r} Components, \
                    {r_skipped} skipped:"
            )
            for orient, amp, x, y, sx, sy, rot, dist, a in zip(*r_printout):
                print(
                    f"amp = {amp:.6f}, x = {x:.0f}, y = {y:.0f}, \
                        sx = {sx:.3f}, sy = {sy:.3f}, rot = {rot:.3f}, \
                        dist = {dist:.3f}"
                )
        if shock_comp_l > 0:
            print(
                f"Left Shock, {shock_comp_l} Components, \
                    {l_skipped} skipped:"
            )
            for orient, amp, x, y, sx, sy, rot, dist, a in zip(*l_printout):
                print(
                    f"amp = {amp:.6f}, x = {x:.0f}, y = {y:.0f}, \
                        sx = {sx:.3f}, sy = {sy:.3f}, rot = {rot:.3f}, \
                        dist = {dist:.3f}"
                )
        print("Swirl:")
        print(
            f"Right: strength = {r_swirl_comp[0]:.3f}, \
                radius = {r_swirl_comp[1]:.3f}, center = {r_swirl_comp[2]}"
        )
        print(
            f"Left: strength = {l_swirl_comp[0]:.3f}, \
                radius = {l_swirl_comp[1]:.3f}, center = {l_swirl_comp[2]}"
        )

    return glx


def gen_one_jet(rng: int, size: int = 1024, printout: bool = False) -> ArrayLike:
    """
    Generate a one jet galaxy.

    Parameters
    ----------
    rng : np.Generator | int
        numpy generator or seed to generate random number generator
    size : int, default: 1024
        size of the galaxy to be genrated
    printout : bool, default: False
        print out debug information

    Returns
    -------
    glx : ndarray
        simulated one jet source
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    amp_params = (1.715e-4, 1.985e-2)
    width_params = (4.37, 22.71, 16.85)
    length_params = (8.62, 69.60, 54.73)

    amp = expon.rvs(*amp_params, random_state=rng)

    width = skewnorm.rvs(*width_params, random_state=rng) / 12  # 10
    length = skewnorm.rvs(*length_params, random_state=rng) / 6.5  # 10
    jet_amp = amp * rng.power(0.7)

    dimensions = np.sort([width / 2, length / 5])

    sx = rng.uniform(*dimensions) * 1.5  # too large
    sy = rng.uniform(*dimensions) * 1.5

    # redraw if gaussian is too elliptical
    while np.any([sx / sy < 1 / 2, sx / sy > 2]):
        sy = rng.uniform(dimensions.min(), dimensions.max()) * 1.5
    rot = rng.uniform(0, 2 * np.pi)

    a = 4

    jet = gen_jet(size, jet_amp, width, length, a)

    shock_comp = int(np.round(rng.power(0.5), decimals=1) * 10)

    if shock_comp > 0:
        jet, shock_printout, skipped = gen_shocks(
            jet, shock_comp, length, width, sx, rng
        )

    jet, swirl_comp = add_swirl(jet, rng=rng)

    x = size / 2
    y = size / 2

    gauss = twodgaussian([amp, size / 2, size / 2, sx, sy, rot], size)
    glx = jet + gauss

    angle = rng.uniform(0, 360)

    glx = rotate(glx, angle=angle)

    glx[np.isclose(glx, 0)] = 0

    if printout:
        print("Skewed dist:")
        print(
            f"width = {width:.3f}, length = {length:.3f}, \
                jet_amp = {jet_amp:.3f}, a = {a:.3f}"
        )
        print("Source:")
        print(
            f"amp = {amp:.3f}, x = {x:.0f}, y = {y:.0f}, sx = {sx:.3f}, \
                sy = {sy:.3f}, rot = {rot:.3f}"
        )
        if shock_comp > 0:
            print(f"{shock_comp} shock components, {skipped} skipped:")
            for orient, amp, x, y, sx, sy, rot, dist, a in zip(shock_printout):
                print(
                    f"amp = {amp:.6f}, x = {x:.0f}, y = {y:.0f}, \
                        sx = {sx:.3f}, sy = {sy:.3f}, rot = {rot:.3f}, \
                        dist = {dist:.3f}"
                )
        print("Swirl:")
        print(
            f"strength = {swirl_comp[0]:.3f}, \
                radius = {swirl_comp[1]:.3f}, center = {swirl_comp[2]}"
        )

    return glx


def gen_compact(rng: int, size: int = 1024, printout: bool = False) -> ArrayLike:
    """Generate a compact galaxy.

    Parameters
    ----------
    rng : np.Generator | int
        numpy generator or seed to generate random number generator.
    size : int, default: 1024
        size of the galaxy to be genrated
    printout : bool, default: False
        print out debug information.

    Returns
    -------
    glx : ndarray
        Simulated compact source.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    amp_params = (5.54e-05, 0.02)
    width_params = (8.41, 3.04, 3.46)
    length_params = (6.04, 3.02, 2.59)

    ncomps = rng.integers(1, 4)  # draw amount of components
    outflow = rng.uniform() > 0.5  # draw if outflow of the jet is generated

    amp = expon.rvs(*amp_params, random_state=rng)

    jet_amp = amp * rng.normal(2 / 3, 0.05)

    y = np.arange(size)

    width = skewnorm.rvs(*width_params, random_state=rng) * 0.8
    length = skewnorm.rvs(*length_params, random_state=rng) * 1

    if outflow:
        a = 4
        glx = gen_jet(size, jet_amp, width, length, a)
        glx, swirl_comp = add_swirl(glx, rng=rng)
    else:
        glx = np.zeros((size, size))

    dimensions = np.sort([width / 2, length / 4])
    dimensions[0] *= 1.2
    dimensions = np.sort(dimensions)

    i = 0
    comp_amp = []
    comp_x = []
    comp_y = []
    comp_sx = []
    comp_sy = []
    comp_rot = []
    vertical = True
    while i < ncomps:
        deviation = 1 + rng.normal(loc=0, scale=0.2, size=2)
        if i != 0:
            x = rng.uniform(-5, 5)
            y = rng.uniform(-5, 5)
        else:
            x = 0
            y = 0
        if i == 0:
            sx = rng.uniform(*dimensions) * 1.7  # too large
            j = 0
            while sx > 15:
                if j > 10:
                    sx /= 5
                    break
                sx = rng.uniform(*dimensions) * 1.7
                j += 1

            sy = rng.uniform(*dimensions) * 1.7

            j = 0
            while sy > 15:
                if j > 10:
                    sy /= 5
                    break
                sy = rng.uniform(*dimensions) * 1.7
                j += 1

            j = 0
            while np.any([sx / sy < 0.33, sx / sy > 3]) or np.all(
                [sx / sy > 0.75, sx / sy < 1.33]
            ):
                if j > 10:
                    sy = sx * rng.choice([rng.uniform(0.4, 0.8), rng.uniform(1.2, 2.5)])
                    if sy > 15:
                        sy /= 5
                    break
                sy = rng.uniform(*dimensions) * 1.7
                k = 0
                while sy > 15:
                    if k > 10:
                        sy /= 5
                        break
                    sy = rng.uniform(*dimensions) * 1.7
                    k += 1
                j += 1
        else:
            sx = comp_sx[-1] * deviation[0]
            sy = comp_sy[-1] * deviation[1]

        if i == 0:
            if sx > sy:
                vertical = True
            else:
                vertical = False

        try:
            rot = rng.uniform(comp_rot[-1] - 0.1, comp_rot[-1] + 0.1)
        except IndexError:
            if vertical:
                rot = rng.uniform(-np.pi / 4, np.pi / 4)
            else:
                rot = rng.uniform(-np.pi / 4 + np.pi / 2, np.pi / 4 + np.pi / 2)

        c_amp = amp * rng.uniform(0.5, 1.2)

        gauss = twodgaussian([c_amp, size / 2 - x, size / 2 - y, sx, sy, rot], size)
        glx += gauss

        comp_amp.append(c_amp)
        comp_x.append(size / 2 - x)
        comp_y.append(size / 2 - y)
        comp_sx.append(sx)
        comp_sy.append(sy)
        comp_rot.append(rot)

        i += 1

    angle = rng.uniform(0, 360)
    glx = rotate(glx, angle=angle)

    shift = size // 2 - np.argwhere(glx == glx.max())[0]

    glx = np.roll(glx, shift=shift, axis=[0, 1])

    glx[np.isclose(glx, 0)] = 0

    glx /= glx.max()
    glx *= amp

    if printout:
        if outflow:
            print("Skewed dist:")
            print(
                f"width = {width:.3f}, length = {length:.3f}, jet_amp = {jet_amp:.3f}, a = {a:.3f}"
            )
            print("Swirl:")
            print(
                f"strength = {swirl_comp[0]:.3f}, radius = {swirl_comp[1]:.3f}, center = {swirl_comp[2]}"
            )
        print("Sources:")
        for cx, cy, csx, csy, cr in zip(comp_x, comp_y, comp_sx, comp_sy, comp_rot):
            print(
                f"amp = {amp:.3f}, x = {cx:.0f}, y = {cy:.0f}, sx = {csx:.3f}, sy = {csy:.3f}, rot = {cr:.3f}"
            )

    return glx
