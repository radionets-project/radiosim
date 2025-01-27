from scipy import stats


def get_start_amp(scale_type):
    if scale_type == "mojave":
        amp_start = draw_from_mojave_dist()
    return amp_start / 1e3


def draw_from_mojave_dist():
    """Values from a fit to the peak fluxes distribution
    in the MOJAVE data archive.
    """
    a = 0.8639672251677816
    b = 47.64189171625089
    loc = 0.09163404776954732
    scale = 1892.3881692069087
    return stats.beta(a, b, loc, scale).rvs(1)
