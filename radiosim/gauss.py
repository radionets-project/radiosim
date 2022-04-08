import numpy as np


def twodgaussian(params: list, shape: tuple):
    """
    Returns a 2d gaussian function of the form:
    x' = np.cos(rot) * x - np.sin(rot) * y
    y' = np.sin(rot) * x + np.cos(rot) * y
    (rot should be in degrees)
    g = a * np.exp ( - ( ((x-center_x)/width_x)**2 +
    ((y-center_y)/width_y)**2 ) / 2 )
    
    Parameters
    ----------
    params: list
        [amplitude, center_x, center_y, width_x, width_y, rot]
        (rot (rotation) in degrees)
    shape: tuple
        shape of the returned image
    
    Returns
    -------
    rotgauss: 2darray
        gaussian distribution in two dimensions
    
    Short version of 'twodgaussian':
    https://github.com/keflavich/gaussfitter/blob/0891cd3605ab5ba000c2d5e1300dd21c15eee1fd/gaussfitter/gaussfitter.py#L75
    """
    amplitude = params[0]
    center_y, center_x = params[1], params[2]
    width_x, width_y = params[3], params[4]
    rot = params[5]
    rot = np.pi/180 * rot
    rcen_x = center_x * np.cos(rot) - center_y * np.sin(rot)
    rcen_y = center_x * np.sin(rot) + center_y * np.cos(rot)

    def rotgauss(x, y):
        xp = x * np.cos(rot) - y * np.sin(rot)
        yp = x * np.sin(rot) + y * np.cos(rot)
        g = amplitude*np.exp(-(((rcen_x-xp)/width_x)**2 +
                                      ((rcen_y-yp)/width_y)**2)/2.)
        return g
    
    return rotgauss(*np.indices(shape))