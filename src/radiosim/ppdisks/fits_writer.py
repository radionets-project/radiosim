# import numpy as np
#
# from astropy.io import fits

from radiosim.ppdisks.simulation import DiskModel


class FITSWriter:
    def __init__(self, model: DiskModel):
        self._model: DiskModel = model
