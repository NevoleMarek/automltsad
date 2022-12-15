import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)


class Dataset:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        _LOGGER.info('Created dataset')
        _LOGGER.info(f'Shape of data {self.data.shape}')
