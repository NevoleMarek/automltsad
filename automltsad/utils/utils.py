import logging

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_LOGGER = logging.getLogger(__name__)


def sliding_window_sequences(
    data: np.ndarray,
    window_size: int,
):
    n_samples, n_timepoints, n_features = data.shape
    output = sliding_window_view(data, window_shape=window_size, axis=1)
    output = output.reshape(-1, window_size * n_features)
    _LOGGER.info(f'Rolling window op: Shape of output {output.shape}')
    return output
