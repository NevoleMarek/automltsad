import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)


def sliding_window_sequences(
    data: np.ndarray,
    window_size: int,
):
    output = []
    for i in range(data.shape[0] - window_size + 1):
        output.append(data[i : (i + window_size)])

    output = np.stack(output)
    _LOGGER.info(f'Rolling window op: Shape of output {output.shape}')
    return output
