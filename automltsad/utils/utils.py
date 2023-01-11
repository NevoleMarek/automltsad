import logging
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_LOGGER = logging.getLogger(__name__)


def sliding_window_sequences(
    data: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Generate sliding window sequences of a 3D numpy array.

    Parameters
    ----------
    data : np.ndarray
        3D input array of shape (n_samples, n_timepoints, n_features)
    window_size : int
        Size of the sliding window

    Returns
    -------
    np.ndarray
        2D array with shape (n_samples * (n_timepoints - window_size + 1), window_size * n_features)
    """
    n_samples, n_timepoints, n_features = data.shape
    output = sliding_window_view(data, window_shape=window_size, axis=1)
    output = output.reshape(-1, window_size * n_features)
    _LOGGER.info(f'Rolling window op: Shape of output {output.shape}')
    return output


def sliding_target_window_sequences(
    data: np.ndarray,
    predictor_size: int,
    target_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictors and targets from sliding window sequences of a 3D numpy array.

    Parameters
    ----------
    data : np.ndarray
        3D input array of shape (n_samples, n_timepoints, n_features)
    predictor_size : int
        Size of the predictor window
    target_size : int
        Size of the target window

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of 2D arrays,
        first of shape (n_samples * (n_timepoints - (predictor_size + target_size) + 1), predictor_size * n_features)
        and the second with shape (n_samples * (n_timepoints - (predictor_size + target_size) + 1), target_size * n_features)
    """
    windows = sliding_window_sequences(data, predictor_size + target_size)
    predictors = windows[:, :predictor_size]
    targets = windows[:, predictor_size:]
    return predictors, targets


def reduce_window_scores(scores: np.ndarray, window_size: int) -> np.ndarray:
    """Reduce scores array using a rolling window.

    Parameters
    ----------
    scores : np.ndarray
        1D input array of scores.
    window_size : int
        Size of the rolling window.

    Returns
    -------
    np.ndarray
        1D array of mean of scores with shape (len(scores) - window_size + 1)
    """
    unwindowed_length = (window_size - 1) + len(scores)
    unwindowed_scores = np.full(
        shape=(unwindowed_length, window_size), fill_value=np.nan
    )
    unwindowed_scores[: len(scores), 0] = scores

    for w in range(1, window_size):
        unwindowed_scores[:, w] = np.roll(unwindowed_scores[:, 0], w)

    return np.nanmean(unwindowed_scores, axis=1)
