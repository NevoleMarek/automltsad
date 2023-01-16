import numpy as np


def walk_forward_validation(
    data: np.ndarray,
    window_size: int,
    target_size: int = 1,
    keep_history: bool = False,
    axis: int = 1,
):
    """
    Perform walk-forward validation

    Parameters
    ----------
    data : np.ndarray
        3D input array of shape (n_datasets, n_timepoints, n_features)
    window_size : int
        Number of timepoints used for training in each iteration.
    target_size : int, optional
        Number of timepoints used for testing in each iteration, by default 1.
    keep_history : bool, optional
        If True, the indices of the previously used training examples are also included in the training set, by default False.
    axis : int, optional
        Axis along which to perform the walk-forward validation, by default 1.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of 2 1D arrays, where the first array contains the indices of the training examples and the second array contains the indices of the test examples.
    """
    n_timepoints = data.shape[axis]
    indices = np.arange(n_timepoints)
    test_starts = range(window_size, n_timepoints, target_size)
    for test_start in test_starts:
        train_end = test_start
        test_end = test_start + target_size
        train_start = train_end - window_size
        if keep_history:
            yield (indices[:train_end], indices[test_start:test_end])
        else:
            yield (
                indices[train_start:train_end],
                indices[test_start:test_end],
            )
