import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)


def validate_data_format(X: np.ndarray):
    _LOGGER.info('Validating data')
    n_dims = len(X.shape)
    if n_dims != 3:
        raise Exception(
            'Data should be in 3D numpy format (n_samples, n_timepoints, n_features)'
        )


def check_if(obj, attr):
    if hasattr(obj, attr):
        if not getattr(obj, attr):
            raise Exception(f'Value of "{attr}" is {getattr(obj, attr)}')
    else:
        raise AttributeError(
            f'Object {obj.__class__.__name__} has no attribute called {attr}'
        )


def check_if_fitted(obj):
    return check_if(obj, '_fitted')
