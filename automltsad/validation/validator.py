import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)


def validate_data_3d(X: np.ndarray):
    _LOGGER.info('Validating_data_format')
    n_dims = len(X.shape)
    if n_dims != 3:
        raise Exception(
            'Data should be in 3D numpy format (n_samples, n_timepoints, n_features)'
        )


def validate_data_2d(X: np.ndarray):
    _LOGGER.info('Validating_data_format')
    n_dims = len(X.shape)
    if n_dims != 2:
        raise Exception(
            'Data should be in 2D numpy format (n_samples, n_timepoints)'
        )


def check_if(obj, attr, message):
    if hasattr(obj, attr):
        if not getattr(obj, attr):
            raise Exception(message)
    else:
        raise AttributeError(
            f'Object {obj.__class__.__name__} has no attribute called {attr}'
        )


def check_if_fitted(obj):
    check_if(obj, 'fitted', f'Fit the object {obj.__class__.__name__}')


def check_one_sample(X: np.ndarray):
    _LOGGER.info('check_one_sample')
    n_samples = X.shape[0]
    if n_samples != 1:
        raise Exception('More than 1 sample in data')
