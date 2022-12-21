import numpy as np
from sklearn.base import TransformerMixin
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from automltsad.transform import BaseTransformer


class MeanVarianceScaler(TransformerMixin, BaseTransformer):
    '''
    MeanVarianceScaler
    Wrapper for tslearn.preprocessing.TimeSeriesScalerMeanVariance to work
    with automltsad data format.
    Scale according to (X - mean) / std formula.

    Parameters
    ----------
    mean: float
        Mean
    std: float
        Standard deviation
    '''

    def __init__(self, mean: float = 0, std: float = 1) -> None:
        self._mean = mean
        self._std = std
        self.scaler = TimeSeriesScalerMeanVariance(mu=mean, std=std)

    def fit(self, X: np.ndarray):
        '''
        Dummy fit method to comply to sklearn API.

        Parameters
        ----------
        X : np.ndarray
            Ignored

        Returns
        -------
        self

        '''
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray):
        '''
        Scale data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints) for subsequences or
            (n_samples, n_timepoints, n_features) for entire timeseries.
            Data to fit the transformer to.

        Returns
        -------
        np.ndarray of shape (n_samples, n_timepoints) for subsequences or
            (n_samples, n_timepoints, n_features) for entire timeseries.
            Scaled data
        '''
        n_dim = len(X.shape)
        if n_dim == 3:  # Entire time series
            return self.scaler.transform(X)
        elif n_dim == 2:  # Dataset of time series subsequences
            return np.squeeze(self.scaler.transform(X))
        else:
            raise ValueError(f'X has wrong format of {X.shape}')
