import logging

import numpy as np
import pytorch_lightning as pl
import pywt
import torch
from sklearn.base import BaseEstimator
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset

from automltsad.detectors.deeplearning import GDN, LSTM_AE, VAE, TranAD
from automltsad.detectors.GTA.gta import GTA
from automltsad.transform import MeanVarianceScaler
from automltsad.utils import (
    conv_3d_to_2d,
    reduce_window_scores,
    sliding_window_sequences,
)
from automltsad.validation import (
    check_if_fitted,
    check_one_sample,
    validate_data_2d,
    validate_data_3d,
)

_LOGGER = logging.getLogger(__name__)


class TrivialDetector(BaseEstimator):
    """
    TrivialDetector is a simple model implementation.

    TrivialDetector standardizes data based on mean and std of training data.
    Anomaly score is the number of stds from mean value of training data.
    Threshold is determined as 1-contamination-th quantile of standardized
    data.


    Parameters
    ----------
    contamination : float
        Contamination parameter is used to select threshold.
    """

    def __init__(self, contamination: float) -> None:
        self.name = 'trivial'
        self.fitted = False
        self.contamination = contamination

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the trivial model on training dataset.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self: TrivialDetector
            The fitted trivial model.
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)
        check_one_sample(X)
        self.fitted = True
        self.mean = np.mean(X, axis=1)
        self.std = np.std(X, axis=1)
        self.threshold = np.quantile(
            (X - self.mean) / self.std, 1 - self.contamination
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (1, n_timepoints, n_features)
            Labels of data points
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        validate_data_3d(X)
        check_one_sample(X)
        check_if_fitted(self)

        scores = self.predict_anomaly_scores(X)
        return scores > self.threshold

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            Array of shape (1, n_timepoints, n_features)
            Anomaly scores.
        """
        validate_data_3d(X)
        check_one_sample(X)
        check_if_fitted(self)

        return np.abs((X - self.mean) / self.std)


class WindowingDetector(BaseEstimator):
    """
    WindowingDetector allows for regular outlier/anomaly detection algorithms
    to be used on time series data. Subsequences are extracted from the
    original time series and then are served as vectors to the regular models.

    This class is just a wrapper of a model using windows. It's purpose is
    to allow easily deploy a selected model.

    Parameters
    ----------
    model
        model model to be applied on subsequences.
    window_size: int
        Size of the subsequences.
    standardize: bool
        Whether the subsequences should be standardized or not.
    scaler_kwargs: dictionary
        Dictionary of MeanVarianceScaler parameters
    """

    def __init__(
        self,
        model,
        window_size: int,
        standardize: bool = False,
        **scaler_kwargs,
    ) -> None:
        if window_size < 1:
            raise ValueError('Window size should be > 0')
        self.fitted = False
        self.window_size = window_size
        self.model = model
        self.standardize = standardize
        if self.standardize:
            self.scaler = MeanVarianceScaler(**scaler_kwargs)

    def fit(self, X: np.ndarray, y=None):
        """
        Fit supplied model to transformed data.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Training data.

        Returns
        -------
        self
        """
        validate_data_3d(X)
        check_one_sample(X)
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        self.fitted = True
        X_sequences = self._prep(X)
        if self.standardize:
            X_sequences = self.scaler.fit_transform(X_sequences)

        self.model.fit(X_sequences)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Input data

        Returns
        -------
        np.ndarray
            _description_
        """
        check_if_fitted(self)
        validate_data_3d(X)
        check_one_sample(X)
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        X_sequences = self._prep(X)
        if self.standardize:
            X_sequences = self.scaler.transform(X_sequences)

        labels = self.model.predict(X_sequences)
        return labels

    def predict_anomaly_scores(self, X: np.ndarray):
        check_if_fitted(self)
        validate_data_3d(X)
        check_one_sample(X)
        X_sequences = self._prep(X)
        if self.standardize:
            X_sequences = self.scaler.transform(X_sequences)

        return reduce_window_scores(
            self.model.predict_anomaly_scores(X_sequences),
            self.window_size,
        )

    def _prep(self, X: np.ndarray):
        return sliding_window_sequences(X, window_size=self.window_size)


class KNN(BaseEstimator):
    """
    Nearest neighbor detector.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for kneighbors queries.
    radius : float, default=1.0
        Range of parameter space to use by default for radius_neighbors
        queries.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='ball_tree'
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree'
        - 'kd_tree'
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to fit method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, default=100
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.
    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_
        and the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.
        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
    p : float, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
    """

    def __init__(
        self,
        n_neighbors=5,
        radius=1.0,
        algorithm='ball_tree',
        leaf_size=100,
        metric='minkowski',
        p=2,
        metric_params=None,
        n_jobs=None,
    ) -> None:
        self.name = 'knn'
        self.fitted = False
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the nearest neighbor estimator.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Training data.

        Returns
        -------
        self
            Fitted KNN estimator.
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)
        X_ = conv_3d_to_2d(X)
        self.fitted = True
        self.model.fit(X_)
        dist, _ = self.model.kneighbors(X_, self.n_neighbors + 1)
        self.threshold = np.max(np.mean(dist[:, 1:], axis=1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data.

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, 1)
            Anomaly labels
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        dist = self.predict_anomaly_scores(X)
        return dist > self.threshold

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray, np.ndarray
            np.ndarray of shape (n_samples, 1)
            Distance to nearest data point for each sample
            np.ndarray of shape (n_samples, 1)
            Index of nearest neighbor in training data for each sample from X

        """
        check_if_fitted(self)
        X_ = conv_3d_to_2d(X)
        dist, _ = self.model.kneighbors(X_, self.n_neighbors)
        return np.mean(dist[:, 1:], axis=1)


class IsolationForestAD(BaseEstimator):
    """
    Isolation Forest Algorithm.

    Return the anomaly score of each sample using the IsolationForest algorithm
    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.
    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.
    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    max_samples : "auto", int or float, default="auto"
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores of the samples.
            - If 'auto', the threshold is determined as in the
              original paper.
            - If float, the contamination should be in the range (0, 0.5].
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max(1, int(max_features * n_features_in_))`
            features.
        Note: using a float number less than 1.0 or integer less than number of
        features will enable feature subsampling and leads to a longerr runtime.
    bootstrap : bool, default=False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.
    n_jobs : int, default=None
        The number of jobs to run in parallel
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.
    verbose : int, default=0
        Controls the verbosity of the tree building process.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ) -> None:
        self.name = 'isoforest'
        self.fitted = False
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the isolation forest estimator.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Training data.

        Returns
        -------
        self
            Fitted IF estimator.
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)
        X_ = conv_3d_to_2d(X)
        self.fitted = True
        self.model.fit(X_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, )
            Anomaly labels
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        X_ = conv_3d_to_2d(X)
        return self.model.predict(X_)

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores.

        The higher the score the more anomalous the point.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, )
            Anomaly score
        """
        check_if_fitted(self)
        X_ = conv_3d_to_2d(X)
        return -self.model.score_samples(X_)


class LOF(BaseEstimator):
    """
    Unsupervised Outlier Detection using the Local Outlier Factor (LOF).
    The anomaly score of each sample is called the Local Outlier Factor.
    It measures the local deviation of the density of a given sample with
    respect to its neighbors.
    It is local in that the anomaly score depends on how isolated the object
    is with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of a sample to the local densities of its
    neighbors, one can identify samples that have a substantially lower density
    than their neighbors. These are considered outliers.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='ball_tree'
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree'
        - 'kd_tree'
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to fit method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, default=100
        Leaf is size passed to BallTree or KDTree. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.
    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_
        and the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.
        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
    p : int, default=2
        Parameter for the Minkowski metric from
        :func:`sklearn.metrics.pairwise.pairwise_distances`. When p = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric_params : dict, default=None
        Additional keyword arguments for the metric function.
    contamination : 'auto' or float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the scores of the samples.
        - if 'auto', the threshold is determined as in the
          original paper,
        - if a float, the contamination should be in the range (0, 0.5].
    novelty : bool, default=False
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set; and note that the
        results obtained this way may differ from the standard LOF results.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
    """

    def __init__(
        self,
        n_neighbors=20,
        algorithm="ball_tree",
        leaf_size=100,
        metric="minkowski",
        p=2,
        metric_params=None,
        contamination="auto",
        novelty=True,
        n_jobs=None,
    ):
        self.name = 'lof'
        self.fitted = False
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.contamination = contamination
        self.novelty = novelty
        self.n_jos = n_jobs
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            contamination=contamination,
            novelty=novelty,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the Local outlier factor estimator.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Training data.

        Returns
        -------
        self
            Fitted LOF estimator.
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)
        X_ = conv_3d_to_2d(X)
        self.fitted = True
        self.model.fit(X_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, )
            Anomaly labels
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        X_ = conv_3d_to_2d(X)
        return self.model.predict(X_)

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores.

        The higher the score the more anomalous the point.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, )
            Anomaly scores


        """
        check_if_fitted(self)
        X_ = conv_3d_to_2d(X)
        return -self.model.score_samples(X_)


class DWTMLEAD(BaseEstimator):
    """
    Unsupervised offline method

     “Time Series Anomaly Detection with Discrete Wavelet Transforms and Maximum Likelihood Estimation.” https://www.researchgate.net/publication/330400907_Time_Series_Anomaly_Detection_with_Discrete_Wavelet_Transforms_and_Maximum_Likelihood_Estimation (accessed Dec. 12, 2022).

    Parameters
    ----------
    l: int, default=4
        Starting level of DWT transform. Has to be < ceil(log2(len(data)))
    epsilon: float [0,1], default=0.01
        Used as percentile of probabilities when deciding whether window is anomalous.
    b: int, default=2
        Used as threshold to predict anomaly labels of data points.
    """

    def __init__(
        self,
        l=2,
        epsilon=0.01,
        b=2,
    ):
        self.name = 'dwtmlead'
        self.fitted = False
        self.l = l
        self.epsilon = epsilon
        self.b = b

    def fit(self, X=None, y=None):
        """
        Ignored. Kept for API consistency.

        Parameters
        ----------
        X
            Ignored.
        y
            Ignored.
        Returns
        -------
        self
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, )
            Anomaly labels
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')

        return self.predict_anomaly_scores(X) > self.b

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores.

        The higher the score the more anomalous the point.

        Parameters
        ----------
        X : np.ndarray of shape (1, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, )
            Index of nearest neighbor in training data for each sample from X

        """
        check_if_fitted(self)
        validate_data_3d(X)
        check_one_sample(X)
        return self._detect(X)

    def _pad(self, X: np.ndarray):
        n = X.shape[1]
        exp = int(np.ceil(np.log2(n)))
        m = 2**exp
        return pywt.pad(X, ((0, 0), (0, m - n), (0, 0)), 'symmetric')

    def _detect(self, X: np.ndarray):
        X_pad = self._pad(X)
        A, D, ls = self._get_dwt_coeffs(X_pad)
        ws = [max(2, l - self.l + 1) for l in ls]

        # Get anomalous events
        a_scores = []
        l_scores = []
        for a, w, l in list(zip(A, ws[1:], ls[1:])) + list(zip(D, ws, ls)):
            a_wl = sliding_window_sequences(data=a, window_size=w)
            a_wl = conv_3d_to_2d(a_wl)
            est = EmpiricalCovariance(assume_centered=False).fit(a_wl)
            p = np.zeros((a_wl.shape[0],))
            for i, sample in enumerate(a_wl):
                p[i] = est.score(sample.reshape(1, -1))
            z_eps = np.percentile(p, 100 * self.epsilon)
            a = p < z_eps
            a_scores.append(reduce_window_scores(a, w))
            l_scores.append(l)

        # Count events
        anomaly_score = np.zeros((X_pad.shape[1],))
        for a, l in zip(a_scores, l_scores):
            anomaly_score += np.repeat(
                a, 2 ** int(np.log2(X_pad.shape[1]) - l)
            )
        anomaly_score[anomaly_score < 2] = 0
        return anomaly_score[: X.shape[1]]

    def _get_dwt_coeffs(self, X: np.ndarray):
        A = []
        D = []
        ls = []

        a_l = X
        D.append(X)
        L = int(np.ceil(np.log2(X.shape[1])))
        ls.append(L)
        for l in range(L - self.l):
            a_l, d_l = pywt.dwt(a_l, wavelet='haar', axis=1)
            ls.append(L - l - 1)
            A.append(a_l)
            D.append(d_l)
        return A, D, ls


class RandomForest(BaseEstimator):
    """
    A random forest regressor.
    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.
    Read more in the :ref:`User Guide <forest>`.
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.
        Training using "absolute_error" is significantly slower
        than when using "squared_error".

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : {"sqrt", "log2", None}, int or float, default=1.0
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None or 1.0, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
        Only available if bootstrap=True.
    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.
    random_state : int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`Glossary <warm_start>` and
        :ref:`gradient_boosting_warm_start` for details.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0.0, 1.0]`.

    """

    def __init__(
        self,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    ) -> None:
        self.name = 'randomforest'
        self.fitted = False
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the random forest estimator.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Training data.
        y : np.ndarray of shape (n_samples, n_outputs) or (n_samples,)
            The target values
        Returns
        -------
        self
            Fitted estimator.
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)
        X_ = conv_3d_to_2d(X)
        self.fitted = True
        self.model.fit(X_, y)
        y_pred = self.model.predict(X_)
        self.threshold = np.nanmax(np.abs(y_pred - y))
        return self

    def predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data
        y : np.ndarray of shape (n_samples, n_outputs) or (n_samples,)
            Observed values

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, 1)
            Anomaly labels
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        check_if_fitted(self)
        scores = self.predict_anomaly_scores(X, y)
        return scores > self.threshold

    def predict_anomaly_scores(self, X: np.ndarray, y: np.ndarray):
        """
        Predict anomaly scores.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data
        y : np.ndarray of shape (n_samples, n_outputs) or (n_samples,)
            Observed values

        Returns
        -------
        np.ndarray, np.ndarray
            np.ndarray of shape (n_samples,)
            Absolute difference between predicted and observed values.
        """
        check_if_fitted(self)
        validate_data_3d(X)
        X_ = conv_3d_to_2d(X)
        y_pred = self.model.predict(X_)
        return np.abs(y_pred - y)


class OCSVM(BaseEstimator):
    """
    Unsupervised Outlier Detection.
    Estimate the support of a high-dimensional distribution.
    The implementation is based on libsvm.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
         Specifies the kernel type to be used in the algorithm.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Must be non-negative. Ignored by all other kernels.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features
        - if float, must be non-negative.
    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    nu : float, default=0.5
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.
    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.
    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).
    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    """

    def __init__(
        self,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        nu=0.5,
        shrinking=True,
        cache_size=200,
        verbose=False,
        max_iter=500,
    ) -> None:
        self.name = 'ocsvm'
        self.fitted = False
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.model = OneClassSVM(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            nu=nu,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the OCSVM estimator.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Training data.
        y : ignored
        Returns
        -------
        self
            Fitted estimator.
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)
        self.fitted = True
        X_ = conv_3d_to_2d(X)
        self.model.fit(X_, y)
        return self

    def predict(self, X: np.ndarray):
        """
        Predict anomaly labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray
            np.ndarray of shape (n_samples, )
            Anomaly labels
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        check_if_fitted(self)
        validate_data_3d(X)
        X_ = conv_3d_to_2d(X)
        return self.model.predict(X_) == -1
        # scores = self.predict_anomaly_scores(X, y)
        # return scores > self.threshold

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_timepoints, n_features)
            Data

        Returns
        -------
        np.ndarray, np.ndarray
            np.ndarray of shape (n_samples,)
            Absolute difference between predicted and observed values.
        """
        check_if_fitted(self)
        validate_data_3d(X)
        X_ = conv_3d_to_2d(X)
        return -self.model.score_samples(X_)


class LSTM_AE_Det(BaseEstimator):
    """
    Wrapper class for LSTM encoder decoder network.
    Based on paper:
        LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection
        https://arxiv.org/abs/1607.00148

    Parameters
    ----------
    n_feats : int, default 1
        Number of features in the input for timeseries of a shape
        (n_samples, n_timepoints, n_features)
    hidden_size : int, default 8
        Size of hidden state vector
    n_layers : int, default 1
        Number of stacked LSTM cells
    dropout : tuple[float, float], default (0.1, 0.1)
        Tuple of floats used for encoder and decoder layers
    lr : float, default 1r-3
        Learning rate
    batch_size : int, default 256
        Number of samples in one batch
    trainer_config : dict
        Kwargs for pl.Trainer when fitting the model
    """

    def __init__(
        self,
        n_feats=1,
        hidden_size=8,
        n_layers=1,
        dropout=(0.1, 0.1),
        lr=1e-3,
        batch_size=256,
        trainer_config=None,
        **kwargs,
    ) -> None:
        self.name = 'lstmae'
        self.fitted = False
        self.n_feats = n_feats
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.model = LSTM_AE(
            n_feats=self.n_feats,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
            learning_rate=self.lr,
        )
        self.trainer_config = trainer_config

    def fit(self, X: np.ndarray, y=None):
        """
        Fit method of the network.
        Contains training loop and estimation of covariance for
        anomaly scoring.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_timepoints, n_features)
        y : np.ndarray, optional
            Ignored.

        Returns
        -------
        self
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)

        # Prepare data
        X_tensor = torch.from_numpy(X)
        X_train, X_valid = train_test_split(
            X_tensor, test_size=0.3, shuffle=False
        )

        train_loader = DataLoader(
            X_train.to(torch.float32),
            batch_size=self.batch_size,
            drop_last=True,
        )
        val_loader = DataLoader(
            X_valid.to(torch.float32), batch_size=self.batch_size * 4
        )

        trainer = pl.Trainer(**self.trainer_config)
        trainer.fit(self.model, train_loader, val_loader)

        # Empirical covariance for anomaly scoring
        out = pl.Trainer().predict(self.model, val_loader)
        out = torch.cat(out, dim=0)
        errs = (out - X_valid).abs()
        err_vecs = errs.squeeze().detach().numpy()
        self.est = EmpiricalCovariance(assume_centered=False).fit(err_vecs)
        return self

    def predict(self, X: np.ndarray):
        """
        Predict method
        Not implemented

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Raises
        ------
        NotImplementedError
            Not implemented
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        raise NotImplementedError()

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, )
            Mahanalobis distance to estimated covariance
        """
        validate_data_3d(X)
        X_tensor = torch.from_numpy(X)
        eval_loader = DataLoader(
            X_tensor.to(torch.float32), batch_size=self.batch_size * 8
        )

        trainer = pl.Trainer()
        out = trainer.predict(self.model, eval_loader)
        out = torch.cat(out, dim=0)
        errs = (out - X_tensor).abs()
        errs = errs.squeeze().detach().numpy()
        p = self.est.mahalanobis(errs)
        return p


class VAE_Det(BaseEstimator):
    """
    Wrapper class for Variational autoencoder network.
    Based on paper:
        Auto-Encoding Variational Bayes, Kingma, Welling
        http://arxiv.org/abs/1312.6114

    Parameters
    ----------
    window_size : int
        Size of input
    encoder_hidden : List[int]
        List of integers where each integer is size of hidden layer in encoder
    decoder_hidden : List[int]
        List of integers where each integer is size of hidden layer in decoder
    latent_dim : int
        Size of latent linear layer
    lr : float, default 1r-3
        Learning rate
    batch_size : int, default 256
        Number of samples in one batch
    trainer_config : dict
        Kwargs for pl.Trainer class when fitting the model
    """

    def __init__(
        self,
        window_size,
        encoder_hidden=[128, 64, 32],
        decoder_hidden=[32, 64, 128],
        latent_dim=16,
        lr=1e-3,
        batch_size=256,
        trainer_config=None,
        **kwargs,
    ) -> None:
        self.name = 'vae'
        self.fitted = False
        self.window_size = window_size
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.model = VAE(
            window_size=self.window_size,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
            latent_dim=self.latent_dim,
            learning_rate=self.lr,
        )
        self.trainer_config = trainer_config

    def fit(self, X: np.ndarray, y=None):
        """
        Fit method of the network.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_timepoints, n_features)
        y : np.ndarray, optional
            Ignored.

        Returns
        -------
        self
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)

        n_s, n_t, n_f = X.shape
        X_tensor = torch.from_numpy(X).view(n_s, n_t * n_f)
        X_train, X_valid = train_test_split(
            X_tensor, test_size=0.3, shuffle=False
        )

        train_loader = DataLoader(
            X_train.to(torch.float32),
            batch_size=self.batch_size,
            drop_last=True,
        )
        val_loader = DataLoader(
            X_valid.to(torch.float32), batch_size=X_valid.shape[0]
        )

        trainer = pl.Trainer(**self.trainer_config)
        trainer.fit(self.model, train_loader, val_loader)
        return self

    def predict(self, X: np.ndarray):
        """
        Predict method
        Not implemented

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Raises
        ------
        NotImplementedError
            Not implemented
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        raise NotImplementedError()

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, )
            Reconstruction error as anomaly score
        """
        validate_data_3d(X)
        n_s, n_t, n_f = X.shape
        X_tensor = torch.from_numpy(X).view(n_s, n_t * n_f)
        eval_loader = DataLoader(X_tensor.to(torch.float32), batch_size=n_s)

        trainer = pl.Trainer()
        errs = trainer.predict(self.model, eval_loader)
        return torch.cat(errs, dim=0).squeeze().detach().numpy()


class TranAD_Det(BaseEstimator):
    """
    Wrapper class for TranAD.
    Based on paper:
        TranAD: Deep Transformer Networks for Anomaly Detection in
        Multivariate Time Series Data,
        http://arxiv.org/abs/2201.07284

    Parameters
    ----------
    window_size : int
        Size of input
    n_layers : int
        Number of encoder and decoder layers
    ff_dim : int
        Dimension of feedforward layer in TransformerEncoder and Decoder Layer
    nhead : int
        Number of heads in Transformer
    n_feats : int, default 1
        Number of channels of time series
    lr : float, default 1r-3
        Learning rate
    batch_size : int, default 256
        Number of samples in one batch
    trainer_config : dict
        Kwargs for pl.Trainer class when fitting the model
    """

    def __init__(
        self,
        window_size,
        n_layers,
        ff_dim,
        nhead,
        n_feats=1,
        lr=1e-3,
        batch_size=256,
        trainer_config=None,
        **kwargs,
    ) -> None:
        self.name = 'tranad'
        self.fitted = False
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.nhead = nhead
        self.n_feats = n_feats
        self.window_size = window_size
        self.lr = lr
        self.batch_size = batch_size
        self.model = TranAD(
            n_feats=self.n_feats,
            window_size=self.window_size,
            learning_rate=self.lr,
            n_layers=self.n_layers,
            ff_dim=self.ff_dim,
            nhead=self.nhead,
        )
        self.trainer_config = trainer_config

    def fit(self, X: np.ndarray, y=None):
        """
        Fit method of the network.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_timepoints, n_features)
        y : np.ndarray, optional
            Ignored.

        Returns
        -------
        self
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)

        n_s, n_t, n_f = X.shape
        X_tensor = torch.from_numpy(X).to(torch.float32)

        X_train, X_valid = train_test_split(
            X_tensor, test_size=0.2, shuffle=False
        )

        train_loader = DataLoader(
            TensorDataset(X_train, X_train),
            batch_size=self.batch_size,
            drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_valid, X_valid), batch_size=X_valid.shape[0]
        )

        trainer = pl.Trainer(**self.trainer_config)
        trainer.fit(self.model, train_loader, val_loader)
        return self

    def predict(self, X: np.ndarray):
        """
        Predict method
        Not implemented

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Raises
        ------
        NotImplementedError
            Not implemented
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        raise NotImplementedError()

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, )
            Reconstruction error as anomaly score
        """
        validate_data_3d(X)
        n_s, n_t, n_f = X.shape
        X_tensor = torch.from_numpy(X).to(torch.float32)
        eval_loader = DataLoader(
            TensorDataset(X_tensor, X_tensor), batch_size=n_s
        )

        trainer = pl.Trainer()
        errs = trainer.predict(self.model, eval_loader)
        return torch.cat(errs, dim=0).squeeze().detach().numpy()


class GDN_Det(BaseEstimator):
    """
    Wrapper class for GDN.
    Based on paper:
        TranAD: Deep Transformer Networks for Anomaly Detection in
        Multivariate Time Series Data,
        http://arxiv.org/abs/2201.07284

    Parameters
    ----------
    window_size : int
        Size of input
    n_hidden : int
        Size of hidden layer
    n_feats : int, default 1
        Number of channels of time series
    lr : float, default 1r-3
        Learning rate
    batch_size : int, default 256
        Number of samples in one batch
    trainer_config : dict
        Kwargs for pl.Trainer class when fitting the model
    """

    def __init__(
        self,
        window_size,
        n_hidden,
        n_feats=1,
        lr=1e-3,
        batch_size=256,
        trainer_config=None,
        **kwargs,
    ) -> None:
        self.name = 'gdn'
        self.fitted = False
        self.n_feats = n_feats
        self.window_size = window_size
        self.n_hidden = n_hidden
        self.lr = lr
        self.batch_size = batch_size
        self.model = GDN(
            n_feats=self.n_feats,
            window_size=self.window_size,
            n_hidden=self.n_hidden,
            learning_rate=self.lr,
        )
        self.trainer_config = trainer_config

    def fit(self, X: np.ndarray, y=None):
        """
        Fit method of the network.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_timepoints, n_features)
        y : np.ndarray, optional
            Ignored.

        Returns
        -------
        self
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)

        X_tensor = torch.from_numpy(X).to(torch.float32)

        X_train, X_valid = train_test_split(
            X_tensor, test_size=0.2, shuffle=False
        )

        y_train = X_train[:, -1, :]
        X_train = X_train[:, :-1, :]

        y_valid = X_valid[:, -1, :]
        X_valid = X_valid[:, :-1, :]

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_valid, y_valid), batch_size=X_valid.shape[0]
        )

        trainer = pl.Trainer(**self.trainer_config)
        trainer.fit(self.model, train_loader, val_loader)
        return self

    def predict(self, X: np.ndarray):
        """
        Predict method
        Not implemented

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Raises
        ------
        NotImplementedError
            Not implemented
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        raise NotImplementedError()

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, )
            Reconstruction error as anomaly score
        """
        validate_data_3d(X)
        X_tensor = torch.from_numpy(X).to(torch.float32)
        y_eval = X_tensor[:, -1, :]
        X_eval = X_tensor[:, :-1, :]

        eval_loader = DataLoader(
            TensorDataset(X_eval, y_eval), batch_size=X_eval.shape[0]
        )

        trainer = pl.Trainer()
        errs = trainer.predict(self.model, eval_loader)
        zeros = torch.zeros(X_eval.shape[1], 1)
        return torch.cat([zeros] + errs, dim=0).squeeze().numpy()


class GTA_Det(BaseEstimator):
    """
    Wrapper class for GTA.
    Based on paper:
        Learning Graph Structures with Transformer for Multivariate Time Series
        Anomaly Detection in IoT
        https://arxiv.org/abs/2104.03466

    Parameters
    ----------
    num_nodes : int
        Number of features (sensors/channels)
    seq_len : int
        Size of input to encoder
    label_len : int
        Number of steps from input to feed to decoder
    out_len : int
        Number of time steps to forecast
    num_levels : int
        Number of layers in GraphTemporalEmbedding module
    window_size : int
        Size of input to fit method must be equal to seq_len + out_len
    lr : float, default 1r-3
        Learning rate
    batch_size : int, default 256
        Number of samples in one batch
    trainer_config : dict
        Kwargs for pl.Trainer class when fitting the model

    """

    def __init__(
        self,
        num_nodes,
        seq_len,
        label_len,
        out_len,
        num_levels,
        window_size,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        lr=1e-3,
        batch_size=256,
        trainer_config=None,
        **kwargs,
    ) -> None:
        self.name = 'gta'
        self.fitted = False
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len
        self.num_levels = num_levels
        self.factor = factor
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.window_size = window_size
        self.lr = lr
        self.batch_size = batch_size
        self.model = GTA(
            factor=self.factor,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            num_nodes=self.num_nodes,
            seq_len=self.seq_len,
            label_len=self.label_len,
            out_len=self.out_len,
            num_levels=self.num_levels,
            learning_rate=self.lr,
        )
        self.trainer_config = trainer_config
        if seq_len + out_len != window_size:
            raise ValueError(
                (
                    f'seq_len + out_len has to == window_size'
                    f', {seq_len + out_len} != {window_size}'
                )
            )

    def fit(self, X: np.ndarray, y=None):
        """
        Fit method of the network.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, seq_len+out_len, n_features)
        y : np.ndarray, optional
            Ignored.

        Returns
        -------
        self
        """
        _LOGGER.info(f'Fitting {self.__class__.__name__}')
        validate_data_3d(X)

        X_tensor = torch.from_numpy(X).to(torch.float32)
        y_tensor = X_tensor[:, -(self.label_len + self.out_len) :, :]
        X_tensor = X_tensor[:, : self.seq_len, :]
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_tensor, y_tensor, test_size=0.2, shuffle=False
        )

        train_loader = DataLoader(
            TensorDataset(X_train, y_train, X_train, y_train),
            drop_last=True,
            batch_size=self.batch_size,
        )
        val_loader = DataLoader(
            TensorDataset(X_valid, y_valid, X_valid, y_valid),
            batch_size=X_valid.shape[0],
        )

        trainer = pl.Trainer(**self.trainer_config)
        trainer.fit(self.model, train_loader, val_loader)
        return self

    def predict(self, X: np.ndarray):
        """
        Predict method
        Not implemented

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Raises
        ------
        NotImplementedError
            Not implemented
        """
        _LOGGER.info(f'Predicting {self.__class__.__name__}')
        raise NotImplementedError()

    def predict_anomaly_scores(self, X: np.ndarray):
        """
        Predict anomaly scores

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_timepoints, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, )
            Reconstruction error as anomaly score
        """
        validate_data_3d(X)
        X_tensor = torch.from_numpy(X).to(torch.float32)

        y_eval = X_tensor[:, -(self.label_len + self.out_len) :, :]

        X_eval = X_tensor[:, : self.seq_len, :]

        eval_loader = DataLoader(
            TensorDataset(X_eval, y_eval, X_eval, y_eval),
            batch_size=X_eval.shape[0],
        )

        trainer = pl.Trainer()
        errs = trainer.predict(self.model, eval_loader)
        zeros = torch.zeros(X_eval.shape[1], 1)
        return torch.cat([zeros] + errs, dim=0).squeeze().numpy()
