from functools import partial
from typing import Tuple

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve


def point_adjust(
    labels: np.ndarray, scores: np.ndarray, threshold: float, k: float = 0
):
    '''
    Code from: https://github.com/tuslkkk/tadpak/blob/master/tadpak/pak.py

    The implementation of PA%K evaluation protocol of Towards a Rigorous
    Evaluation of Time-sereis anomaly detection, Siwon Kim, Kukjin Choi,
    Hyun-Soo Choi, Byunghan Lee, and Sungroh Yoon, AAAI 2022.

    This function point adjusts predictions.

    Assumption is that in real scenario it's sufficient to detect just part of
    the anomaly, not the entire sequence.

    If just k% of anomaly sequence is detected as anomaly count the entire
    sequence as True positive.

    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels
    scores : np.ndarray
        Anomaly scores
    threshold : float
        Anomaly threshold
    k : float, optional
        Value in [0,1] , by default 0

    Returns
    -------
    np.ndarray
        Point adjusted predictions
    '''

    predicts = scores > threshold

    anml_seg_starts = np.where(np.diff(labels, prepend=0) == 1)[0]
    anml_seg_ends = np.where(np.diff(labels, prepend=0) == -1)[0]

    # case where labels end with anomaly (labels[end] == 1)
    if len(anml_seg_starts) == len(anml_seg_ends) + 1:
        anml_seg_ends = np.append(anml_seg_ends, len(predicts))

    for i in range(len(anml_seg_starts)):
        if predicts[anml_seg_starts[i] : anml_seg_ends[i]].sum() > k * (
            anml_seg_ends[i] - anml_seg_starts[i]
        ):
            predicts[anml_seg_starts[i] : anml_seg_ends[i]] = 1
    return predicts


def f1_pa(
    labels: np.ndarray, scores: np.ndarray, threshold: float, k: float = 0
):
    '''
    F1 score with point adjusted protocol @ k

    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels
    scores : np.ndarray
        Anomaly scores
    threshold : float
        Anomaly threshold
    k : float, optional
        Value in [0,1] , by default 0

    Returns
    -------
    float
        F1 score on point adjusted predictions
    '''
    preds = point_adjust(labels, scores, threshold, k)
    return f1_score(labels, preds)


def f1_pa_curve(labels: np.ndarray, scores: np.ndarray, threshold: float):
    '''
    Curve for f1 score at different k
    for point adjusted predictions.

    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels
    pred : np.ndarray
        Predicted labels

    Returns
    -------
    np.ndarray
        Different k values
    np.ndarray
        F1 score @ k values
    '''
    f1_pa_k = np.zeros((11,))
    ks = np.zeros((11,))
    for i, k in enumerate(np.arange(0, 1.1, 0.1)):
        f1_pa_k[i] = f1_pa(labels, scores, threshold, k)
        ks[i] = k
    return ks, f1_pa_k


def f1_pa_auc_score(labels: np.ndarray, scores: np.ndarray, threshold: float):
    '''
    Area under curve for f1 score at different k
    for point adjusted predictions.

    Parameters
    ----------
    labels : np.ndarray
        Ground truth labels
    pred : np.ndarray
        Predicted labels

    Returns
    -------
    float
        Area under curve in [0,1] range
    '''
    k, f1 = f1_pa_curve(labels, scores, threshold)
    return auc(k, f1)


def mass_volume_curve(
    detector,
    X: np.ndarray,
    scores: np.ndarray = None,
    alpha_min: float = 0.9,
    alpha_max: float = 0.999,
    alphas_count: int = 100,
    mc_samples_count: int = 32768,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mass-volume curve

    Parameters
    ----------
    detector : object
        detector object
    X : np.ndarray
        input feature matrix, usually test data
    scores : np.ndarray, optional
        scores for the input feature matrix, by default None
    alpha_min : float, optional
        minimum threshold, by default 0.9
    alpha_max : float, optional
        maximum threshold, by default 0.999
    alphas_count : int, optional
        number of thresholds, by default 128
    mc_samples_count : int, optional
        number of monte carlo samples, by default 131072

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two arrays, first contains the threshold (mass) values and second contains the volume values
    """
    if scores is None:
        scores = detector.predict_anomaly_scores(X)

    # MV paper assumes that lower score corresponds to higher abnormality
    scores_vec = -scores
    alphas_vec = np.linspace(alpha_min, alpha_max, alphas_count)
    ro_vec = np.quantile(scores_vec, 1 - alphas_vec)

    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    # volume of hypercube enclosing the data
    volume = np.prod(X_max - X_min)

    n_samples, n_features, *_ = X.shape
    mc_samples = np.random.uniform(
        X_min, X_max, (mc_samples_count, n_features)
    )
    scores_samples = -detector.predict_anomaly_scores(mc_samples)

    mv_alpha = np.zeros_like(alphas_vec)
    for i, ro in enumerate(ro_vec):
        mv_alpha[i] = (scores_samples >= ro).sum() / mc_samples_count * volume
    return alphas_vec, mv_alpha


def mass_volume_auc_score(
    detector,
    X: np.ndarray,
    scores: np.ndarray = None,
    alpha_min: float = 0.9,
    alpha_max: float = 0.999,
    alphas_count: int = 100,
    mc_samples_count: int = 32768,
) -> float:
    """
    Compute the area under the mass-volume curve (AUC)

    Parameters
    ----------
    detector : object
        detector object
    X : np.ndarray
        input feature matrix. Usually test data.
    scores : np.ndarray
        scores for the input feature matrix, by default None
    alpha_min : float, optional
        minimum threshold, by default 0.9
    alpha_max : float, optional
        maximum threshold, by default 0.999
    alphas_count : int, optional
        number of thresholds, by default 128
    mc_samples_count : int, optional
        number of monte carlo samples, by default 131072

    Returns
    -------
    float
        AUC score
    """
    alphas, mvs = mass_volume_curve(
        detector,
        X,
        scores,
        alpha_min,
        alpha_max,
        alphas_count,
        mc_samples_count,
    )
    return auc(alphas, mvs)


def feature_subsampling_auc(
    detector,
    X_train: np.ndarray,
    X_test: np.ndarray,
    auc_func,
    n_subfeatures: int = 5,
    n_tries: int = 50,
) -> float:
    """
    Compute AUC for given metric on feature subsample trained detector

    Parameters
    ----------
    detector : object
        detector object
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Testing feature matrix
    auc_func : function
        function to compute AUC
    n_subfeatures : int, optional
        number of subsampled features, by default 5
    n_tries : int, optional
        number of subsampling tries, by default 50

    Returns
    -------
    float
        mean AUC across subsampling tries
    """
    _, n_features = X_train.shape
    aucs = np.zeros((n_tries,))
    for i in range(n_tries):
        features = np.random.choice(
            a=n_features, size=n_subfeatures, replace=False
        )
        X_train_ = X_train[:, features]
        X_test_ = X_test[:, features]
        detector.fit(X_train_)
        aucs[i] = auc_func(detector, X_test_)
    return np.mean(aucs)


def mv_feature_subsampling_auc_score(
    detector,
    X_train: np.ndarray,
    X_test: np.ndarray,
    alpha_min: float = 0.9,
    alpha_max: float = 0.999,
    alphas_count: int = 100,
    mc_samples_count: int = 32768,
    n_subfeatures: int = 5,
    n_tries: int = 50,
) -> float:
    """
    Compute mean AUC of mass volume metric on feature subsample trained
    detector

    Parameters
    ----------
    detector : object
        detector object
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Testing feature matrix
    auc_func : function
        function to compute AUC
    n_subfeatures : int, optional
        number of subsampled features, by default 5
    n_tries : int, optional
        number of subsampling tries, by default 50

    Returns
    -------
    float
        mean AUC across subsampling tries
    """
    mvfunc = partial(
        mass_volume_auc_score,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alphas_count=alphas_count,
        mc_samples_count=mc_samples_count,
    )
    return feature_subsampling_auc(
        detector, X_train, X_test, mvfunc, n_subfeatures, n_tries
    )


def excess_mass_curve(
    detector,
    X: np.ndarray,
    scores: np.ndarray = None,
    t_count: int = 2048,
    mc_samples_count: int = 32768,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Excess Mass Curve (EM) for an anomaly detector.

    Reference: http://arxiv.org/abs/1502.01684

    Parameters
    ----------
    detector: object
        An anomaly detector implementing a 'predict_anomaly_scores' method
    X : np.ndarray of shape (n_samples, n_features)
        The dataset to evaluate the detector on.
    scores : np.ndarray of shape (n_samples,)
        The anomaly scores for the samples in `X`, if already computed.
        If not provided, the detector's 'predict_anomaly_scores' method will be used to compute them.
    t_count: int, default=2048
        The number of points at which to evaluate the EM
    mc_samples_count : int, default=32768
        The number of random samples to use for the Monte Carlo estimation of the EM

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        The first element of tuple is the array of threshold values used to compute the EM
        The second element of tuple is the corresponding EM values
    """
    if scores is None:
        scores = detector.predict_anomaly_scores(X)

    # EM paper assumes that lower score corresponds to higher abnormality
    scores_vec = -scores
    us = np.unique(scores_vec)

    # volume of hypercube enclosing the data
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    volume = np.prod(X_max - X_min)

    # Monte Carlo sampling
    n_samples, n_features, *_ = X.shape
    mc_samples = np.random.uniform(
        X_min, X_max, (mc_samples_count, n_features)
    )
    scores_samples = -detector.predict_anomaly_scores(mc_samples)

    ts = np.linspace(0, 100 / volume, t_count)
    ems = np.zeros_like(ts)
    ems[0] = 1.0
    for u in us:
        mass = (scores_vec > u).sum() / n_samples
        vol = (scores_samples > u).sum() / mc_samples_count * volume
        ems = np.maximum(ems, mass - ts * vol)
    return ts, ems


def excess_mass_auc_score(
    detector,
    X: np.ndarray,
    scores: np.ndarray = None,
    t_count: int = 2048,
    em_max: float = 0.9,
    mc_samples_count: int = 32768,
) -> float:
    """
    Compute the area under the Excess Mass curve (AUC-EM) for a given detector and data.
    The Excess Mass curve is a measure of the performance of a detector on a given dataset.
    AUC-EM is a single number measure of the detector performance that can be used for comparison.

    Parameters
    ----------
        detector: object
            An object which has a fit method and a predict_anomaly_scores method.
        X: np.ndarray, shape (n_samples, n_features)
            The input data.
        scores: np.ndarray, shape (n_samples,), optional (default=None)
            Anomaly scores for the samples. If None, detector.predict_anomaly_scores(X) will be used.
        t_count: int, optional default=2048
            Number of threshold values to use in the computation of the EM curve.
        em_max: float, optional (default=0.9)
            Maximum value for the y-axis of the EM curve.
        mc_samples_count: int, optional (default=32768)
            Number of samples to use in the Monte Carlo sampling of the volume.

    Returns
    -------
        float
            The area under the Excess Mass curve.
    """
    ts, ems = excess_mass_curve(
        detector=detector,
        X=X,
        scores=scores,
        t_count=t_count,
        mc_samples_count=mc_samples_count,
    )

    em_max_idx = np.argmax(ems < em_max)
    if em_max_idx == 0:
        em_max_idx = ems.shape[0]
    ts = ts[:em_max_idx]
    ems = ems[:em_max_idx]
    return auc(ts, ems)


def em_feature_subsampling_auc_score(
    detector,
    X_train: np.ndarray,
    X_test: np.ndarray,
    t_count: int = 2048,
    em_max: float = 0.9,
    mc_samples_count: int = 32768,
    n_subfeatures: int = 5,
    n_tries: int = 50,
) -> float:
    """
    Compute the EM-AUC score for a detector on a given test set, using subsampling of features.

    Parameters
    ----------
    detector : object
        An object with a `predict_anomaly_scores` method that takes in a numpy array and returns anomaly scores.
    X_train : numpy array
        The training set.
    X_test : numpy array
        The test set.
    t_count : int, optional by default = 2048
        Number of points on the EM curve.
    em_max : float, optional by default = 0.9
        The maximum value of the EM curve.
    mc_samples_count : int, optional by default = 32768
        Number of samples to use for Monte Carlo sampling.
    n_subfeatures : int, optional by default = 5
        Number of features to subsample for each try.
    n_tries : int, optional by default = 50
        Number of subsampling tries.

    Returns
    -------
    float
        The EM-AUC score.
    """
    emfunc = partial(
        excess_mass_auc_score,
        t_count=t_count,
        em_max=em_max,
        mc_samples_count=mc_samples_count,
    )
    return feature_subsampling_auc(
        detector, X_train, X_test, emfunc, n_subfeatures, n_tries
    )
