## Baseline

1) Window size
2) KNN, LOF, IFOREST
3) Evaluate using labels

## Unsupervised AutoML pipeline

1) Window size
2) Hyperopt optuna train on test datasets
3) Select best using MV, EM
4) Evaluate using labels

## Metalearning

1) Window size
2) Extract meta features
3) Hyperopt on train datasets
4) Compute F1, F1_PA, AUCPR
5) Reduce dimensionality
6) Matrix factor
7) Multioutput regressor
8) Predict for test datasets
9) Use hyperparams from training as warmstart

## Combine Unsupervised and metalearning

1) Window size
2) Extract meta features
3) Hyperopt on train datasets
4) Compute F1, F1_PA, AUCPR
5) Reduce dimensionality
6) Matrix factor
7) Multioutput regressor
8) Predict for test datasets
9) Use hyperparams from training as warmstart for unsupervised training
10) Select best using MV, EM
11) Evaluate using labels
