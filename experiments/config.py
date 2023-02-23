from automltsad.detectors import (
    DWTMLEAD,
    KNN,
    LOF,
    OCSVM,
    GDN_Det,
    GTA_Det,
    IsolationForestAD,
    LSTM_AE_Det,
    RandomForest,
    TranAD_Det,
    VAE_Det,
)

NAME_TO_MODEL = dict(
    dwtmlead=DWTMLEAD,
    knn=KNN,
    lof=LOF,
    ocsvm=OCSVM,
    gdn=GDN_Det,
    gta=GTA_Det,
    isoforest=IsolationForestAD,
    lstmae=LSTM_AE_Det,
    randomforest=RandomForest,
    tranad=TranAD_Det,
    vae=VAE_Det,
)

DIR = './experiments'
DATA_DIR = DIR + '/data/'
DATASET_DIR = DIR + DATA_DIR + 'datasets/'
MODEL_DIR = DIR + '/models/'
CONFIG_DIR = DIR + '/config/'
TRAIN_DATASETS = DATA_DIR + '/train.txt'
TEST_DATASETS = DATA_DIR + '/test.txt'
