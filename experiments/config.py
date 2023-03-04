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
DATASET_DIR = DATA_DIR + 'datasets/'
MODEL_DIR = DIR + '/model/'
CONFIG_DIR = DIR + '/config/'
TRAIN_DATASETS = DATA_DIR + '/train.txt'
TEST_DATASETS = DATA_DIR + '/test.txt'
RESULTS_DIR = DIR + '/results/'


def get_hparams(trial, det, det_cfg):
    hp = det_cfg['hyperparams']
    match det:
        case 'knn':
            return dict(
                n_neighbors=trial.suggest_int(
                    'n_neighbors',
                    hp['n_neighbors']['min'],
                    hp['n_neighbors']['max'],
                    hp['n_neighbors']['step'],
                ),
                metric=trial.suggest_categorical('metric', hp['metric']),
            )
        case 'lof':
            return dict(
                n_neighbors=trial.suggest_int(
                    'n_neighbors',
                    hp['n_neighbors']['min'],
                    hp['n_neighbors']['max'],
                    hp['n_neighbors']['step'],
                ),
                metric=trial.suggest_categorical('metric', hp['metric']),
            )
        case 'isoforest':
            return dict(
                n_estimators=trial.suggest_categorical(
                    'n_estimators', hp['n_estimators']
                ),
                max_samples=trial.suggest_categorical(
                    'max_samples', hp['max_samples']
                ),
                max_features=trial.suggest_categorical(
                    'max_features', hp['max_features']
                ),
                bootstrap=trial.suggest_categorical(
                    'bootstrap', hp['bootstrap']
                ),
            )
        case 'dwtmlead':
            return dict(
                l=trial.suggest_int('l', hp['l']['min'], hp['l']['max']),
                epsilon=trial.suggest_float(
                    'epsilon', hp['epsilon']['min'], hp['epsilon']['max']
                ),
            )
        case 'ocsvm':
            return dict(
                nu=trial.suggest_float(
                    'nu', hp['nu']['min'], hp['nu']['max'], log=True
                )
            )
        case 'lstmae':
            return dict(
                hidden_size=trial.suggest_int(
                    'hidden_size',
                    hp['hidden_size']['min'],
                    hp['hidden_size']['max'],
                    hp['hidden_size']['step'],
                ),
                n_layers=trial.suggest_categorical('n_layers', hp['n_layers']),
                dropout=trial.suggest_categorical('dropout', hp['dropout']),
                batch_size=trial.suggest_categorical(
                    'batch_size', hp['batch_size']
                ),
                lr=trial.suggest_float(
                    'lr', hp['lr']['min'], hp['lr']['max'], log=True
                ),
            )
        case 'vae':
            return dict(
                encoder_hidden=trial.suggest_categorical(
                    'encoder_hidden', hp['encoder_hidden']
                ),
                decoder_hidden=trial.suggest_categorical(
                    'decoder_hidden', hp['decoder_hidden']
                ),
                batch_size=trial.suggest_categorical(
                    'batch_size', hp['batch_size']
                ),
                latent_dim=trial.suggest_int(
                    'latent_dim',
                    hp['latent_dim']['min'],
                    hp['latent_dim']['max'],
                ),
                lr=trial.suggest_float(
                    'lr', hp['lr']['min'], hp['lr']['max'], log=True
                ),
            )
        case 'tranad':
            return dict(
                n_layers=trial.suggest_categorical('n_layers', hp['n_layers']),
                ff_dim=trial.suggest_categorical('ff_dim', hp['ff_dim']),
                nhead=trial.suggest_categorical('nhead', hp['nhead']),
                batch_size=trial.suggest_categorical(
                    'batch_size', hp['batch_size']
                ),
                lr=trial.suggest_float(
                    'lr', hp['lr']['min'], hp['lr']['max'], log=True
                ),
            )
        case 'gta':
            return dict(
                num_levels=trial.suggest_categorical(
                    'num_levels', hp['num_levels']
                ),
                batch_size=trial.suggest_categorical(
                    'batch_size', hp['batch_size']
                ),
                factor=trial.suggest_categorical('factor', hp['factor']),
                d_model=trial.suggest_categorical('d_model', hp['d_model']),
                n_heads=trial.suggest_categorical('n_heads', hp['n_heads']),
                e_layers=trial.suggest_categorical('e_layers', hp['e_layers']),
                d_layers=trial.suggest_categorical('d_layers', hp['d_layers']),
                d_ff=trial.suggest_categorical('d_ff', hp['d_ff']),
                dropout=trial.suggest_categorical('dropout', hp['dropout']),
                lr=trial.suggest_float(
                    'lr', hp['lr']['min'], hp['lr']['max'], log=True
                ),
            )
        case _:
            raise ValueError(f'Detector {det} not supported.')


PYT_MODELS = [
    'gdn',
    'gta',
    'lstmae',
    'tranad',
    'vae',
]
