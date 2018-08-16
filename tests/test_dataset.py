import sklearn.datasets as skl_datasets
from akebono.dataset import get_dataset
import pandas as pd


def test_generator_make_regression(pd_assert_equal):
    dataset_config = {
        'loader_config': {
            'func': 'make_regression@akebono.dataset.generator.sklearn',
            'func_kwargs': {
                'random_state': 0,
                'coef': False,
            },
        },
        'target_column': 'target',
    }
    r = get_dataset(dataset_config)
    X, y = r.get_predictor_target()
    r_raw = skl_datasets.make_regression(**dataset_config['loader_config']['func_kwargs'])
    X_raw, y_raw = r_raw
    pd_assert_equal(y, pd.Series(y_raw))
    pd_assert_equal(X, pd.DataFrame(X_raw, columns=X.columns))


def test_generator_load_iris(pd_assert_equal):
    dataset_config = {
        'loader_config': {
            'func': 'load_iris@akebono.dataset.generator.sklearn',
            'func_kwargs': {},
        },
        'target_column': 'target',
    }
    r = get_dataset(dataset_config)
    X, y = r.get_predictor_target()
    r_raw = skl_datasets.load_iris(**dataset_config['loader_config']['func_kwargs'])
    X_raw, y_raw = r_raw.data, r_raw.target
    pd_assert_equal(y, pd.Series(y_raw))
    pd_assert_equal(X, pd.DataFrame(X_raw, columns=X.columns))
