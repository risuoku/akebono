import sklearn.datasets as skl_datasets
from akebono.dataset.generator.sklearn import (
    make_regression,
)
import pandas as pd


def test_generator_make_regression(pd_assert_equal):
    kwargs = {
        'random_state': 0,
        'coef': False,
    }
    r = make_regression(
        origin_func_kwargs=kwargs,
        target_column='target'
    )
    X, y = r.get_predictor_target()
    r_raw = skl_datasets.make_regression(**kwargs)
    X_raw, y_raw = r_raw
    pd_assert_equal(y, pd.Series(y_raw))
    pd_assert_equal(X, pd.DataFrame(X_raw, columns=X.columns))
