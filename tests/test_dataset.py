import sklearn.datasets as skl_datasets
from akebono.dataset.generator.sklearn import (
    make_regression,
)
import pandas as pd


def pd_assert_equal(a, b):
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        s_bool = (a != b)
        assert s_bool[s_bool].size == 0
    elif isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        df_bool = (a != b)
        columns = df_bool.columns
        for c in columns:
            assert df_bool[c][df_bool[c]].size == 0
    else:
        raise TypeError('invalid type')


def test_generator_make_regression():
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
    pd_assert_equal(pd.Series(y), pd.Series(y_raw))
    dfX = pd.DataFrame(X)
    pd_assert_equal(dfX, pd.DataFrame(X_raw, columns=dfX.columns))
