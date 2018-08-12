import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def pd_assert_equal():
    def _func(a, b):
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            if a.dtype is np.dtype('float64') and b.dtype is np.dtype('float64'):
                s_bool = (a - b).abs().apply(lambda s: s > 0.00001)
            else:
                s_bool = a != b
            assert s_bool[s_bool].size == 0
        elif isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            for c in a.columns:
                s_a = a[c]
                s_b = b[c]
                if s_a.dtype is np.dtype('float64') and s_b.dtype is np.dtype('float64'):
                    s_bool = (s_a - s_b).abs().apply(lambda s: s > 0.00001)
                else:
                    s_bool = s_a != s_b
                assert s_bool[s_bool].size == 0
        else:
            raise TypeError('invalid type')
    return _func
