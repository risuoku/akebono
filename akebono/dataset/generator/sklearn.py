import sklearn.datasets as skl_datasets
import pandas as pd
from akebono.logging import getLogger


logger = getLogger(__name__)


def make_regression(load_func_kwargs, param):
    logger.debug('sklearn make_regression loader invoked.')
    load_func_kwargs['coef'] = False
    X, y = skl_datasets.make_regression(**load_func_kwargs)
    lenx, leny = X.shape
    df = pd.DataFrame(X, columns=['x' + str(i+1) for i in range(leny)])
    target_column = param['target_column']
    if target_column is not None:
        df[target_column] = y
    return df
