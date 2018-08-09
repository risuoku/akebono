import sklearn.datasets as skl_datasets
from akebono.dataset.model import Dataset
from akebono.utils import load_object_by_str
import pandas as pd


def make_regression(
    origin_func_kwargs={},
    target_column='target',
    preprocess_func='identify@akebono.dataset.preprocessors',
    preprocess_func_kwargs={}
    ):
    origin_func_kwargs['coef'] = False
    X, y = skl_datasets.make_regression(**origin_func_kwargs)
    lenx, leny = X.shape
    df = pd.DataFrame(X, columns=['x' + str(i+1) for i in range(leny)])
    if target_column is not None:
        df[target_column] = y

    preprocess_func = load_object_by_str(preprocess_func)
    df = preprocess_func(df, **preprocess_func_kwargs)
    return Dataset(df, target_column)
