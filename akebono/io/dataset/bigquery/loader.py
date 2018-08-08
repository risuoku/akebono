from .client import Client as BqClient
import akebono.settings as settings
from akebono.dataset.model import Dataset
from akebono.utils import (
    Param,
    cache_located_at,
    load_object_by_str,
)
import pandas as pd
import os


_bqclient = BqClient()


def load_from_sql(sql):
    return _bqclient.query_sync(sql)


def render_sql(bqdataname, param):
    if not isinstance(param, dict):
        raise TypeError('invalid type')
    return settings.get_template_env().get_template('{}.sql'.format(bqdataname)).render(**param)


def load(
    name=None,
    target_column='target', io_func_kwargs={},
    preprocess_func='identify@akebono.dataset.preprocessors',
    preprocess_func_kwargs={},
    cache_enabled=True,
    ):
    if name is None:
        raise Exception('dataset name must be specified for bigquery loader.')
    _p_io = Param(io_func_kwargs)
    _p2 = Param(preprocess_func_kwargs)
    fname = '{}_{}{}.pkl'.format(name, _p_io.get_hashed_id(length=32), _p2.get_hashed_id(length=32))

    preprocess_func = load_object_by_str(preprocess_func)

    def _func():
        sql = render_sql(name, _p_io.value)
        r = pd.DataFrame(load_from_sql(sql))
        return preprocess_func(r, **preprocess_func_kwargs)

    if cache_enabled:
        _func = cache_located_at(os.path.join(settings.cache_dir, fname))(_func)

    return Dataset(_func(), target_column)
