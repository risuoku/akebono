from .client import Client as BqClient
import akebono.settings as settings
from akebono.dataset.model import Dataset
from akebono.utils import (
    Param,
    cache_located_at,
    load_object_by_str,
    pathjoin,
)
import pandas as pd
from akebono.logging import getLogger
from jinja2 import Template


logger = getLogger(__name__)
_bqclient = BqClient()


def load_from_sql(sql):
    return _bqclient.query_sync(sql)


def render_sql(bqdataname, param, sql=None):
    if not isinstance(param, dict):
        raise TypeError('invalid type')
    if sql is None:
        return settings.get_template_env().get_template('{}.sql'.format(bqdataname)).render(**param)
    else:
        return Template(sql).render(**param)


def load(
    name=None,
    sql=None,
    target_column='target', origin_func_kwargs={},
    preprocess_func='identify@akebono.dataset.preprocessors',
    preprocess_func_kwargs={},
    cache_enabled=True,
    ):
    if name is None:
        raise Exception('dataset name must be specified for bigquery loader.')
    if sql is not None and not isinstance(sql, str):
        raise TypeError('sql must be str.')
    _p_io = Param(origin_func_kwargs)
    _p2 = Param(preprocess_func_kwargs)
    fname = '{}_{}{}.pkl'.format(name, _p_io.get_hashed_id(length=32), _p2.get_hashed_id(length=32))

    preprocess_func = load_object_by_str(preprocess_func)

    def _func():
        _rendered_sql = render_sql(name, _p_io.value, sql=sql)
        r = pd.DataFrame(load_from_sql(_rendered_sql))
        return preprocess_func(r, **preprocess_func_kwargs)

    if cache_enabled:
        logger.info('bigquery loader cache enabled .. load from cache if {} exists.'.format(name))
        _func = cache_located_at(pathjoin(settings.cache_dir, fname))(_func)

    return Dataset(_func(), target_column)
