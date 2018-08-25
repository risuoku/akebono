import akebono.settings as settings
import pandas as pd
from akebono.logging import getLogger
from jinja2 import Template


logger = getLogger(__name__)


def load_from_sql(sql):
    return _bqclient.query_sync(sql)


def render_sql(bqdataname, param, sql=None):
    if not isinstance(param, dict):
        raise TypeError('invalid type')
    if sql is None:
        return settings.get_template_env().get_template('{}.sql'.format(bqdataname)).render(**param)
    else:
        return Template(sql).render(**param)


def load(load_func_kwargs, param):
    logger.debug('bigquery loader invoked.')
    sql = param.get('sql')
    if sql is not None and not isinstance(sql, str):
        raise TypeError('sql must be str.')
    _rendered_sql = render_sql(param['dataset_name'], load_func_kwargs, sql=sql)
    return pd.read_gbq(_rendered_sql, configuration=settings.bq_read_config)
