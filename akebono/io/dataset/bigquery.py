import akebono.settings as settings
import pandas as pd
import os
import jinja2

from akebono.logging import getLogger


logger = getLogger(__name__)


def get_template_env(sql_template_dir):
    return jinja2.Environment(
        loader = jinja2.FileSystemLoader(sql_template_dir, encoding='utf-8')
    )


def load_from_sql(sql):
    return _bqclient.query_sync(sql)


def render_sql(bqdataname, param, sql_template_dir, sql=None):
    if not isinstance(param, dict):
        raise TypeError('invalid type')
    if sql is None:
        return get_template_env(sql_template_dir).get_template('{}.sql'.format(bqdataname)).render(**param)
    else:
        return jinja2.Template(sql).render(**param)


def load(load_func_kwargs, param):
    logger.debug('bigquery loader invoked.')
    sql = param.get('sql')
    sql_template_dir = param.get('sql_template_dir', os.path.join(settings.project_root_dir, '_dataset/bq_sql_templates'))
    client_read_config = param.get('client_read_config', {
        'query': {
            'useLegacySql': False,
            'useQueryCache': False,
        },
        'jobTimeoutMs': 3600 * 1000,
    })
    if sql is not None and not isinstance(sql, str):
        raise TypeError('sql must be str.')
    _rendered_sql = render_sql(param['dataset_name'], load_func_kwargs, sql_template_dir, sql=sql)
    return pd.read_gbq(_rendered_sql, configuration=client_read_config)
