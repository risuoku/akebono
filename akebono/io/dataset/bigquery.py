from .client import Client as BqClient
import akebono.settings as settings
import pandas as pd
from akebono.logging import getLogger
from jinja2 import Template
from google.cloud import bigquery


class Client:
    def __init__(self):
        self._client = bigquery.Client()
        self._client.use_legacy_sql = False
        self._client.use_query_cache = False

    @property
    def client(self):
        return self._client

    def query_sync(self, q):
        qr = self.client.query(q)
        qrresult = qr.result(timeout=3600.0)
        rows = list(qrresult)
        snames = [a.name for a in qrresult.schema]
        r = [dict(zip(snames, [row[sn] for sn in snames])) for row in rows]
        return r


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


def load(load_func_kwargs, param):
    logger.debug('bigquery loader invoked.')
    sql = param.get('sql')
    if sql is not None and not isinstance(sql, str):
        raise TypeError('sql must be str.')
    _rendered_sql = render_sql(param['dataset_name'], load_func_kwargs, sql=sql)
    return pd.DataFrame(load_from_sql(_rendered_sql))
