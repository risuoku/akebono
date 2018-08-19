from akebono.utils import (
    pd_read_csv,
    pathjoin,
)
from akebono.logging import getLogger
import akebono.settings as settings


logger = getLogger(__name__)


def load(load_func_kwargs, param):
    logger.debug('bigquery loader invoked.')
    if param['dataset_name'] is None:
        raise ValueError('dataset_name must be set for csv loader')
    fname = param['dataset_name'] + '.csv'
    df = pd_read_csv(pathjoin(settings.datasource_dir, fname), **load_func_kwargs)
    return df
