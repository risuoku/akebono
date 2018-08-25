import sys
import os
import re
import jinja2


self = sys.modules[__name__]
_init = False


_valid_attributes = [
    'storage_root_dir',
    'storage_type',
    'storage_option',
    'bq_sql_template_dir',
    'bq_read_config',
    'project_name',
    'project_root_dir',
    'train_config',
    'predict_config',
]


def _update_associated_attrs():
    self.storage_project_root_dir = pathjoin(storage_root_dir, project_name)
    self.cache_dir = pathjoin(storage_project_root_dir, 'cache')
    self.operation_results_dir = pathjoin(storage_project_root_dir, 'operation_results')


def init():
    if storage_type == 'local':
        os.makedirs(storage_project_root_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(operation_results_dir, exist_ok=True)
        os.makedirs(datasource_dir, exist_ok=True)
    
    
def get_template_env():
    return jinja2.Environment(
        loader = jinja2.FileSystemLoader(bq_sql_template_dir, encoding='utf-8')
    )


def apply(config):
    """
    configを適用するための関数

    :param config: akebonoの設定
    :type config: python module object
    """

    # phase1.
    for va in _valid_attributes:
        value = getattr(config, va, None)
        if value is not None:
            setattr(self, va, value)

    _update_associated_attrs()

    # phase2. _update_associated_attrに依存する設定を反映させるため、
    # phase1と同じ処理を入れる
    for va in _valid_attributes:
        value = getattr(config, va, None)
        if value is not None:
            setattr(self, va, value)


def get_train_configs():
    """
    train_configのリストを返す関数

    :return: list[dict]
    """
    if isinstance(train_config, dict):
        return [train_config]
    elif isinstance(train_config, list):
        return train_config
    else:
        raise TypeError('invalid type .. train_config must be dict or list.')


def get_predict_configs():
    """
    predict_configのリストを返す関数

    :return: list[dict]
    """
    if isinstance(predict_config, dict):
        return [predict_config]
    elif isinstance(predict_config, list):
        return predict_config
    else:
        raise TypeError('invalid type .. predict_config must be dict or list.')


def reset():
    self.storage_root_dir = '_storage'
    self.storage_type = 'local'
    self.storage_option = {}
    self.project_name = 'default'
    self.project_root_dir = os.getcwd()
    self.bq_sql_template_dir = os.path.join(project_root_dir, '_dataset/bq_sql_templates')
    self.bq_read_config = {
        'query': {
            'useLegacySql': False,
            'useQueryCache': False,
        },
        'jobTimeoutMs': 3600 * 1000,
    }
    self.train_config = []
    self.predict_config = []
    self.datasource_dir = pathjoin(storage_root_dir, project_name, 'datasource')
    _update_associated_attrs()


### default settings
### settings moduleロード時に一度だけ実行される

if not _init:
    _pathjoin_gcs_pattern = re.compile('^(\/+)([^/].*)$')
    def pathjoin(*args, **kwargs):
        if self.storage_type == 'local':
            return os.path.join(*args, **kwargs)
        elif self.storage_type == 'gcs':
            r = '/'.join(args)
            reg = re.search(_pathjoin_gcs_pattern, r)
            if reg is not None:
                r = reg.group(2)
            return r
        else:
            raise ValueError('invalid storage_type')

    reset()
    _init = True
