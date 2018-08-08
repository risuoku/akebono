import sys
import os
import jinja2


self = sys.modules[__name__]
_init = False


_valid_attributes = [
    'storage_root_dir',
    'storage_type',
    'storage_option',
    'bq_sql_template_dir',
    'project_name',
    'project_root_dir',
    'train_operations',
    'predict_operations',
]


def _update_associated_attrs():
    _storage_project_root_dir = os.path.join(storage_root_dir, project_name)
    self.cache_dir = os.path.join(_storage_project_root_dir, 'cache')
    self.operation_results_dir = os.path.join(_storage_project_root_dir, 'operation_results')
    if storage_type == 'local' and storage_auto_create_dir:
        os.makedirs(_storage_project_root_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(operation_results_dir, exist_ok=True)
    
    
def get_template_env():
    _bq_tpl_absdir = os.path.join(project_root_dir, bq_sql_template_dir)
    return jinja2.Environment(
        loader = jinja2.FileSystemLoader(_bq_tpl_absdir, encoding='utf-8')
    )


def load(config):
    for va in _valid_attributes:
        value = getattr(config, va, None)
        if value is not None:
            setattr(self, va, value)
    _update_associated_attrs()


### default settings
### settings moduleロード時に一度だけ実行される

if not _init:
    storage_root_dir = '_storage'
    storage_type = 'local'
    storage_option = {}
    storage_auto_create_dir = True
    bq_sql_template_dir = '_templates/bq_sql'
    project_name = 'default'
    project_root_dir = os.getcwd()
    train_operations = []
    predict_operations = []
    
    _update_associated_attrs()
    _init = True
