from akebono.utils import (
    load_object_by_str,
    cache_located_at,
    pathjoin,
    Param,
    get_hash,
)
import akebono.settings as settings
from .model import Dataset
from .state import datasetholder
from akebono.logging import getLogger
import copy


logger = getLogger(__name__)


_loader_name_alias = {
    'bigquery': 'load@akebono.dataset.bigquery',
    'csv': 'load@akebono.dataset.csv',
    'iris': 'load_iris@akebono.dataset.generator.sklearn',
    'regression_sample': 'make_regression@akebono.dataset.generator.sklearn',
    'binary_classifier_sample_moon': 'make_moons@akebono.dataset.generator.sklearn',
}


def get_dataset(dataset_config):
    """
    Datasetを生成するための関数

    :param dataset_config: Datasetについての設定
    :type dataset_config: dict
    :return: :class:`Dataset` object

    Usage:
        >>> from akebono.dataset import get_dataset
        >>> dataset_config = {
                'loader_config': {
                    'name': 'make_regression@akebono.dataset.generator.sklearn',
                    'kwargs': {
                        'n_features': 1,
                        'noise': 30.0,
                        'random_state': 0,
                    },
                },
                'target_column': 'target',
                'cache_enabled': False,
            }
        >>> ds = get_dataset(dataset_config)
        >>> ds
        <akebono.dataset.model.Dataset object at 0x11291acc0>
    """

    dataset_name = dataset_config.get('name')
    target_column = dataset_config.get('target_column', 'target')
    cache_enabled = dataset_config.get('cache_enabled', False)
    evacuated_columns = dataset_config.get('evacuated_columns', [])
    if not isinstance(evacuated_columns, list):
        raise TypeError('evacuated_columns must be list.')
    loader_config = dataset_config.get('loader_config')
    if not isinstance(loader_config, dict):
        raise Exception('loader_config must be specified and this type is dict.')
    load_func = loader_config.get('name')
    load_func = _loader_name_alias.get(load_func, load_func) # エイリアスがあったらそれを使う
    if load_func is None:
        raise Exception('loader_config.name must be specified.')
    load_func = load_object_by_str(load_func)
    load_func_kwargs = Param(loader_config.get('kwargs', {}))
    loader_param = loader_config.get('param', {})
    _reserved_params = ('dataset_name', 'target_column',)
    for rp in _reserved_params:
        if rp in loader_param:
            raise KeyError('{} is reserved param.'.format(rp))
    loader_param['dataset_name'] = dataset_name
    loader_param['target_column'] = target_column

    preprocess_func_str = dataset_config.get('preprocess_func', 'identify@akebono.dataset.preprocessors')
    preprocess_func_hash = get_hash(preprocess_func_str)
    preprocess_func = load_object_by_str(preprocess_func_str)
    preprocess_func_kwargs = Param(dataset_config.get('preprocess_func_kwargs', {}))

    def _core_func():
        return preprocess_func(
            load_func(copy.deepcopy(load_func_kwargs.value), loader_param),
            **copy.copy(preprocess_func_kwargs.value)
        )

    fname = '{}_{}_{}_{}'.format(
        dataset_name,
        load_func_kwargs.get_hashed_id(length=24),
        preprocess_func_hash[:24],
        preprocess_func_kwargs.get_hashed_id(length=24)
    )
    dataset_loading_cache_enabled = dataset_config.get('dataset_loading_cache_enabled', True)
    if dataset_loading_cache_enabled:
        ds = datasetholder.get(fname)
        if ds is not None:
            logger.debug('dataset_loading_cache enabled .. {} get done.'.format(ds.name))
            return ds

    pkl_fname = fname + '.pkl'
    if cache_enabled:
        if dataset_name is not None:
            logger.info('dataset cache enabled')
            _core_func = cache_located_at(pathjoin(settings.cache_dir, pkl_fname))(_core_func)
        else:
            raise Exception('dataset_config.cache_enabled is True, but dataset_config.name is None')

    ds = Dataset(fname, _core_func(), target_column, evacuated_columns)

    if dataset_loading_cache_enabled:
        datasetholder.set(ds)
        logger.debug('dataset_loading_cache enabled .. {} set done.'.format(ds.name))
    return ds
