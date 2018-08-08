from akebono.utils import load_object_by_str


def load_dataset(dataset_config):
    if 'load_func' not in dataset_config:
        raise Exception('load_func must be specified.')
    load_func = load_object_by_str(dataset_config['load_func'])
    load_func_kwargs = dataset_config.get('load_func_kwargs', {})
    return load_func(**load_func_kwargs)
