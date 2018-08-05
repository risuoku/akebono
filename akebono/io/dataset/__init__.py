from akebono.utils import load_object_by_str


def load_dataset(dataset_config):
    if ('name' not in dataset_config) or ('load_func' not in dataset_config):
        raise Exception('name and load_func must be specified.')
    load_func = load_object_by_str(dataset_config['load_func'])
    load_func_kwargs = dataset_config.get('load_func_kwargs', {})
    return load_func(dataset_config['name'], **load_func_kwargs)
