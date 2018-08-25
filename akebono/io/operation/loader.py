import os
import re
from akebono.utils import (
    from_pickle,
    list_directory,
    pathjoin,
)
import akebono.settings as settings
import pandas as pd


def load_sklearn_model(model, dirpath, model_name):
    model._value = from_pickle(pathjoin(dirpath, '{}.pkl'.format(model_name)))
    if model._value is None:
        raise Exception('load {} failed.'.format(model_name))
    return model


def load_train_results(scenario_tag='default', train_ids='all'):
    dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
    file_paths = list_directory(dirpath)
    result_paths = [
        fp for fp in file_paths
        if re.search('.+\{}train_result_meta_[0-9]+\.pkl$'.format(os.sep), fp) is not None
    ]
    results = [
        from_pickle(rp)
        for rp in result_paths
    ]
    if not (isinstance(train_ids, list) or train_ids == 'all'):
        raise ValueError('train_ids must be list type or str "all"')
    if not train_ids == 'all':
        results = [r for r in results if r['id'] in train_ids]

    # convert evaluate result to pandas.DataFrame
    for idx, r in enumerate(results):
        if 'evaluate' in r:
            results[idx]['evaluate']['metrics'] = pd.DataFrame([
                {o['name']:o['value'] for o in met}
                for met in r['evaluate']['metrics']
            ])
    return results


def get_train_result(scenario_tag='default', train_id=0):
    rlist = load_train_results(scenario_tag=scenario_tag, train_ids=[train_id])
    if len(rlist) == 0:
        return None
    elif len(rlist) == 1:
        return rlist[0]
    else:
        raise Exception('unexpected result.')
