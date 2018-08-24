from akebono.io.operation.loader import load_train_results
from akebono.utils import (
    list_directory,
    pathjoin,
)
import akebono.settings as settings
import pandas as pd
import copy
import re


def get_scenario_summary(scenario_tag, performance_sort_key):
    trs = load_train_results(scenario_tag=scenario_tag, train_ids='all')
    trs = [tr for tr in trs if tr['evaluate_enabled'] and tr['dump_result_enabled']]
    if len(trs) < 1:
        raise Exception('evaluated result not found.')
    if performance_sort_key is not None and performance_sort_key not in trs[0]['evaluate']['metrics'].columns:
        raise KeyError('invalid key .. valid keys: {}'.format(list(trs[0]['evaluate']['metrics'].columns)))
    all_metrics = []
    for tr in trs:
        mr = tr['evaluate']['metrics'].mean()
        trcp = copy.deepcopy(tr)
        trcp.pop('evaluate')
        trcp.pop('evaluate_enabled')
        trcp.pop('dump_result_enabled')
        mr['_akebono_train'] = trcp
        all_metrics.append(mr)

    concated = pd.concat(all_metrics, axis=1).T.reset_index(drop=True)
    if performance_sort_key is not None:
        concated = concated.sort_values(performance_sort_key, ascending=False)
    return concated


_train_result_meta_pattern = re.compile('^train_result_meta_([^. ]+)\.(\S+)$')
_predict_result_meta_pattern = re.compile('^predict_result_meta_([^. ]+)\.(\S+)$')
def get_scenario_ids(scenario_tag):
    dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
    filenames = list_directory(dirpath, mode='filename')
    train_regexps = [re.search(_train_result_meta_pattern, fn) for fn in filenames]
    train_ids = [tr.group(1) for tr in train_regexps if tr is not None]
    predict_regexps = [re.search(_predict_result_meta_pattern, fn) for fn in filenames]
    predict_ids = [tr.group(1) for tr in predict_regexps if tr is not None]
    return {
        'train': train_ids,
        'predict': predict_ids,
    }
