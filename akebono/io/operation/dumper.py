from akebono.utils import (
    to_pickle,
    remove_directory,
    rename_directory,
    isdir,
)
import os
import akebono.settings as settings


def dump_sklearn_model(obj, dirpath, model_name):
    return to_pickle(
        os.path.join(dirpath, '{}.pkl'.format(model_name)),
        obj.value)


def dump_operation_result(operation_index, scenario_tag, result):
    model = result.get('fit_model')
    if model is not None:
        result.pop('fit_model')
    model_name = 'model_{}'.format(operation_index)
    result_name = 'result_{}'.format(operation_index)
    tag_list = ['latest']
    if scenario_tag is not None:
        tag_list.append(scenario_tag)
    for tag in tag_list:
        dirpath = os.path.join(settings.operation_results_dir, tag)
        tmpdirpath = os.path.join(settings.operation_results_dir, 'tmp_' + tag)
        if isdir(tmpdirpath):
            raise Exception('tmpdirpath exists .. please rename or remove {} before save.'.format(tmpdirpath))
        if isdir(dirpath):
            rename_directory(dirpath, tmpdirpath)
        if settings.storage_type == 'local':
            os.makedirs(dirpath, exist_ok=True)
        to_pickle(os.path.join(dirpath, '{}.pkl'.format(result_name)), result) # dump result
        if model is not None:
            model.dump(dirpath, model_name)
        if isdir(tmpdirpath):
            remove_directory(tmpdirpath)
