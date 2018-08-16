from akebono.utils import (
    to_pickle,
    pd_to_csv,
    remove_directory,
    rename_directory,
    isdir,
    pathjoin,
)
import os
import akebono.settings as settings


def dump_sklearn_model(obj, dirpath, model_name):
    return to_pickle(
        pathjoin(dirpath, '{}.pkl'.format(model_name)),
        obj.value)


def dump_train_result(train_id, scenario_tag, result):
    model = result.get('model')
    preprocessor = result.get('preprocessor')
    if model is not None:
        result.pop('model')
    if preprocessor is not None:
        result.pop('preprocessor')
    model_name = 'train_result_model_{}'.format(train_id)
    result_name = 'train_result_meta_{}'.format(train_id)
    tag_list = ['latest']
    if scenario_tag is not None:
        tag_list.append(scenario_tag)
    for tag in tag_list:
        dirpath = pathjoin(settings.operation_results_dir, tag)
        to_pickle(pathjoin(dirpath, '{}.pkl'.format(result_name)), result) # dump result
        if model is not None:
            model.dump(dirpath, model_name)
        if preprocessor is not None:
            preprocessor_name = 'train_result_preprocessor_{}_{}'.format(preprocessor.__class__.__name__, train_id)
            preprocessor.dump(dirpath, preprocessor_name)


def dump_predicted_result(predict_id, scenario_tag, dump_result_format, df, meta):
    dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
    fname_meta = 'predict_result_meta_{}'.format(predict_id)
    fname = 'predict_result_{}'.format(predict_id)
    if dump_result_format == 'csv':
        pd_to_csv(df, pathjoin(dirpath, fname + '.csv'), index=False)
    elif dump_result_format == 'pickle':
        to_pickle(pathjoin(dirpath, fname + '.pkl'), df)
    else:
        raise Exception('invalid format')
    to_pickle(pathjoin(dirpath, fname_meta + '.pkl'), meta)
