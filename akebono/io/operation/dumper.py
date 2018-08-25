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
    model = result.pop('model', None)
    preprocessor = result.pop('preprocessor', None)
    model_name = 'train_result_model_{}'.format(train_id)
    result_name = 'train_result_meta_{}'.format(train_id)

    dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
    to_pickle(pathjoin(dirpath, '{}.pkl'.format(result_name)), result) # dump result
    if model is not None:
        model.dump(dirpath, model_name)
    if preprocessor is not None:
        preprocessor.dump_with_operation_rule(dirpath, train_id)


def dump_predicted_result(predict_id, scenario_tag, dumper_config, df, meta):
    dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
    fname_meta = 'predict_result_meta_{}'.format(predict_id)
    fname = 'predict_result_{}'.format(predict_id)
    dump_result_format = dumper_config['name']
    if dump_result_format == 'csv':
        pd_to_csv(df, pathjoin(dirpath, fname + '.csv'), index=False)
    elif dump_result_format == 'pickle':
        to_pickle(pathjoin(dirpath, fname + '.pkl'), df)
    elif dump_result_format == 'bigquery':
        import pandas_gbq
        destination_table = dumper_config.get('destination_table')
        if destination_table is None:
            raise ValueError('destination_table must be set for bigquery dumper.')
        if dumper_config.get('add_predict_id_enabled', True):
            destination_table += ('_' + predict_id)
        pandas_gbq.to_gbq(df, destination_table, **dumper_config.get('kwargs', {}))
    else:
        raise Exception('invalid format')
    to_pickle(pathjoin(dirpath, fname_meta + '.pkl'), meta)
