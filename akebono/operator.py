from akebono.logging import getLogger
from akebono.io.operation.dumper import (
    dump_train_result,
    dump_predicted_result,
)
from akebono.io.operation.loader import get_train_result 
from akebono.dataset import get_dataset
from akebono.model import get_model
from akebono.preprocessor import (
    get_preprocessor,
)
from akebono.utils import (
    load_object_by_str,
    pathjoin,
)
from akebono.exceptions import EmptyDatasetError
import akebono.settings as settings
import pandas as pd
import gc


logger = getLogger(__name__)


def train(train_id, scenario_tag,
    dataset_config=None,
    model_config=None,
    preprocessor_config=None,
    formatter_config_for_predictor=None,
    formatter_config_for_target=None,
    evaluate_enabled=False,
    fit_model_enabled=False,
    dump_result_enabled=False
    ):
        """
        モデルの訓練を実行する関数

        :param train_id: シナリオ中における訓練実行の識別子
        :type train_id: str
        :param scenario_tag: シナリオに付与されるタグ
        :type scenario_tag: str
        :param dataset_config: Datasetの設定。:class:`akebono.dataset.get_dataset` の引数。
        :type dataset_config: dict
        :param model_config: Modelの設定。:class:`akebono.model.get_model` の引数。
        :type model_config: dict
        :param preprocessor_config: Preprocessorの設定。:class:`akebono.preprocessor.get_preprocessor` の引数。
        :type preprocessor_config: dict
        :param formatter_config_for_predictor: 特徴用Formatterの設定。
        :type formatter_config_for_predictor: dict
        :param formatter_config_for_target: 目標用Formatterの設定。
        :type formatter_config_for_target: dict
        :param evaluate_enabled: モデルの評価を実行するかのフラグ
        :type evaluate_enabled: bool
        :param fit_model_enabled: モデルの訓練を実行するかのフラグ
        :type fit_model_enabled: bool
        :param dump_result_enabled: モデル、評価結果の永続化を実行するかのフラグ
        :type dump_result_enabled: bool
        """
        if model_config is None:
            raise ValueError('model_config must be set.')
        if dataset_config is None:
            raise ValueError('dataset_config must be set.')
        if preprocessor_config is None:
            preprocessor_config = {
                'name': 'identify',
                'kwargs': {},
            }
        if formatter_config_for_predictor is None:
            formatter_config_for_predictor = {
                'name': 'get_values@akebono.formatter',
            }
        if formatter_config_for_target is None:
            formatter_config_for_target = {
                'name': 'get_values@akebono.formatter',
            }

        ret = {
            'type': 'train',
            'id': train_id,
            'dataset_config': dataset_config,
            'model_config': model_config,
            'preprocessor_config': preprocessor_config,
            'formatter_config_for_predictor': formatter_config_for_predictor,
            'formatter_config_for_target': formatter_config_for_target,
            'evaluate_enabled': evaluate_enabled,
            'fit_model_enabled': fit_model_enabled,
            'dump_result_enabled': dump_result_enabled
        }

        dataset = get_dataset(dataset_config)
        
        preprocessor = get_preprocessor(preprocessor_config)
        preprocessor.set_operation_mode('train')
        logger.debug('load dataset start.')
        X, y = dataset.get_predictor_target()

        if X.index.size == 0:
            raise EmptyDatasetError('empty record')

        logger.debug('load dataset done.')
        
        model_config['is_rebuild'] = False
        model = get_model(model_config)

        format_func_for_predictor = load_object_by_str(formatter_config_for_predictor['name'])
        format_func_for_target = load_object_by_str(formatter_config_for_target['name'])

        if evaluate_enabled:
            logger.debug('evaluate start.')
            rep = model.evaluate(X, y, preprocessor, format_func_for_predictor, format_func_for_target)
            gc.collect()
            logger.debug('evaluate done.')
            ret['evaluate'] = rep
        if fit_model_enabled:
            logger.debug('fit start.')
            fX_p, _ = preprocessor.process(X, None)
            fX = format_func_for_predictor(fX_p)
            model.fit(fX, format_func_for_target(y))
            gc.collect()
            logger.debug('fit done.')
            ret['model'] = model
        if dump_result_enabled:
            logger.debug('dump_train_result start.')
            ret['preprocessor'] = preprocessor
            dump_train_result(train_id, scenario_tag, ret)
            logger.debug('dump_train_result done.')
        
        return ret


def predict(predict_id, scenario_tag,
    method_type='predict',
    dataset_config=None,
    train_id='0',
    dump_result_enabled=False,
    append_evacuated_columns_enabled=False,
    dumper_config={},
    result_target_columns='all',
    result_predict_column='predicted'
    ):
        """
        予測を実行する関数

        :param predict_id: シナリオ中における予測実行の識別子
        :type predict_id: str
        :param scenario_tag: シナリオに付与されるタグ
        :type scenario_tag: str
        :param method_type: 予測のタイプ。設定可能なタイプは `predict` or `predict_proba`
        :type method_type: str
        :param dataset_config: Datasetの設定。:class:`akebono.dataset.get_dataset` の引数。
        :type dataset_config: dict
        :param train_id: 予測で使うモデルのtrain_id
        :type train_id: str
        :param dump_result_enabled: 予測結果の永続化を実行するかのフラグ
        :type dump_result_enabled: bool
        :param append_evacuated_columns_enabled: Dataset中で退避したカラムをpredictの結果に加えるかを決めるフラグ
        :type append_evacuated_columns_enabled: bool
        :param dumper_config: 予測結果の設定。
        :type dumper_config: dict
        :param result_target_columns: 予測結果に含めるべき説明変数のカラム名のリスト。全ての場合は'all'とする
        :type result_target_columns: str or list(str)
        :param result_predict_column: 予測結果が格納されるカラム名
        :type result_predict_column: str
        """
        if dataset_config is None:
            raise ValueError('dataset_config must be set.')

        if dump_result_enabled and 'name' not in dumper_config:
            raise ValueError('`name` key must be contained in dumper_config.')

        train_id = str(train_id)
        model_config = {}
        ret = {
            'type': 'predict',
            'method_type': method_type,
            'dataset_config': dataset_config,
            'train_id': train_id,
            'dump_result_enabled': dump_result_enabled,
            'dumper_config': dumper_config,
            'result_target_columns': result_target_columns,
            'result_predict_column': result_predict_column,
        }

        model_config['train_id'] = train_id
        model_config['scenario_tag'] = scenario_tag

        tr = get_train_result(scenario_tag=scenario_tag, train_id=train_id)
        if tr is None:
            raise Exception('target result not found.')
        ret['train_result'] = tr

        dataset_config['target_column'] = None # target_columnがNoneだと、predict用のDatasetが返ってくる
        dataset = get_dataset(dataset_config)
        preprocessor = get_preprocessor(tr['preprocessor_config'])
        preprocessor.set_operation_mode('predict')
        dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
        preprocessor.load_with_operation_rule(dirpath, train_id)

        X = dataset.get_predictor()

        if X.index.size == 0:
            raise EmptyDatasetError('empty record')

        fX, _ = preprocessor.process(X, None)
        gc.collect()
        
        model_config.update(tr['model_config'])
        model_config['is_rebuild'] = True
        model = get_model(model_config)

        predict_func = getattr(model, method_type, None)
        if predict_func is None:
            raise Exception('{} is not defined.'.format(method_type))
        rawresult = predict_func(fX)
        gc.collect()
        predict_result = fX.copy()
        if not result_target_columns == 'all':
            if not isinstance(result_target_columns, list):
                raise TypeError('result_target_columns must be list.')
            predict_result = predict_result[result_target_columns]

        # len(rawresult.shape) > 1でもpredict_resultのdfに格納できるようにするためlistにしている
        # そもそもpredict_resultを１つのdfにするべきなのかは考え直しても良いと思う
        predict_result.loc[:,result_predict_column] = list(rawresult)

        if append_evacuated_columns_enabled:
            predict_result = pd.concat([dataset.get_evacuated(), predict_result], axis=1)

        if dump_result_enabled:
            logger.debug('dump_predicted_result start.')
            dump_predicted_result(predict_id, scenario_tag, dumper_config, predict_result, ret)
            logger.debug('dump_predicted_result done.')

        ret['predict_result'] = predict_result
        return ret
