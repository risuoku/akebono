from akebono.utils import (
    snake2camel,
    load_object_by_str,
    pathjoin,
)
from .pipeline import PreprocessorPipeline
import akebono.settings as settings
from akebono.io.operation.loader import get_train_result
from akebono.logging import getLogger


logger = getLogger(__name__)


_preprocessor_name_alias = {
    'Identify': 'Identify@akebono.preprocessor.statelessmodels',
    'SelectColumns': 'SelectColumns@akebono.preprocessor.statelessmodels',
    'ExcludeColumns': 'ExcludeColumns@akebono.preprocessor.statelessmodels',

    'ApplyStandardScaler': 'ApplyStandardScaler@akebono.preprocessor.statefulmodels',
    'ApplyPca': 'ApplyPca@akebono.preprocessor.statefulmodels',
}


def _get_preprocessor_strict(config):
    name = config.get('name')
    if name is None:
        raise Exception('preprocessor_config.name must be set.')
    cameledname = snake2camel(name)
    ppcls = load_object_by_str(_preprocessor_name_alias.get(cameledname, name))
    return ppcls(**config.get('kwargs', {}))


def get_preprocessor(preprocessor_config, pipeline_enabled=True):
    """
    Preprocessorを生成するための関数

    :param preprocessor_config: Preprocessorについての設定
    :type preprocessor_config: list or dict
    :param pipeline_enabled: PreprocessorPipelineを有効にするかどうかのフラグ
    :type pipiline_enabled: bool
    :return: :class:`PreprocessorPipeline` or :class:`StatelessPreprocessor` or :class:`StatefulPreprocessor`
    """

    if pipeline_enabled:
        ppcl = []
        if isinstance(preprocessor_config, dict):
            ppcl = [preprocessor_config]
        elif isinstance(preprocessor_config, list):
            ppcl = preprocessor_config
        else:
            raise TypeError('preprocessor_config must be list or dict.')

        pp_list = []
        for ppc in ppcl:
            pp_list.append(_get_preprocessor_strict(ppc))
        return PreprocessorPipeline(pp_list)
    else:
        if not isinstance(preprocessor_config, dict):
            raise TypeError('preprocessor_config must be dict if pipeline_enabled is False')
        return _get_preprocessor_strict(preprocessor_config)


def get_preprocessor_for_prediction(scenario_tag, train_id, train_result=None, dirpath=None):
    if train_result is None:
        logger.debug('train_result is None .. load from scenario_tag: {}, train_id: {}'.format(scenario_tag, train_id))
        train_result = get_train_result(scenario_tag=scenario_tag, train_id=train_id)
    preprocessor = get_preprocessor(train_result['preprocessor_config'])
    preprocessor.set_operation_mode('predict')
    if dirpath is None:
        dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
    preprocessor.load_with_operation_rule(dirpath, train_id)
    return preprocessor
