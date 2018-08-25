from akebono.utils import (
    snake2camel,
    load_object_by_str,
)
from .pipeline import PreprocessorPipeline


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
