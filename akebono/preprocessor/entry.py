from akebono.utils import snake2camel
from .statelessmodels import (
    Identify,
    SelectColumns,
    ExcludeColumns,
)
from .statefulmodels import (
    ApplyStandardScaler,
)
from .pipeline import PreprocessorPipeline
import sys


self = sys.modules[__name__]


def _get_preprocessor_strict(config):
    name = config.get('name')
    if name is None:
        raise Exception('preprocessor_config.name must be set.')
    ppcls = getattr(self, snake2camel(name), None)
    if ppcls is None:
        raise Exception('unsuppoorted preprocessor .. name: {}'.format(name))
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
