from akebono.utils import snake2camel
from .statelessmodels import (
    Identify,
    SelectColumns,
    ExcludeColumns,
)
import sys


self = sys.modules[__name__]


def get_preprocessor(preprocessor_config):
    """
    Preprocessorを生成するための関数

    :param preprocessor_config: Preprocessorについての設定
    :type preprocessor_config: dict
    :return: :class:`StatelessPreprocessor` or :class:`StatefullPreprocessor`
    """

    name = preprocessor_config.get('name')
    if name is None:
        raise Exception('preprocessor_config.name must be set.')
    ppcls = getattr(self, snake2camel(name), None)
    if ppcls is None:
        raise Exception('unsuppoorted preprocessor .. name: {}'.format(name))
    ppobj = ppcls(**preprocessor_config.get('kwargs', {}))
    return ppobj
