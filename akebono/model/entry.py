import copy
import re
from ._models import (
    WrappedLGBMClassifier,
    WrappedXGBClassifier,
    get_wrapped_sklearn_model,
)
import akebono.settings as settings
from akebono.utils import (
    pathjoin,
    load_object_by_str,
)
from akebono.logging import getLogger


logger = getLogger(__name__)


def get_model(model_config):
    """
    Modelを生成するための関数

    :param model_config: Modelについての設定
    :type model_config: dict
    :return: :class:`WrappedModel` object

    Usage:
        >>> from akebono.model import get_model
        >>> model_config = {
                'name': 'SklearnRandomForestClassifier',
                'init_kwargs': {},
                'fit_kwargs': {},
                'evaluate_kwargs': {
                    'cross_val_iterator': 'KFold@sklearn.model_selection',
                },
                'pos_index': 1,
                'is_rebuild': False,
            }
        >>> model = get_model(model_config)
        >>> model
        <akebono.model._models.sklearn.WrappedSklearnRandomForestClassifier object at 0x1006c0b00>
    """

    if not isinstance(model_config, dict):
        raise TypeError('model_config must be dict.')
    mcc = copy.deepcopy(model_config)
    if 'name' not in mcc:
        raise Exception('name must be set in model_config.')
    model_name = mcc.pop('name')
    is_rebuild = mcc.pop('is_rebuild')
    scenario_tag = mcc.pop('scenario_tag', None)
    train_id = mcc.pop('train_id', None)
    model_type = mcc.pop('model_type', None)

    model = None
    if re.search('^Sklearn.+$', model_name) is not None:
        model = get_wrapped_sklearn_model(model_name)(**mcc)
    elif model_name == 'LGBMClassifier':
        model = WrappedLGBMClassifier(**mcc)
    elif model_name == 'XGBClassifier':
        model = WrappedXGBClassifier(**mcc)
    else:
        model_cls = load_object_by_str(model_name)
        model = model_cls(**mcc)

    if model is None:
        raise Exception('unexpedted.')
    if model_type is not None:
        model.set_model_type(model_type=model_type)

    if is_rebuild:
        if scenario_tag is None or train_id is None:
            raise Exception('invalid state.')
        dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
        mname = 'train_result_model_{}'.format(train_id)
        model.load(dirpath, mname)
    logger.debug('get_model done in {} mode.'.format('predict' if is_rebuild else 'train'))
    return model
