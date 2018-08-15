import copy
import re
from ._models import (
    WrappedLGBMClassifier,
    get_wrapped_sklearn_model,
)
import akebono.settings as settings
from akebono.utils import pathjoin
from akebono.logging import getLogger


logger = getLogger(__name__)


def get_model(model_config):
    if not isinstance(model_config, dict):
        raise TypeError('model_config must be dict.')
    mcc = copy.copy(model_config)
    if 'name' not in mcc:
        raise Exception('name must be set in model_config.')
    model_name = mcc.pop('name')
    is_rebuild = mcc.pop('is_rebuild')
    scenario_tag = mcc.pop('scenario_tag', None)
    train_id = mcc.pop('train_id', None)

    model = None
    if re.search('^Sklearn.+$', model_name) is not None:
        model = get_wrapped_sklearn_model(model_name)(**mcc)
    elif model_name == 'LGBMClassifier':
        model = WrappedLGBMClassifier(**mcc)
    else:
        raise Exception('{} does not found.'.format(model_name))

    if model is None:
        raise Exception('unexpedted.')

    if is_rebuild:
        if scenario_tag is None or train_id is None:
            raise Exception('invalid state.')
        dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
        mname = 'train_result_model_{}'.format(train_id)
        model.load(dirpath, mname)
    logger.debug('get_model done in {} mode.'.format('predict' if is_rebuild else 'train'))
    return model
