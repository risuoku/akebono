from akebono.io.operation.loader import get_train_result
from akebono.model import get_model
from akebono.logging import getLogger


logger = getLogger(__name__)


def get_trained_model(scenario_tag, train_id, train_result=None):
    train_id = str(train_id)

    model_config = {
        'train_id': train_id,
        'scenario_tag': scenario_tag,
    }

    if train_result is None:
        logger.debug('train_result is None .. load from scenario_tag: {}, train_id: {}'.format(scenario_tag, train_id))
        train_result = get_train_result(scenario_tag=scenario_tag, train_id=train_id)
        if train_result is None:
            raise Exception('target result not found.')

    model_config.update(train_result['model_config'])
    model_config['is_rebuild'] = True
    model = get_model(model_config)

    return model, model_config, train_result