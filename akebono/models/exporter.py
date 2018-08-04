from akebono.utils import to_pickle
import os
import akebono.settings as settings


def export_sklearn_model(obj, model_name):
    return to_pickle(
        os.path.join(settings.models_dir, '{}.pkl'.format(model_name)),
        obj.value)
