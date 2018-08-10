import datetime
import copy

import akebono.settings as settings
import akebono.operator as operator
from akebono.utils import get_random_string
from .base import CommandBase


class Predict(CommandBase):
    def apply_arguments(self, parser):
        parser.add_argument('-c', '--config', default='config')
        parser.add_argument('-t', '--scenario-tag', default='latest')

    def execute(self, namespace):
        for op in settings.predict_operations:
            predict_id = get_random_string(16)
            operator.predict(predict_id, namespace.scenario_tag, **op)
