import datetime
import copy

import akebono.settings as settings
import akebono.operator as operator
from akebono.utils import get_hash
from .base import CommandBase


class Predict(CommandBase):
    def apply_arguments(self, parser):
        parser.add_argument('-c', '--config', default='config')
        parser.add_argument('-t', '--scenario-tag', default='latest')

    def execute(self, namespace):
        for idx, op in enumerate(settings.predict_operations):
            operator.predict(idx, namespace.scenario_tag, **op)
