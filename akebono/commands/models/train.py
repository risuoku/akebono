import datetime
import copy

import akebono.settings as settings
import akebono.operator as operator
from akebono.utils import get_hash
from .base import CommandBase


class Train(CommandBase):
    def apply_arguments(self, parser):
        parser.add_argument('-c', '--config', default='config')
        parser.add_argument('-t', '--scenario-tag', default=None)
        parser.add_argument('--auto-scenario-tag-enabled', action='store_true', default=False)

    def execute(self, namespace):
        scenario_tag = None
        if namespace.auto_scenario_tag_enabled:
            scenario_tag = get_hash(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
        if namespace.scenario_tag is not None:
            scenario_tag = namespace.scenario_tag
        
        for idx, op in enumerate(settings.train_operations):
            operator.train(idx, scenario_tag, **op)
