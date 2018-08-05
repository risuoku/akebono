import argparse
import datetime
import importlib
import copy
import akebono.settings as settings
import akebono.operator as operator
from akebono.utils import get_hash


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config')
    parser.add_argument('--scenario-tag', default=None)
    parser.add_argument('--auto-scenario-tag-enabled', action='store_true', default=False)
    ns = parser.parse_args()
    mod_config = importlib.import_module(ns.config)
    settings.load(mod_config)

    # get scenario_tag
    scenario_tag = None
    if ns.auto_scenario_tag_enabled:
        scenario_tag = get_hash(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
    if ns.scenario_tag is not None:
        scenario_tag = ns.scenario_tag
    
    for idx, op in enumerate(settings.operations):
        opc = copy.copy(op)
        if 'kind' in opc:
            op_kind = opc.pop('kind')
            op_func = getattr(operator, op_kind, None)
            if op_func is not None:
                op_func(idx, scenario_tag, **opc)
