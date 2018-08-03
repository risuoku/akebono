import argparse
import importlib
import copy
import akebono.settings as settings
import akebono.operator as operator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config')
    ns = parser.parse_args()
    mod_config = importlib.import_module(ns.config)
    settings.load(mod_config)
    
    for op in settings.operations:
        opc = copy.copy(op)
        if 'kind' in opc:
            op_kind = opc.pop('kind')
            op_func = getattr(operator, op_kind, None)
            if op_func is not None:
                op_func(**opc)
