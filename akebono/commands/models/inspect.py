import datetime
import copy

from .base import CommandBase
from akebono.inspector import get_scenario_summary


def _get_fixed_length_str(s, length):
    if not isinstance(s, str):
        raise TypeError('invalid type')
    l = length - len(s)
    if l < 0:
        raise Exception('invalid length')
    return s + (' ' * l)


class Inspect(CommandBase):
    def apply_arguments(self, parser):
        parser.add_argument('--config', default='config')
        parser.add_argument('-t', '--scenario-tag', default='latest')
        parser.add_argument('-k', '--performance-sort-key', default=None)

    def execute(self, namespace):
        ss = get_scenario_summary(namespace.scenario_tag, namespace.performance_sort_key)

        print('=== performance summary .. scenario_tag: {} ==='.format(namespace.scenario_tag))
        for idx, row in ss.iterrows():
            print('')
            print('------------------------------')
            op_index = row['_akebono_op_index']
            print('operation_index: {}'.format(int(op_index)))
            print('')
            attrs = list(row.index.copy())
            attrs.remove('_akebono_op_index')
            alen = max([len(a) for a in attrs] + [8])
            v1 = ' '.join([_get_fixed_length_str(a, alen) for a in attrs])
            v2 = ' '.join([_get_fixed_length_str('{:.5f}'.format(row[a]), alen) for a in attrs])
            print(v1)
            print(v2)
        print('')
