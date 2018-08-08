import argparse
import sys

from . import models


class AkebonoArgumentParser(argparse.ArgumentParser):
    def build(self):
        self._apply_config(self, models.tree, 'root')
        return self

    def _apply_config(self, parser, config, current_name):
        if isinstance(config, dict):
            subparsers = parser.add_subparsers(dest='subparser__' + current_name)
            for k, v in config.items():
                p2 = subparsers.add_parser(k)
                self._apply_config(p2, v, '{}_{}'.format(current_name, k))
        else:
            config.apply_arguments(parser)
