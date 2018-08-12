import akebono.settings as settings
from .base import CommandBase


class Init(CommandBase):
    def apply_arguments(self, parser):
        parser.add_argument('-c', '--config', default='config')

    def execute(self, namespace):
        settings.init()
