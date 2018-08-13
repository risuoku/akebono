import datetime
import os

import akebono.settings as settings
import akebono.operator as operator
from akebono.utils import (
    pathjoin,
    get_hash,
    isdir,
    rename_directory,
    remove_directory,
)
from akebono.logging import getLogger
from .base import CommandBase


logger = getLogger(__name__)


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
        
        scenario_tags = ['latest'] + ([] if scenario_tag is None else [scenario_tag])
        try:
            # prepare environment
            for tag in scenario_tags:
                dirpath = pathjoin(settings.operation_results_dir, tag)
                tmpdirpath = pathjoin(settings.operation_results_dir, 'tmp_' + tag)
                if isdir(tmpdirpath):
                    raise Exception('tmpdirpath exists .. please rename or remove {} before save.'.format(tmpdirpath))
                if isdir(dirpath):
                    rename_directory(dirpath, tmpdirpath)
                    logger.info('old scenario_dir {} is renamed to {}.'.format(dirpath, tmpdirpath))
                if settings.storage_type == 'local':
                    os.makedirs(dirpath, exist_ok=True)

            # train
            for idx, op in enumerate(settings.train_operations):
                operator.train(str(idx), scenario_tag, **op)
            
            # cleanup
            for tag in scenario_tags:
                tmpdirpath = pathjoin(settings.operation_results_dir, 'tmp_' + tag)
                if isdir(tmpdirpath):
                    remove_directory(tmpdirpath)
                    logger.info('old scenario_dir {} is removed.'.format(tmpdirpath))
        
        except Exception as e:
            # rollback
            for tag in scenario_tags:
                dirpath = pathjoin(settings.operation_results_dir, tag)
                tmpdirpath = pathjoin(settings.operation_results_dir, 'tmp_' + tag)
                if isdir(tmpdirpath):
                    if isdir(dirpath):
                        remove_directory(dirpath)
                        logger.info('operation failed .. this scenario state is totally removed')
                    rename_directory(tmpdirpath, dirpath)
                    logger.info('operation failed .. old scenario state is rollbacked.')
            raise e
