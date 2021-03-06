import os
import gc

import akebono.settings as settings
import akebono.operator as operator
from akebono.utils import (
    pathjoin,
    get_random_string,
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
            scenario_tag = get_random_string(16)
        if namespace.scenario_tag is not None:
            scenario_tag = namespace.scenario_tag
        
        scenario_tag = scenario_tag or 'default'
        try:
            # prepare environment
            dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
            tmpdirpath = pathjoin(settings.operation_results_dir, 'tmp_' + scenario_tag)
            if isdir(tmpdirpath):
                raise Exception('tmpdirpath exists .. please rename or remove {} before save.'.format(tmpdirpath))
            if isdir(dirpath):
                rename_directory(dirpath, tmpdirpath)
                logger.info('old scenario_dir {} is renamed to {}.'.format(dirpath, tmpdirpath))
            if settings.storage_type == 'local':
                os.makedirs(dirpath, exist_ok=True)

            # train
            logger.info('===== train start .. config: {} ====='.format(namespace.config))
            for idx, op in enumerate(settings.get_train_configs()):
                logger.info('training .. id: {}'.format(idx))
                operator.train(str(idx), scenario_tag, **op)
                gc.collect() # free memory
            logger.info('===== train done =====')
            
            # cleanup
            tmpdirpath = pathjoin(settings.operation_results_dir, 'tmp_' + scenario_tag)
            if isdir(tmpdirpath):
                remove_directory(tmpdirpath)
                logger.info('old scenario_dir {} is removed.'.format(tmpdirpath))
        
        except Exception as e:
            # rollback
            dirpath = pathjoin(settings.operation_results_dir, scenario_tag)
            tmpdirpath = pathjoin(settings.operation_results_dir, 'tmp_' + scenario_tag)
            if isdir(tmpdirpath):
                if isdir(dirpath):
                    remove_directory(dirpath)
                    logger.info('operation failed .. this scenario state is totally removed')
                rename_directory(tmpdirpath, dirpath)
                logger.info('operation failed .. old scenario state is rollbacked.')
            raise e
