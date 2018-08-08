import sys
import re
import copy
from akebono.logging import getLogger
from .parser import AkebonoArgumentParser
from . import models

logger = getLogger(__name__)
parser = AkebonoArgumentParser().build()


def is_help(s):
    return s in ('-h', '--help')


def main(argv = None):
    rawargs = sys.argv[1:]
    if len(rawargs) == 0:
        rawargs = ['--help']

    namespace = parser.parse_args(rawargs)
    subparsers = [(re.sub('subparser__', '', k), v) for k, v in vars(namespace).items() if re.search('^subparser__.+$', k) is not None and v is not None]
    if len(subparsers) > 0:
        longest, method = sorted(subparsers, key=lambda x: x[0], reverse=True)[0]
        cmd_stack = longest.split('_')[1:]
        cmd_stack.append(method)
        cmd = copy.copy(models.tree)
        for k in cmd_stack:
            cmd = cmd[k]
        if isinstance(cmd, dict):
            parser.parse_args(rawargs + ['--help'])
        try:
            cmd.execute_all(namespace)
            logger.debug('{} normally completed.'.format(' '.join(cmd_stack)))
        except Exception as e:
            logger.error(e, exc_info=True)
            sys.exit(1)
    else:
        pass


if __name__ == '__main__':
    main()
