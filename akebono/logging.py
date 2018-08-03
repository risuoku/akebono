import logging
import sys
import os


_format = '[%(levelname)s] %(asctime)s %(message)s %(name)s'


def getLogger(name = None):
    logger = logging.getLogger(name)

    handler = logging.StreamHandler(stream = sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_format))

    logger.addHandler(handler)
    logger.propagate = False
    if os.environ.get('DEBUG') is None:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    return logger
