import os
import hashlib
import subprocess
import shlex
import pickle
import pandas as pd
import importlib
import re

from akebono.logging import getLogger

from . import settings


logger = getLogger(__name__)


def _get_gcs_bucket(bucket_name):
    from google.cloud import storage as gstorage
    _c = gstorage.Client()
    return _c.get_bucket(bucket_name)


def to_pickle(filepath, obj):
    if settings.storage_type == 'local':
        with open(filepath, 'wb') as fp:
            pickle.dump(obj, fp, protocol=4)
    elif settings.storage_type == 'gcs':
        bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
        obj_pkl = pickle.dumps(obj, protocol=4)
        bkt.blob(filepath, chunk_size=1048576000).upload_from_string(obj_pkl, content_type='application/octet-stream')
    else:
        raise ValueError('invalid storage_type')


def from_pickle(filepath):
    if settings.storage_type == 'local':
        obj = None
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as fp:
                obj = pickle.load(fp)
            if obj is None:
                raise ValueError('`obj` must not be None.')
        return obj
    elif settings.storage_type == 'gcs':
        bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
        obj = bkt.get_blob(filepath)
        if obj is None:
            return None
        return pickle.loads(obj.download_as_string())
    else:
        raise ValueError('invalid storage_type')


def cache_located_at(filepath):
    def _func(f):
        def __func(*args, **kwargs):
            obj = from_pickle(filepath)
            if obj is not None:
                logger.debug('load from cache .. {}'.format(filepath))
                return obj

            result =  f(*args, **kwargs)

            to_pickle(filepath, result)
            return result
        return __func
    return _func


def get_hash(s):
    if isinstance(s, str):
        s = s.encode('utf-8')
    return hashlib.sha256(s).hexdigest()


class Param:
    def __init__(self, d):
        if not isinstance(d, dict):
            raise TypeError('d must be dict')
        self._d = d

    def __getitem__(self, key):
        return self._d[key]

    def get_param_expression(self, ignore_keys = []):
        s_items = sorted(self._d.items(), key=lambda x: x[0])
        h_str = '__'.join(['{}_{}'.format(k, v) for k, v in s_items if not k in ignore_keys])
        return h_str

    def get_hashed_id(self):
        return get_hash(self.get_param_expression())
    
    @property
    def value(self):
        return self._d

    def __repr__(self):
        return self.__class__.__name__ + '_' + self.get_hashed_id()

    def __str__(self):
        return self.__repr__()

    
class ConsoleCommand:
    def __init__(self, s):
        self._raw_string = s
        self._stdout = None
        self._stderr = None
    
    def execute(self):
        pargs = shlex.split(self._raw_string)
        with subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            self._stdout, self._stderr = proc.communicate()
        return self
    
    def get_stdout(self):
        return [s for s in self._stdout.decode('utf8').split('\n') if not s == '']


def get_shellcmd_result(s):
    return ConsoleCommand(s).execute().get_stdout()


def load_object_by_str(s, type_is=None):
    r = re.search('(\S+)@(\S+)$', s)
    if r is None:
        raise Exception('invlid format')
    fs = r.group(1)
    mods = r.group(2)
    mod = importlib.import_module(mods)
    r = getattr(mod, fs)
    if type_is is not None and not isinstance(r, type_is):
        raise TypeError('invalid type')
    return r


def get_label_by_object(obj):
    return '{}@{}'.format(obj.__name__, obj.__module__)