import os
import sys
import hashlib
import subprocess
import shlex
import shutil
import pickle
import pandas as pd
import importlib
import re

from akebono.logging import getLogger

from . import settings


logger = getLogger(__name__)

self = sys.modules[__name__]
_bkt = None


def _get_gcs_bucket(bucket_name):
    if self._bkt is None:
        from google.cloud import storage as gstorage
        _c = gstorage.Client()
        self._bkt = _c.get_bucket(bucket_name)
    return self._bkt


def pd_to_csv(df, path, **kwargs):
    if settings.storage_type == 'local':
        df.to_csv(path, **kwargs)
    elif settings.storage_type == 'gcs':
        try:
            df.to_csv(path, **kwargs)
            with open(path, 'r') as f:
                bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
                bkt.blob(filepath, chunk_size=1048576000).upload_from_file(f, content_type='text/csv')
        finally:
            if os.path.isfile(path):
                os.remove(path)
    else:
        raise ValueError('invalid storage_type')


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


def remove_directory(dirpath):
    if settings.storage_type == 'local':
        shutil.rmtree(dirpath)
    elif settings.storage_type == 'gcs':
        bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
        blob_list = [blob for blob in bkt.list_blobs() if re.search(dirpath, blob.name) is not None]
        bkt.delete_blobs(blob_list)
    else:
        raise ValueError('invalid storage_type')


def rename_directory(src, dst):
    if settings.storage_type == 'local':
        shutil.move(src, dst)
    elif settings.storage_type == 'gcs':
        bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
        blob_list = [blob for blob in bkt.list_blobs() if re.search(src, blob.name) is not None]
        for blob in blob_list:
            bkt.rename_blob(blob, re.sub(src, dst, blob.name))
    else:
        raise ValueError('invalid storage_type')


def isdir(dirpath):
    if settings.storage_type == 'local':
        return os.path.isdir(dirpath)
    elif settings.storage_type == 'gcs':
        bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
        target_blob = bkt.get_blob(dirpath)
        return target_blob is not None
    else:
        raise ValueError('invalid storage_type')


def list_directory(dirpath):
    if settings.storage_type == 'local':
        return [os.path.join(dirpath, f) for f in os.listdir(dirpath)]
    elif settings.storage_type == 'gcs':
        bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
        blob_list = [blob for blob in bkt.list_blobs() if re.search(dirpath, blob.name) is not None]
        return [blob.name for blob in blob_list if not blob.name == dirpath]
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
