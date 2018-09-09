import os
import sys
import hashlib
import shutil
import pickle
import pandas as pd
import importlib
import re
import io

from akebono.logging import getLogger

from . import settings
pathjoin = settings.pathjoin


logger = getLogger(__name__)

self = sys.modules[__name__]
_bkt = None


def _get_gcs_bucket(bucket_name):
    if self._bkt is None:
        from google.cloud import storage as gstorage
        _c = gstorage.Client()
        self._bkt = _c.get_bucket(bucket_name)
    return self._bkt


def try_decode_bytes_and_get(bt):
    if not isinstance(bt, bytes):
        raise TypeError('`bt` must be bytes type.')
    for charcode in ['utf8', 'sjis', 'euc-jp']:
        try:
            decoded = bt.decode(charcode)
            return decoded
        except UnicodeDecodeError:
            # ignore
            pass
    return None


def pd_to_csv(df, path, **kwargs):
    if settings.storage_type == 'local':
        df.to_csv(path, **kwargs)
    elif settings.storage_type == 'gcs':
        buf = df.to_csv(None, encoding='utf-8', **kwargs)
        bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
        bkt.blob(path, chunk_size=1048576000).upload_from_string(buf, content_type='text/csv')
    else:
        raise ValueError('invalid storage_type')


def pd_read_csv(path, **kwargs):
    if settings.storage_type == 'local':
        return pd.read_csv(path, **kwargs)
    elif settings.storage_type == 'gcs':
        bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
        obj = bkt.get_blob(path)
        if obj is None:
            return None
        s = try_decode_bytes_and_get(obj.download_as_string())
        if s is None:
            raise Exception('decode bytes failed.')
        return pd.read_csv(io.StringIO(s), **kwargs)
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


def list_directory(dirpath, mode='path'):
    if settings.storage_type == 'local':
        filenames = os.listdir(dirpath)
        if mode == 'path':
            return [os.path.join(dirpath, f) for f in filenames]
        elif mode == 'filename':
            return filenames
        else:
            raise ValueError('invalid mode.')
    elif settings.storage_type == 'gcs':
        bkt = _get_gcs_bucket(settings.storage_option['bucket_name'])
        blob_list = [blob for blob in bkt.list_blobs() if re.search(dirpath, blob.name) is not None]
        pathlist = [blob.name for blob in blob_list if not blob.name == dirpath]
        if mode == 'path':
            return pathlist
        elif mode == 'filename':
            r1 = [re.sub('^' + dirpath, '', p) for p in pathlist]
            return [f[1:] if f[0] == '/' else f for f in r1]
        else:
            raise ValueError('invalid mode.')
    else:
        raise ValueError('invalid storage_type')


def isdir(dirpath):
    if settings.storage_type == 'local':
        return os.path.isdir(dirpath)
    elif settings.storage_type == 'gcs':
        flist = list_directory(dirpath, mode='path')
        return len(flist) > 0
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

    def get_hashed_id(self, length=None):
        r = get_hash(self.get_param_expression())
        if length is not None:
            r = r[:length]
        return r
    
    @property
    def value(self):
        return self._d

    def __repr__(self):
        return self.__class__.__name__ + '_' + self.get_hashed_id()

    def __str__(self):
        return self.__repr__()


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


# Use the system PRNG if possible
try:
    import random
    random = random.SystemRandom()
    using_sysrandom = True
except NotImplementedError:
    import warnings
    warnings.warn('A secure pseudo-random number generator is not available '
                  'on your system. Falling back to Mersenne Twister.')
    using_sysrandom = False

DEFAULT_ALLOWED_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
def get_random_string(length=12, allowed_chars=DEFAULT_ALLOWED_CHARS):
    """
    implementation inspired by Django
    """
    if not using_sysrandom:
        import hashlib
        import time
        random.seed(
                    hashlib.sha256(
                        ("%s%s%s" % (
                            random.getstate(),
                            time.time(),
                            'aaaaa')).encode('utf-8')
                    ).digest())
    return ''.join(random.choice(allowed_chars) for i in range(length))


def snake2camel(s):
    return ''.join([a.capitalize() for a in s.split('_')]) # snake case -> camel case


def camel2snake(s):
    return re.sub("([A-Z])",lambda x:"_" + x.group(1).lower(), s)[1:] # camel case -> snake case
