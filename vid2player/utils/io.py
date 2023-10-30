import json
import pickle
from io import BytesIO
from PIL import Image
import base64
import gzip
import numpy as np


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, 'rt', encoding='ascii') as fp:
        return json.load(fp)


def dump_json(obj, fpath, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(obj, fpath):
    with gzip.open(fpath, 'wt', encoding='ascii') as fp:
        json.dump(obj, fp)


def load_pkl(fpath):
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)


def dump_pkl(obj, fpath):
    with open(fpath, 'wb') as fp:
        return pickle.dump(obj, fp)


def decode_png(data):
    if isinstance(data, str):
        data = base64.decodebytes(data.encode())
    else:
        assert isinstance(data, bytes)
    fstream = BytesIO(data)
    im = Image.open(fstream)
    return np.array(im)


def encode_png(data, optimize=True):
    im = Image.fromarray(data)
    fstream = BytesIO()
    im.save(fstream, format='png', optimize=optimize)
    s = base64.encodebytes(fstream.getvalue()).decode()
    return s


def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines


def store_text(fpath, s):
    with open(fpath, 'w') as fp:
        fp.write(s)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)