# Where every step reads and writes, shared by the scripts and experiments.py

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# outputs can go elsewhere, e.g. a scratch folder for a quick test of the whole pipeline
ROOT = os.environ.get('DMRI_PNET_ROOT', REPO)


def synth(name):
    return os.path.join(ROOT, 'data/synth', name + '.npz')


def real(path):
    return os.path.join(REPO, 'data/real', path)


def run(name):
    return os.path.join(ROOT, 'runs', name)


def preds(name, snr, method):
    return os.path.join(ROOT, 'data/preds', f'{name}_snr{snr}', method + '.npy')


def maps(subject, name):
    # in the subject's folder: a method's map prefix (<method>_{ndi,odi,fwf}.nii.gz), or any file
    return os.path.join(ROOT, 'data/maps', subject, name)


def results(*parts):
    return os.path.join(ROOT, 'results', *parts)


def ensure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
