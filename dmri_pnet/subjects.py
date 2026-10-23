# Real datasets under data/real. 'ref': NODDI maps from the original pipeline, as [ndi, odi, fwf].

import os

from . import data, paths


def motif(folder, dwi, maps, mask=None):
    return dict(dwi=f'{folder}/{dwi}.nii.gz', bval=f'{folder}/{dwi}.bval', bvec=f'{folder}/{dwi}.bvec',
                ref=[f'{folder}/{maps}_{p}.nii.gz' for p in ('NDI', 'ODI', 'FWF')],
                # no brain mask shipped for sub-05: the reference ODI map is non-zero exactly inside the brain
                mask=mask or f'{folder}/{maps}_ODI.nii.gz')


SUBJECTS = {
    'sub-01': motif('philippe/sub-01', 'sub-01_desc-MSEddyCorrected_dwi', 'sub-01_model-NODDI',
                    'philippe/sub-01/sub-01_mask.nii.gz'),
    'sub-05r1': motif('philippe/sub-05r1', 'sub-05_desc-MSEddyCorrected_run-1_dwi', 'sub-05_run-1_model-NODDI'),
    'sub-05r2fewdir': motif('philippe/sub-05r2fewdir', 'sub-05_desc-MSEddyCorrected_run-2_dwi',
                            'sub-05_run-2_model-NODDI'),
    'sub-05r2manydir': motif('philippe/sub-05r2manydir', 'sub-05_desc-MSEddyCorrected_run-2_dwi',
                             'sub-05_run-2_model-NODDI'),
    'antoine': dict(dwi='antoine/dwi_corr.nii.gz', bval='antoine/bvals.bval', bvec='antoine/bvecs.bvec',
                    mask='antoine/brain_mask.nii.gz',
                    ref=[f'antoine/noddi_{p}.nii.gz' for p in ('ficvf', 'odi', 'fiso')]),
}


def files(name):
    return {k: [paths.real(p) for p in v] if k == 'ref' else paths.real(v) for k, v in SUBJECTS[name].items()}


def load(name, z=None):
    f = files(name)
    return data.load_dwi(f['dwi'], f['bval'], f['bvec'], f['mask'], z)


def estimates(name, method):
    # [ndi, odi, fwf] map files of a method on a subject; 'ref' is the maps shipped with the data
    return files(name)['ref'] if method == 'ref' else data.map_paths(paths.maps(name, method))


def tissues(name, vox):
    # WM, cortical GM and deep GM masks at vox, for those masks.py has written
    out = {}
    for t in ['wm', 'gm', 'deep']:
        f = paths.maps(name, f'tissue_{t}.nii.gz')
        if os.path.exists(f):
            out[t.upper()] = data.load_maps([f], vox)[:, 0] > 0
    return out
