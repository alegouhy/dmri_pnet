import os
import argparse
import tempfile
import numpy as np
import nibabel as nib

from dmri_pnet import paths, protocols, subjects, tools

p = argparse.ArgumentParser(description='Mean b=0 and tissue masks of a subject from SynthSeg '
                                        '-> data/maps/<subject>/{b0,synthseg,tissue_*}.nii.gz')
p.add_argument('--subject', required=True)
a = p.parse_args()

# FreeSurfer labels; cerebellum and brainstem are left out
LABELS = dict(wm=[2, 41], gm=[3, 42], deep=[10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60],
              csf=[4, 5, 14, 15, 24, 43, 44])

f = subjects.files(a.subject)
img = nib.load(f['dwi'])
mask = nib.load(f['mask']).get_fdata() > 0
is_b0 = protocols.read(f['bval'], f['bvec'])[2]
# eddy-corrected data can carry NaNs and negative values, which break SynthSeg
b0 = np.clip(np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32)[..., is_b0].mean(-1)), 0, None)
nib.save(nib.Nifti1Image(b0 * mask, img.affine), paths.ensure(paths.maps(a.subject, 'b0.nii.gz')))

with tempfile.TemporaryDirectory() as tmp:
    nib.save(nib.Nifti1Image(b0, img.affine), os.path.join(tmp, 'b0.nii.gz'))
    tools.synthseg(os.path.join(tmp, 'b0.nii.gz'), paths.maps(a.subject, 'synthseg.nii.gz'))
labels = nib.load(paths.maps(a.subject, 'synthseg.nii.gz')).get_fdata()

for name, values in LABELS.items():
    sel = np.isin(labels, values) & mask
    nib.save(nib.Nifti1Image(sel.astype(np.uint8), img.affine), paths.maps(a.subject, f'tissue_{name}.nii.gz'))
    print(f'{a.subject} {name}: {sel.sum()} voxels')
