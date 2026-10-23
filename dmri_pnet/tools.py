# External tools, called from PATH: Anima (rigid registration) and FreeSurfer's mri_synthseg (segmentation)

import os
import subprocess


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True, stdout=subprocess.DEVNULL)


def register(moving, ref, xml):
    # rigid block-matching registration of moving onto ref, saved as an Anima transform series
    stem = xml[:-len('.xml')]
    run('animaPyramidalBMRegistration', '-m', moving, '-r', ref, '-o', stem + '_moved.nii.gz', '-O', stem + '.txt',
        '--ot', 0)
    run('animaTransformSerieXmlGenerator', '-i', stem + '.txt', '-o', xml)


def resample(image, ref, xml, out, interp='linear'):
    run('animaApplyTransformSerie', '-i', image, '-t', xml, '-g', ref, '-o', out, '-n', interp)


def synthseg(image, out):
    # contrast-agnostic, so a b=0 image needs no T1; --keepgeom writes the labels on the input grid
    run('mri_synthseg', '--i', image, '--o', out, '--robust', '--keepgeom', '--threads', os.cpu_count())
