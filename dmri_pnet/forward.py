# NODDI-Watson in the toolbox formulation (Zhang et al. 2012), built from dmipy compartments.
# dmipy's own NODDI averages zeppelin *signals* over the Watson distribution; the toolbox uses one
# zeppelin along mu with Watson-averaged *diffusivities*. By Jensen's inequality the two differ
# (up to ~0.012 on the signal), so only the extra-neurite compartment is redefined here.
# Public functions take b in s/mm²; dmipy works in SI units.

import os
import numpy as np
from scipy.special import dawsn
from threadpoolctl import threadpool_limits
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import ModelProperties, MultiCompartmentModel
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.signal_models.cylinder_models import C1Stick
from dmipy.signal_models.gaussian_models import G1Ball, G2Zeppelin

D_PAR = 1.7e-9
D_ISO = 3.0e-9


def odi2kappa(odi):
    return 1 / np.tan(np.asarray(odi, float) * np.pi / 2)


def watson_hindered(odi, ndi, d_par=D_PAR):
    # Watson average of zeppelins with tortuosity d_perp = d_par (1 - ndi), written as one zeppelin:
    # tau = E_W[(mu·n)²] = 1 / (2 √κ F(√κ)) - 1 / (2κ), F the Dawson function
    k = odi2kappa(odi)
    tau = 1 / (2 * np.sqrt(k) * dawsn(np.sqrt(k))) - 1 / (2 * k)
    d_perp = d_par * (1 - ndi)
    return d_perp + (d_par - d_perp) * tau, d_perp + (d_par - d_perp) * (1 - tau) / 2


class NODDITissue(ModelProperties):
    # ndi · Watson-dispersed stick + (1 - ndi) · zeppelin along mu with Watson-averaged diffusivities
    _required_acquisition_parameters = ['bvalues', 'gradient_directions']
    _parameter_ranges = {'mu': ([0, np.pi], [-np.pi, np.pi]), 'odi': (0.02, 0.99), 'ndi': (0., 1.)}
    _parameter_scales = {'mu': np.r_[1., 1.], 'odi': 1., 'ndi': 1.}
    _parameter_types = {'mu': 'orientation', 'odi': 'normal', 'ndi': 'normal'}
    _model_type = 'CompartmentModel'

    def __init__(self, mu=None, odi=None, ndi=None):
        self.mu, self.odi, self.ndi = mu, odi, ndi
        self.stick = SD1WatsonDistributed(models=[C1Stick()])
        self.zeppelin = G2Zeppelin()

    def __call__(self, acquisition_scheme, **kwargs):
        mu = kwargs.get('mu', self.mu)
        odi = float(np.squeeze(kwargs.get('odi', self.odi)))
        ndi = float(np.squeeze(kwargs.get('ndi', self.ndi)))
        E_ic = self.stick(acquisition_scheme, SD1Watson_1_mu=mu, SD1Watson_1_odi=odi, C1Stick_1_lambda_par=D_PAR)
        d_par, d_perp = watson_hindered(odi, ndi)
        E_ec = self.zeppelin(acquisition_scheme, mu=mu, lambda_par=d_par, lambda_perp=d_perp)
        return ndi * E_ic + (1 - ndi) * E_ec


def model():
    m = MultiCompartmentModel(models=[G1Ball(), NODDITissue()])
    m.set_fixed_parameter('G1Ball_1_lambda_iso', D_ISO)
    # dmipy bounds volume fractions to [0.01, 0.99]; the toolbox lets FWF reach 0 (as in most of white matter)
    for p in m.partial_volume_names:
        m.parameter_ranges[p] = (0., 1.)
    return m


def scheme(bvals, bvecs):
    # signals are S/S0 without b=0, but dmipy needs one: a b=0 is prepended, with signal 1
    bvecs = np.asarray(bvecs, float)
    bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
    return acquisition_scheme_from_bvalues(np.r_[0., bvals] * 1e6, np.vstack([[0., 0., 1.], bvecs]))


def simulate(bvals, bvecs, params, mu):
    # noise-free S/S0. params (V, 3) [ndi, odi, fwf], mu (V, 3) unit vectors -> (V, M)
    params, mu = np.atleast_2d(params), np.atleast_2d(mu)
    m = model()
    angles = np.c_[np.arccos(np.clip(mu[:, 2], -1, 1)), np.arctan2(mu[:, 1], mu[:, 0])]
    pv = m.parameters_to_parameter_vector(NODDITissue_1_mu=angles, NODDITissue_1_odi=params[:, 1],
                                          NODDITissue_1_ndi=params[:, 0], partial_volume_0=params[:, 2],
                                          partial_volume_1=1 - params[:, 2])
    return np.atleast_2d(m.simulate_signal(scheme(bvals, bvecs), pv))[:, 1:]


def fit(bvals, bvecs, signals, parallel=True):
    # conventional NLLS (dmipy brute2fine), signals (V, M) as S/S0 without b=0
    signals = np.atleast_2d(signals)
    signals = np.c_[np.ones(len(signals)), signals]
    # multithreaded BLAS oversubscribes the cores: single threads make the fit ~3x faster serially,
    # ~15x in parallel. The environment is for dmipy's worker processes, the limit for this one.
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    # on a cluster node, use the CPUs allocated to the job rather than all of the node's
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    with threadpool_limits(1):
        p = model().fit(scheme(bvals, bvecs), signals, use_parallel_processing=parallel,
                        number_of_processors=ncpu).fitted_parameters
    return np.stack([p['NODDITissue_1_ndi'], p['NODDITissue_1_odi'], p['partial_volume_0']], 1)
