import numpy as np
import pytest

from dmri_pnet import forward, sampling

rng = np.random.default_rng(0)
BVALS = np.repeat([500., 1000, 1700, 3200], 30)
BVECS = sampling.unit_vectors(len(BVALS), rng)


def toolbox_quadrature(params, mu, n=200000):
    # independent reference: Watson averages by quadrature on a Fibonacci sphere
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    N = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    b, out = BVALS * 1e6, []
    for (ndi, odi, fwf), m in zip(params, mu):
        w = np.exp(forward.odi2kappa(odi) * (N @ m) ** 2)
        w /= w.sum()
        E_ic = np.exp(-b[:, None] * forward.D_PAR * (BVECS @ N.T) ** 2) @ w
        tau, d_perp = w @ (N @ m) ** 2, forward.D_PAR * (1 - ndi)
        d_par_w, d_perp_w = d_perp + (forward.D_PAR - d_perp) * tau, d_perp + (forward.D_PAR - d_perp) * (1 - tau) / 2
        E_ec = np.exp(-b * (d_perp_w + (d_par_w - d_perp_w) * (BVECS @ m) ** 2))
        out.append(fwf * np.exp(-b * forward.D_ISO) + (1 - fwf) * (ndi * E_ic + (1 - ndi) * E_ec))
    return np.array(out)


@pytest.mark.parametrize('odi', [0.025, 0.05, 0.1, 0.3, 0.6, 0.975])
def test_matches_quadrature(odi):
    # dmipy samples the Watson density on a fixed hemisphere and caps its SH order at 14,
    # so very concentrated fibres (low ODI, high NDI) are only accurate to a few 1e-3
    params = np.array([[ndi, odi, fwf] for ndi in (0.025, 0.5, 0.975) for fwf in (0.025, 0.5)])
    mu = sampling.unit_vectors(len(params), rng)
    err = np.abs(forward.simulate(BVALS, BVECS, params, mu) - toolbox_quadrature(params, mu)).max()
    assert err < (5e-3 if odi <= 0.05 else 1e-3)


def test_fit_recovers_noise_free():
    params, mu = sampling.noddi_params(30, rng)
    est = forward.fit(BVALS, BVECS, forward.simulate(BVALS, BVECS, params, mu), parallel=False)
    rmse = np.sqrt(((est - params) ** 2).mean(0))
    assert rmse[0] < 0.03 and rmse[2] < 0.03
