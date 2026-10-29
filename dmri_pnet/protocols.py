# Fixed acquisition protocols: bvals (M,) in s/mm² and unit bvecs (M, 3), without b=0.

import numpy as np
from scipy.optimize import minimize

B0_THRESH = 50


def read(bval, bvec):
    # FSL-style files; also returns the b=0 mask over all volumes, used for S0
    bvals = np.loadtxt(bval)
    bvecs = np.loadtxt(bvec).T
    is_b0 = bvals < B0_THRESH
    bvecs = bvecs[~is_b0] / np.linalg.norm(bvecs[~is_b0], axis=1, keepdims=True)
    return bvals[~is_b0], bvecs, is_b0


def electrostatic(sizes, seed=0):
    # antipodal Coulomb repulsion, per shell and over all shells pooled (Caruyer-style),
    # so each shell and the union are both near-uniform
    shell = np.repeat(np.arange(len(sizes)), sizes)
    groups = [shell == k for k in range(len(sizes))] + ([shell >= 0] if len(sizes) > 1 else [])

    def energy(x):
        p = x.reshape(-1, 3)
        n = np.linalg.norm(p, axis=1, keepdims=True)
        u = p / n
        E, G = 0., np.zeros_like(u)
        for sel in groups:
            v = u[sel]
            w = 1 / sel.sum() ** 2
            d1, d2 = v[:, None] - v[None], v[:, None] + v[None]
            r1, r2 = np.linalg.norm(d1, axis=-1), np.linalg.norm(d2, axis=-1)
            np.fill_diagonal(r1, np.inf)
            np.fill_diagonal(r2, np.inf)
            E += w * 0.5 * (1 / r1 + 1 / r2).sum()
            G[sel] -= w * ((d1 / r1[..., None] ** 3).sum(1) + (d2 / r2[..., None] ** 3).sum(1))
        return E, ((G - (G * u).sum(1, keepdims=True) * u) / n).ravel()

    x0 = np.random.default_rng(seed).standard_normal((len(shell), 3))
    x = minimize(energy, x0.ravel(), jac=True, method='L-BFGS-B', options={'maxiter': 2000}).x
    x = x.reshape(-1, 3)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ukb():
    return np.repeat([1000., 2000.], 50), electrostatic([50, 50])


def hardi():
    return np.full(60, 2000.), electrostatic([60])


def dsi(n=5, bmax=5000):
    # Cartesian q-space grid inside the sphere of radius n (DSI-515 for n=5, origin = b=0), b ∝ |q|² up to bmax
    g = np.arange(-n, n + 1)
    q = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    r2 = (q ** 2).sum(1)
    q, r2 = q[(r2 > 0) & (r2 <= n ** 2)], r2[(r2 > 0) & (r2 <= n ** 2)]
    return bmax * r2 / n ** 2, q / np.sqrt(r2)[:, None]
