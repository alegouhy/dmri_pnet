import numpy as np

PARAMS = ['ndi', 'odi', 'fwf']


def unit_vectors(n, rng):
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def noddi_params(n, rng, lo=0.025, hi=0.975):
    # ndi is within-tissue (IC / (IC+EC)), so ndi, odi, fwf are independent on [0,1]³
    return rng.uniform(lo, hi, (n, 3)), unit_vectors(n, rng)


def noddi_grid(n, nrep, rng, lo=0.025, hi=0.975):
    # n³ grid in (ndi, odi, fwf), each point repeated nrep times with a fresh orientation
    axis = np.linspace(lo, hi, n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    params = np.repeat(grid, nrep, axis=0)
    return params, unit_vectors(len(params), rng)


def random_protocol(rng, nshell=(2, 5), b=(250, 5000), ndir=(12, 128)):
    k = rng.integers(nshell[0], nshell[1] + 1)
    sizes = rng.integers(ndir[0], ndir[1] + 1, k)
    bvals = np.repeat(rng.uniform(*b, k), sizes)
    return bvals, unit_vectors(len(bvals), rng)


def rician(signal, sigma, rng):
    # sigma relative to S0 = 1, i.e. 1/SNR; 0 is noise-free
    n1, n2 = rng.standard_normal((2,) + np.shape(signal)) * sigma
    return np.sqrt((signal + n1) ** 2 + n2 ** 2)


def noisy(signal, sigma, rng, n_b0=None):
    # Rician noise; with n_b0 = (min, max), S/S0 as in real data: divided by the mean of a random number of
    # noisy b=0s rather than by the true S0 = 1. Without it, the draws are those of rician alone.
    x = rician(signal, sigma, rng)
    if n_b0 is not None:
        x = x / rician(np.ones(rng.integers(n_b0[0], n_b0[1] + 1)), sigma, rng).mean()
    return x


def keep_points(n, rng, p_drop=0., fdrop=(0, 0.5), max_pts=None):
    # indices kept after random point dropout (prob. p_drop, fraction ~ U(fdrop)) then subsampling to max_pts
    keep = n
    if rng.random() < p_drop:
        keep = max(1, int(n * (1 - rng.uniform(*fdrop))))
    if max_pts is not None:
        keep = min(keep, max_pts)
    return np.arange(n) if keep == n else rng.choice(n, keep, replace=False)
