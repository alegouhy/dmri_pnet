import numpy as np

from dmri_pnet import protocols, sampling

rng = np.random.default_rng(0)


def test_prior_is_flat_and_independent():
    params, mu = sampling.noddi_params(200000, rng)
    for k in range(3):
        hist = np.histogram(params[:, k], 10, (0.025, 0.975))[0]
        assert np.abs(hist / hist.mean() - 1).max() < 0.05
    assert np.abs(np.corrcoef(params.T) - np.eye(3)).max() < 0.01
    assert np.allclose(np.linalg.norm(mu, axis=1), 1)


def test_random_protocol_ranges():
    for _ in range(200):
        bvals, bvecs = sampling.random_protocol(rng)
        shells = np.unique(bvals)
        assert 2 <= len(shells) <= 5 and shells.min() >= 250 and shells.max() <= 5000
        assert all(12 <= (bvals == b).sum() <= 128 for b in shells)
        assert np.allclose(np.linalg.norm(bvecs, axis=1), 1)


def test_keep_points():
    for _ in range(200):
        keep = sampling.keep_points(300, rng, p_drop=1., max_pts=128)
        assert 1 <= len(keep) <= 128 and len(np.unique(keep)) == len(keep)
    assert len(sampling.keep_points(50, rng, max_pts=128)) == 50


def test_fixed_protocols():
    bvals = protocols.dsi()[0]
    assert len(bvals) == 514 and len(np.unique(bvals)) == 22 and bvals.min() == 200 and bvals.max() == 5000
    bvals, bvecs = protocols.hardi()
    cos = np.abs(bvecs @ bvecs.T) - 2 * np.eye(len(bvecs))
    assert np.degrees(np.arccos(cos.max())) > 15      # directions spread out, antipodes included


def test_b0_normalisation():
    x = np.full(50, 0.5)
    # without n_b0, exactly the draws of rician
    assert np.array_equal(sampling.noisy(x, 0.05, np.random.default_rng(1)),
                          sampling.rician(x, 0.05, np.random.default_rng(1)))
    # one noisy b=0 scales S/S0 by 1 ± sigma, more b=0s by less
    r = np.random.default_rng(2)
    spread = [np.std([sampling.noisy(x, 0.05, r, (n, n)).mean() / 0.5 for _ in range(2000)]) for n in (1, 10)]
    assert 0.045 < spread[0] < 0.06 and spread[1] < 0.03
