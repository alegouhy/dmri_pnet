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
    assert len(protocols.dsi()[0]) == 514 and len(np.unique(protocols.dsi()[0])) == 22
    bvals, bvecs = protocols.hardi()
    cos = np.abs(bvecs @ bvecs.T) - 2 * np.eye(len(bvecs))
    assert np.degrees(np.arccos(cos.max())) > 15      # directions spread out, antipodes included
