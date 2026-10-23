import numpy as np
import torch

from dmri_pnet import data, forward, models, sampling

rng = np.random.default_rng(0)


def random_set(n=50):
    params, mu = sampling.noddi_params(n, rng)
    protos = [sampling.random_protocol(rng) for _ in range(n)]
    signals = [forward.simulate(*p, params[i], mu[i])[0] for i, p in enumerate(protos)]
    return data.make_set(protos, np.arange(n), signals=data.flat(signals), params=params, mu=mu,
                         voxel=np.arange(n) // 2), protos, signals


def test_set_roundtrip(tmp_path):
    s, protos, signals = random_set()
    np.savez(tmp_path / 's.npz', **s)
    s = data.load_set(tmp_path / 's.npz')
    for i in [0, 7, 49]:
        assert np.allclose(data.protocol(s, i)[0], protos[i][0]) and np.allclose(data.signal(s, i), signals[i])


def test_split_by_voxel():
    s = random_set()[0]
    train, val = data.split(s, 5)
    assert len(val) == 10 and not set(s['voxel'][train]) & set(s['voxel'][val])


def test_cloud_dataset_is_reproducible_and_changes_with_epoch():
    s = random_set()[0]
    ds = data.CloudDataset(s, snr=(10, 40), p_drop=0.5, max_pts=64)
    a, b = ds[3][0], ds[3][0]
    ds.epoch = 1
    assert torch.equal(a, b) and len(a) <= 64 and not torch.equal(a, ds[3][0])


def test_padding_is_exact():
    s = random_set()[0]
    ds = data.CloudDataset(s, max_pts=128)
    model = models.PairwisePointNet().eval()
    x, _ = data.pad_collate([ds[i] for i in range(8)])
    with torch.no_grad():
        alone = torch.cat([model(ds[i][0][None]) for i in range(8)])
        assert torch.allclose(model(x), alone, atol=1e-6)


def test_predict_fixed_protocol():
    params, mu = sampling.noddi_params(20, rng)
    bvals, bvecs = sampling.random_protocol(rng)
    s = data.make_set([(bvals, bvecs)], np.zeros(20, int),
                      signals=data.flat(forward.simulate(bvals, bvecs, params, mu)), params=params)
    cfg = dict(model='pairwise', max_pts=32, bmax=data.BMAX)
    out = models.predict(models.PairwisePointNet().eval(), cfg, s, n_sub=3)
    assert out.shape == (20, 3)


def test_select_subsets_and_reorders_every_signal_copy():
    params, mu = sampling.noddi_params(10, rng)
    bvals, bvecs = sampling.random_protocol(rng)
    S = forward.simulate(bvals, bvecs, params, mu)
    s = data.make_set([(bvals, bvecs)], np.zeros(10, int), signals=data.flat(S), snr30=data.flat(S + 1), params=params)
    keep = rng.permutation(len(bvals))[:7]
    t = data.select(s, keep)
    b, g, x = data.fixed(t)
    assert np.allclose(b, bvals[keep]) and np.allclose(x, S[:, keep]) and np.allclose(data.fixed(t, 'snr30')[2], S[:, keep] + 1)
    assert np.array_equal(t['params'], s['params'])
