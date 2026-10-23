import pytest
import torch

from dmri_pnet import models

torch.manual_seed(0)


def cloud(n=40):
    g = torch.randn(2, n, 3, dtype=torch.float64)
    return torch.cat([g / g.norm(dim=-1, keepdim=True), torch.rand(2, n, 2, dtype=torch.float64)], -1)


def same(model, a, b):
    with torch.no_grad():
        return torch.allclose(model(a), model(b), atol=1e-10)


ALL = [models.PointNet(), models.PointNet(mirror=True), models.PointNet(tnet=True), models.PairwisePointNet()]
INVARIANT = [models.PairwisePointNet()]


@pytest.mark.parametrize('model', ALL)
def test_permutation_and_duplication(model):
    model.double().eval()
    x = cloud()
    assert same(model, x, x[:, torch.randperm(x.shape[1])])
    assert same(model, x, torch.cat([x, x[:, :5]], 1))


@pytest.mark.parametrize('model', INVARIANT + [models.PointNet(mirror=True)])
def test_antipodal(model):
    model.double().eval()
    x = cloud()
    flipped = x.clone()
    flipped[:, ::3, :3] *= -1
    assert same(model, x, flipped)


@pytest.mark.parametrize('model', INVARIANT)
def test_rotation(model):
    model.double().eval()
    x = cloud()
    R = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
    rotated = x.clone()
    rotated[..., :3] = x[..., :3] @ R.T
    assert same(model, x, rotated)


def test_weight_budget():
    n = lambda m: sum(p.numel() for p in m.parameters())
    w = models.mlp_width(100, 3, 801859)
    assert n(models.PointNet()) == n(models.PairwisePointNet()) == 801859
    assert abs(n(models.SignalMLP(100, [w] * 3)) - 801859) / 801859 < 0.002
