import numpy as np
import pandas as pd

from .sampling import PARAMS

EDGES = np.linspace(0, 1, 11)


def scores(pred, ref):
    # slope < 1: estimates shrunk toward the prior mean, where the data does not determine the parameter
    out = {}
    for k, p in enumerate(PARAMS):
        q, r = pred[:, k], ref[:, k]
        e = q - r
        out[p] = dict(RMSE=np.sqrt((e ** 2).mean()), MAE=np.abs(e).mean(), medAE=np.median(np.abs(e)),
                      P90AE=np.percentile(np.abs(e), 90), bias=e.mean(), slope=np.polyfit(r, q, 1)[0],
                      corr=np.corrcoef(r, q)[0, 1], R2=1 - (e ** 2).mean() / r.var())
    return out


def frame(preds, ref, subset):
    # preds: {method: estimates} -> long table, one row per (method, param, metric)
    return pd.DataFrame([dict(subset=subset, method=m, param=p, metric=k, value=v)
                         for m, est in preds.items() for p, sc in scores(est, ref).items() for k, v in sc.items()])


def table(df):
    # methods as rows and metrics as columns, one block per subset and parameter
    lines = []
    for (subset, param), block in df.groupby(['subset', 'param'], sort=False):
        block = block.pivot(index='method', columns='metric', values='value').loc[block.method.unique(), block.metric.unique()]
        lines += [f'\n{subset} — {param.upper()}', block.to_string(float_format='{:.3f}'.format)]
    return '\n'.join(lines)


def binned(values, by, edges=EDGES):
    # mean of values in each bin of by (nan for empty bins)
    ib = np.clip(np.digitize(by, edges) - 1, 0, len(edges) - 2)
    return np.array([values[ib == k].mean() if (ib == k).any() else np.nan for k in range(len(edges) - 1)])
