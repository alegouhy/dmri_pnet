import numpy as np
import matplotlib.pyplot as plt

from .sampling import PARAMS
from .metrics import EDGES, binned

CENTRES = (EDGES[:-1] + EDGES[1:]) / 2


def colours(methods):
    return {m: plt.cm.tab10(k % 10)[:3] for k, m in enumerate(methods)}


def boxes(ax, data, positions, width, colour):
    bp = ax.boxplot(data, positions=positions, widths=width, showfliers=False,
                    patch_artist=True, manage_ticks=False)
    for part in bp:
        plt.setp(bp[part], color=colour)
    for box in bp['boxes']:
        box.set_facecolor((*colour, 0.5))


def losses(curves):
    # one colour per curve, train and val included
    fig, ax = plt.subplots(figsize=(7, 4))
    lines = {f'{name} {split}': loss for name, (train, val) in curves.items()
             for split, loss in (('train', train), ('val', val))}
    for (label, loss), c in zip(lines.items(), colours(lines).values()):
        ax.plot(loss, color=c, label=label)
    ax.set(xlabel='epoch', ylabel='MSE', yscale='log')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    return fig


def signed_error(preds, ref, ref_name, title=''):
    # signed error per 0.1 bin of the reference: shrinkage toward the mean shows up as a tilt
    col = colours(preds)
    w = 0.08 / len(preds)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for k, (ax, p) in enumerate(zip(axes, PARAMS)):
        ib = np.clip(np.digitize(ref[:, k], EDGES) - 1, 0, len(CENTRES) - 1)
        for r, (m, pred) in enumerate(preds.items()):
            bins = [b for b in range(len(CENTRES)) if (ib == b).any()]
            boxes(ax, [pred[ib == b, k] - ref[ib == b, k] for b in bins],
                  CENTRES[bins] + (r - (len(preds) - 1) / 2) * w, 0.9 * w, col[m])
        ax.axhline(0, color='k', lw=0.8)
        ax.set(xlim=(0, 1), ylim=(-0.8, 0.8), xticks=EDGES, ylabel=f'{p.upper()}\npred - {ref_name}')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(ref_name)
    axes[0].legend(handles=[plt.Line2D([], [], color=col[m], lw=6, label=m) for m in preds],
                   ncol=len(preds), loc='upper right')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def scatter(preds, ref, ref_name, title=''):
    col = colours(preds)
    fig, axes = plt.subplots(len(preds), 3, figsize=(12.6, 3.9 * len(preds)),
                             squeeze=False, sharex=True, sharey=True)
    for r, (m, pred) in enumerate(preds.items()):
        for k, p in enumerate(PARAMS):
            ax = axes[r, k]
            ax.plot([0, 1], [0, 1], 'k', lw=1)
            ax.scatter(ref[:, k], pred[:, k], s=2, alpha=0.05, color=col[m])
            ax.set(xlim=(0, 1), ylim=(0, 1))
            if r == 0:
                ax.set_title(p.upper())
            if k == 0:
                ax.set_ylabel(f'{m}\npredicted')
            if r == len(preds) - 1:
                ax.set_xlabel(ref_name)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def error_dependence(preds, ref, ref_name, title=''):
    # MAE of each parameter binned by each parameter: where in parameter space each method fails
    col = colours(preds)
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 9.3), sharex=True, sharey='row')
    for i, target in enumerate(PARAMS):
        for j, driver in enumerate(PARAMS):
            ax = axes[i, j]
            for m, pred in preds.items():
                ax.plot(CENTRES, binned(np.abs(pred[:, i] - ref[:, i]), ref[:, j]), 'o-', color=col[m], label=m)
            ax.set(xlim=(0, 1), ylim=(0, None))
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f'binned by {driver.upper()}')
            if j == 0:
                ax.set_ylabel(f'MAE of {target.upper()}')
            if i == 2:
                ax.set_xlabel(ref_name)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def maps(entries, vox, shape, z, cmap='viridis', vmin=0, vmax=1, label='', title=''):
    # entries: {name: (V, 3) values at vox}; one row per entry, axial slice z cropped to the data
    sl = vox[:, 2] == z
    x, y = vox[sl, 0], vox[sl, 1]
    crop = (slice(x.min(), x.max() + 1), slice(y.min(), y.max() + 1))
    aspect = (np.ptp(y) + 1) / (np.ptp(x) + 1)
    fig, axes = plt.subplots(len(entries), 3, squeeze=False, figsize=(12, 4 * aspect * len(entries)),
                             gridspec_kw=dict(wspace=0.01, hspace=0.01))
    for r, (name, values) in enumerate(entries.items()):
        for k, p in enumerate(PARAMS):
            img = np.full(shape[:2], np.nan)
            img[x, y] = values[sl, k]
            ax = axes[r, k]
            im = ax.imshow(img[crop].T, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', interpolation='nearest')
            ax.set(xticks=[], yticks=[])
            ax.spines[:].set_visible(False)
            if r == 0:
                ax.set_title(p.upper(), fontsize=20)
            if k == 0:
                ax.set_ylabel(name, fontsize=16)
    fig.colorbar(im, ax=axes, shrink=0.7, label=label)
    fig.suptitle(f'{title} — axial slice {z}', fontsize=20)
    return fig


def qspace(bvals, bvecs, signal, title=''):
    # signal in q-space, one surface per shell, radius = signal (S(g) = S(-g), so mirrored)
    import plotly.graph_objects as go
    from scipy.spatial import ConvexHull
    fig = go.Figure()
    for b in np.unique(bvals):
        sel = np.isclose(bvals, b)
        g = np.vstack([bvecs[sel], -bvecs[sel]])
        p = g * np.r_[signal[sel], signal[sel]][:, None]
        tri = ConvexHull(g).simplices
        fig.add_trace(go.Mesh3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                                opacity=0.4, name=f'b = {b:.0f}', showlegend=True))
    fig.update_layout(title=title, scene=dict(aspectmode='cube'))
    return fig
