import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def _calculate_figsize(nrows, ncols):
    return ncols * 2 + 8, nrows * 2 + 4


def _get_fig_shape(n, wrap, is_row):
    if is_row:
        if wrap is None:
            return n, 1
        else:
            return wrap, int(np.ceil(n / wrap))
    else:
        if wrap is None:
            return 1, n
        else:
            return int(np.ceil(n / wrap)), wrap


def _map_val_to_color(vals):
    n = len(vals)
    if n <= 10:
        colors = plt.cm.tab10(range(n))
    elif n <= 20:
        colors = plt.cm.tab20(range(n))
    else:
        colors = plt.cm.viridis(n)
    return dict(zip(vals, colors))


def _calculate_density(data):
    density_func = gaussian_kde(data)
    min_x, max_x = data.min(), data.max()
    range_x = max_x - min_x
    min_x = min_x - 2 * range_x
    max_x = max_x + 2 * range_x
    x = np.linspace(min_x, max_x, 400)
    density = density_func(x)
    max_density = density.max()
    filt = density > max_density / 1000
    x = x[filt]
    density = density[filt]
    return x, density
