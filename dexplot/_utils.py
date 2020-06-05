import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

RAW_URL = 'https://raw.githubusercontent.com/dexplo/dexplot/master/data/{name}.csv'
DATASETS = ['airbnb']

def load_dataset(name):
    """
    Load a dataset. Must be connected to the internet

    Datasets
    --------
    airbnb
    """
    if name not in DATASETS:
        raise KeyError(f'Dataset {name} does not exist. Choose one of the following: {DATASETS}')

    url = RAW_URL.format(name=name)
    return pd.read_csv(url)


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
