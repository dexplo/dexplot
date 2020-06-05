import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

DATASETS = {
    'airbnb': 'asdf'
}

def load_dataset(name):
    if name not in DATASETS:
        names = ', '.join(DATASETS)
        raise KeyError(f'Dataset {name} does not exist. Choose one of the following: {names}')
    pd.read_csv()


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
