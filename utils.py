import random
from sklearn import datasets
import pandas as pd


def get_unbalanced_moons(n_samples, balance_fraction=0.6, noise=0.4):
    X, y = datasets.make_moons(noise=0.4)
    n_samples_0 = int(balance_fraction * n_samples)
    n_samples_1 = n_samples - n_samples_0
    max_samples_per_class = max(n_samples_0, n_samples_1)
    
    X, y = datasets.make_moons(max_samples_per_class*2, noise=noise)
    
    y_series = pd.Series(y)
    
    sample_indexes_0 = random.sample(list(y_series[y_series == 0].index.values), n_samples_0)
    sample_indexes_1 = random.sample(list(y_series[y_series == 1].index.values), n_samples_1)
    total_indexes = sample_indexes_0 + sample_indexes_1
    random.shuffle(total_indexes)

    return X[total_indexes], y[total_indexes]