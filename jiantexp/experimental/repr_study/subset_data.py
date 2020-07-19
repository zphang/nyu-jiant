import numpy as np


def get_rng(obj):
    if isinstance(obj, int):
        return np.random.RandomState(seed=obj)
    elif isinstance(obj, np.random.RandomState):
        return obj
    elif obj is None:
        return np.random
    else:
        raise TypeError(obj)


def reindex(data, indices):
    return [data[i] for i in indices]


def safe_sample_indices(data_length, n,
                        allow_subsample=True, allow_supersample=True,
                        rng=None):
    rng = get_rng(rng)
    full_indices = np.arange(data_length)
    if n > data_length:
        assert allow_supersample
        return rng.choice(full_indices, size=n, replace=True)
    elif n < data_length:
        assert allow_subsample
        return rng.choice(full_indices, size=n, replace=False)
    else:
        return full_indices
