import numpy as np

gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


def get_rng(rng: int | np.random.Generator = None) -> np.random.Generator:
    '''
    Retrieves or generate with a seed a random number generator RNG.

    Parameters
    ----------
    rng : int or np.random.Generator, default = np.random.default_rng()
        A seed for random generation or directly a numpy random generator.

    Returns
    -------
    rng : np.random.Generator
        A Generator instance.
    '''
    if rng is None:
        return np.random.default_rng()

    if isinstance(rng, int):
        return np.random.default_rng(seed=rng)

    return rng


def random_choice(rng: np.random.Generator,
                  a: np.ndarray,
                  size: int | tuple[int] = 1,
                  replace: bool = True,
                  p: np.ndarray = None) -> np.ndarray:
    '''
    A drop-in replacement for the rng.choice function that interchangeably accepts numpy or cupy arrays.

    Parameters
    ----------
    rng : np.random.Generator
        The random generator to use.
    a : np.ndarray
        The array to choose items out of.
    size : int | tuple[int], default = 1
        How many items are to be chosen.
    replace : bool, default = True
        Whether the items can be chosen after being picked already.
    p : np.ndarray, optional
        Probability distributions to be applied to the elements

    Returns
    -------
    np.ndarray
        _description_
    '''
    if gpu_support and cp.get_array_module(a) == cp:
        assert len(a.shape) == 1, "Shape or array not supported..."
        idx = np.arange(len(a), dtype=int)
        chosen_idx = rng.choice(idx, size, replace, (None if p is None else cp.asnumpy(p)))
        return a[chosen_idx]
    else:
        return rng.choice(a, size, replace, p)