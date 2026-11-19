import concurrent.futures
from functools import partial

import zlib

import numpy as np
import pyemd
from scipy.linalg import toeplitz

from eden.graph import vectorize

# NOTES:
# EMD stands for earth move distance, i.e. Wasserstein metric,
# (\inf_{\gama \in \Gama(\mu, \nu) \int_{M*M} d(x,y)^p d\gama(x,y))^(1/p)
# -------- From Niu et al. (2020) --------

def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
        x, y: 1D pmf of two distributions with the same support
        sigma: standard deviation
    """
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float64)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    x = x.astype(float)
    y = y.astype(float)
    x, y = process_tensor(x, y)
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x, y, sigma=1.0):
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0

    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dist in executor.map(kernel_parallel_worker,
                                     [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1],
            ):
                d += dist
    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e6
    return d

def mmd_terms(within_ref, within_gen, cross):
    """
    Squared MMD between the reference X and the generated Y with its three terms:
        mmd = within_ref + within_gen - 2 cross,
        within_ref = E k(x, x'),  within_gen = E k(y, y')  (within-sample terms: average self-similarity, high for a
                                                            homogeneous set),
        cross      = E k(x, y)                             (cross term).
    The averages include the diagonal k(x, x) (biased V-statistic estimator).
    """
    within_ref, within_gen, cross = float(within_ref), float(within_gen), float(cross)
    return {'mmd': within_ref + within_gen - 2 * cross, 'within_ref': within_ref, 'within_gen': within_gen,
            'cross': cross}


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """ MMD between two samples (samples1: reference, samples2: generated), returned with its terms (mmd_terms)
    """
    # -------- normalize histograms into pmf --------
    if is_hist:
        samples1 = [s1 / (np.sum(s1) + 1e-6) for s1 in samples1]
        samples2 = [s2 / (np.sum(s2) + 1e-6) for s2 in samples2]
    x = disc(samples1, samples1, kernel, *args, **kwargs)
    y = disc(samples2, samples2, kernel, *args, **kwargs)
    xy = disc(samples1, samples2, kernel, *args, **kwargs)
    return mmd_terms(x, y, xy)

### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
def _stable_label(label):
    """An integer standing for the label, the same in every process (zlib.crc32 of its repr for non-integers)."""
    return label if isinstance(label, (int, np.integer)) else zlib.crc32(repr(label).encode())


def _stable_labels(graph):
    """Copy of the graph with integer node / edge labels (_stable_label)."""
    graph = graph.copy()
    for _, data in graph.nodes(data=True):
        if 'label' in data:
            data['label'] = _stable_label(data['label'])
    for *_, data in graph.edges(data=True):
        if 'label' in data:
            data['label'] = _stable_label(data['label'])
    return graph


def nspdk_mean_embedding(graphs):
    """Mean of the unit-norm NSPDK features of the graphs (node / edge attribute 'label'), a dense vector.
    eden hashes the labels with Python's hash(), which is randomized per process for strings (e.g. atom symbols): the
    features of two processes would not share their coordinates (which breaks the cached reference embedding, see
    eval/ref_cache.py). The labels are mapped to integers first, whose hash is the same in every process."""
    X = vectorize([_stable_labels(g) for g in graphs], complexity=4, discrete=True)
    return np.asarray(X.mean(axis=0)).ravel()


def nspdk_terms(ref_mean, gen_mean):
    """NSPDK MMD terms from the mean embeddings: the kernel is linear, so the averages of the kernel matrices are dot
    products of the mean embeddings (within_ref = ||m_ref||^2, within_gen = ||m_gen||^2, cross = <m_ref, m_gen>),
    without computing the n x n kernel matrices."""
    d = max(len(ref_mean), len(gen_mean))
    ref_mean, gen_mean = np.pad(ref_mean, (0, d - len(ref_mean))), np.pad(gen_mean, (0, d - len(gen_mean)))
    return mmd_terms(float(ref_mean @ ref_mean), float(gen_mean @ gen_mean), float(ref_mean @ gen_mean))


def compute_nspdk_mmd(samples1, samples2, metric=None, is_hist=True, n_jobs=None):
    """NSPDK MMD between the reference samples1 and the generated samples2, with its terms (mmd_terms). The kernel is
    linear on unit-norm NSPDK features phi, so within_* = 1 - E||phi - mean phi||^2, one minus the total variance.
    (metric, is_hist, n_jobs: unused, kept for compatibility.)"""
    return nspdk_terms(nspdk_mean_embedding(samples1), nspdk_mean_embedding(samples2))


def process_tensor(x, y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    return x, y
