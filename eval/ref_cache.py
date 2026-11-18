"""
Disk cache of the reference-side statistics of the metrics (e.g. the canonical test smiles, the NSPDK mean embedding,
the FCD mean and covariance, the PGD descriptors of the test set).

The test sets are fixed (qm9 / zinc: test_idx_<dataset>.json, qm9H: seeded split) and their subsets are seeded, so
these statistics only depend on the dataset and the reference size: they are computed once and saved to
`./data/<dataset>/raw/ref_stats/<name>.pkl`, then reused by every run (and kept in memory within a run).
Delete that directory to recompute them, e.g. after changing the test set or a metric's reference computation;
VERSION is bumped when the content of the cache changes.
"""
import os
import pickle
import time

VERSION = 1

_memory = {}


def cache_path(dataset, name):
    return os.path.join('./data', dataset, 'raw', 'ref_stats', f'{name}.pkl')


def load_or_compute(dataset, name, compute):
    """The cached value of `name` for `dataset`, or compute() (then saved) if missing or outdated."""
    path = cache_path(dataset, name)
    if path in _memory:
        return _memory[path]
    value, found = None, False
    if os.path.exists(path):
        with open(path, 'rb') as f:
            stored = pickle.load(f)
        if stored.get('version') == VERSION:
            value, found = stored['value'], True
        else:
            print(f'{path} is outdated (version {stored.get("version")}, expected {VERSION}): recomputing it')
    if not found:
        start = time.time()
        print(f'Computing the reference statistics {path} (once for all runs)')
        value = compute()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp'   # atomic: several jobs may write the same file
        with open(tmp, 'wb') as f:
            pickle.dump({'version': VERSION, 'value': value}, f)
        os.replace(tmp, path)
        print(f'Saved {path} ({time.time() - start:.1f} s)')
    _memory[path] = value
    return value
