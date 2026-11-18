"""
PolyGraph Discrepancy (PGD), Krimmel, Hartout, Borgwardt and Chen, "PolyGraph Discrepancy: a classifier-based metric
for graph generation" (arXiv:2510.06122), with their package polygraph-benchmark
(https://github.com/BorgwardtLab/polygraph-benchmark).

A binary classifier (TabPFN) is trained to tell reference from generated graphs, on each of several graph
descriptors. Its held-out log-likelihood gives a lower bound on the Jensen-Shannon distance between the two
distributions, in [0, 1] (0: indistinguishable, 1: perfectly separated):
    - both sets are split into fit / test halves,
    - for each descriptor, 4-fold cross-validation on the fit half; the descriptor with the best CV score is chosen,
    - its classifier is refit on the whole fit half, and its score on the test half is the PGD.
Descriptors:
    - generic graphs (StandardPGD): orbit counts (4 and 5 nodes), clustering and degree histograms, Laplacian
      spectrum, random GIN embedding,
    - molecules (MoleculePGD): RDKit topological and Lipinski descriptors, Morgan fingerprint, ChemNet and MolCLR
      embeddings (MolCLR weights are downloaded on first use).

Same computation as polygraph's StandardPGD / MoleculePGD.compute, but the reference descriptors are computed
separately (describe), so that they can be cached (eval/ref_cache.py). All descriptors are computed graph by graph
(the random projections and GIN weights are seeded), so the descriptors of a subset are the rows of the set's.

PGD needs as many generated as reference graphs: the larger set is subsampled (fixed seed).
Reported keys: 'pgd', 'pgd_descriptor' (the chosen descriptor) and 'pgd_<descriptor>' (the test score of each).
"""
import random
import warnings

import numpy as np

SEED = 42
MIN_GRAPHS = 16   # below, the fit / test halves and the 4 CV folds are too small

_descriptors = {}   # descriptor functions of each kind, loaded once (MolCLR / ChemNet models)


def get_descriptors(kind):
    if kind not in _descriptors:
        if kind == 'molecule':
            from polygraph.metrics.molecule_pgd import _molecule_descriptors
            _descriptors[kind] = _molecule_descriptors()
        elif kind == 'graph':
            from polygraph.metrics.standard_pgd import _standard_descriptors
            _descriptors[kind] = _standard_descriptors()
        else:
            raise ValueError(f"Unknown kind {kind}, 'graph' or 'molecule'")
    return _descriptors[kind]


def describe(graphs, kind):
    """{descriptor name: descriptions (one row per graph)} of the graphs (None entries are dropped)."""
    graphs = [g for g in graphs if g is not None]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {name: descriptor(graphs) for name, descriptor in get_descriptors(kind).items()}


def compute_pgd(ref_descriptions, generated, kind):
    """
    Args:
        ref_descriptions: describe(reference graphs, kind).
        generated: networkx graphs (kind 'graph') or RDKit molecules (kind 'molecule'; None entries, e.g. invalid
            molecules, are dropped).
    Returns:
        dict with 'pgd', 'pgd_descriptor' and 'pgd_<descriptor>' for each descriptor (nan if it cannot be computed).
    """
    from polygraph.metrics.base.polygraphdiscrepancy import _descriptions_to_classifier_metric, default_classifier

    n_ref = next(iter(ref_descriptions.values())).shape[0]
    generated = [g for g in generated if g is not None]
    n = min(n_ref, len(generated))
    if n < MIN_GRAPHS:
        warnings.warn(f'PGD needs at least {MIN_GRAPHS} reference and generated graphs, got {n_ref} and '
                      f'{len(generated)}: set to nan')
        return {'pgd': float('nan'), 'pgd_descriptor': 'none'}
    if len(generated) > n:
        generated = random.Random(SEED).sample(generated, n)
    ref_idx = np.asarray(random.Random(SEED).sample(range(n_ref), n)) if n_ref > n else np.arange(n_ref)
    gen_descriptions = describe(generated, kind)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier = default_classifier()
            # (CV score, test score) of each descriptor
            scores = {name: _descriptions_to_classifier_metric(ref_descriptions[name][ref_idx],
                                                               gen_descriptions[name], variant='jsd',
                                                               classifier=classifier)
                      for name in gen_descriptions}
    except Exception as e:
        # TabPFN downloads its weights on first use, after a one-time license acceptance (TabPFNLicenseError
        # otherwise): do not stop a run for it
        if type(e).__name__ != 'TabPFNLicenseError':
            raise
        warnings.warn('PGD set to nan: TabPFN needs a license acceptance to download its weights. Accept it at '
                      'https://ux.priorlabs.ai (Licenses tab), then set TABPFN_TOKEN to the API key of '
                      'https://ux.priorlabs.ai/account, or save it in ~/.cache/tabpfn/auth_token.')
        return {'pgd': float('nan'), 'pgd_descriptor': 'none'}
    best = max(scores, key=lambda name: scores[name][0])
    out = {'pgd': float(scores[best][1]), 'pgd_descriptor': best}
    out.update({f'pgd_{name}': float(score[1]) for name, score in scores.items()})
    return out
