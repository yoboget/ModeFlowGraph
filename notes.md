# Notes

## Corrector sampling
- Small corrector (eta~0.05, lambda=0) is quite systematically the best config.
- The new path called sparse seems to yield the best results (c=4 is a good setting).
- The sufficient statistics input produces slightly better results.
- Extra feature with dirichlet seems to help in qm9. To be confirmed on zinc. 

## TODO

1. SDM is missing from the experiments.
7. **Variability on molecules.** The QM9 and ZINC tables have no standard deviations and don't say how many sampling runs were used. The DNA tables do.
8. **Molecular coverage:**
   - the vector field vs. quantile map comparison and the timing only cover QM9, not ZINC;
   - uniqueness and novelty, which are standard on these benchmarks, are not reported.

9. **An algorithm box for the generalized sampler.** One step would be: sample the weights (λ), transport with the quantile map, then apply the corrector (η). It helps reproducibility more than any paragraph.


## Conclusion

The gains actually come from two places:
the quantile map's stability and speed;
the η corrector, used at a setting that is not the exact deterministic sampler.