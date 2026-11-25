# Notes

## Corrector sampling
- Small corrector (eta~0.05, lambda=0) is quite systematically the best config.
- The new path called sparse seems to yield the best results (c=4 is a good setting).
- The sufficient statistics input produces slightly better results.
- Extra feature with dirichlet seems to help in qm9. To be confirmed on zinc. 

## TODO
- Add the molecule protocol
- Define the architecture and hyperparameters . 
- Modify the main text.
- Include the new results in the paper.
  - variation lambda
  - variation eta
  - Not in Simplex Diffusion sampler


The appendix covers background, evaluation protocols and the three experiment subsections. Here is what's still missing, ordered by how exposed the paper is without it.

**Missing support for claims the paper makes**
1. **The proof of the contraction proposition.** Section 4.3 and the third contribution say "we prove" a strict contraction, but there is no proof yet. It needs three steps:
   - the predictor preserves the TV distance, because it is a bijection;
   - the corrector contracts with a constant $\epsilon_\eta > 0$ for η > 0, by Dobrushin's argument with a uniform overlap bound;
   - combining the two.

   Also check that the proposition statement itself is in the main text; it was absent from the last draft.
2. **Your point 3: the training-free improvement against the actual baselines.** The ablation compares against the exact sampler (η = 0), not against DFM or UNSID as typically run. A short summary table would close this: one row per dataset, with columns for DFM (vector field, averaging), UNSID (η = 1) and your tuned configuration, all with the same denoiser and NFE. Without it, the claim rests on the main table alone.
3. **The proof of the exact-flow proposition in `ap:proofs`.** The simplified version exists, but it is written for DFM's path with $F_\alpha$. The main text now uses the general path, $F_t$ and $G^t_{t+h}$, so the notation needs aligning. The generalized-path remark covers the gap.
4. **The values in Table `tab:sampling-params`:** α_max for DNA, and λ and η for each "ours" result.

**Gaps a reviewer may point out**

5. **The off-axis claim.** The design-space paragraph says samplers with λ > 0 and η > 0 are never exact and degrade quickly. There is no experiment for this, and the combined sampler isn't defined. Either add a small (λ, η) grid on one dataset, or drop the sentences.
6. **Training details.** Optimizer, learning rate, epochs, batch size and hardware are missing for the QM9 model and for the ZINC model you retrained. The architecture paragraph only gives sizes.
7. **Variability on molecules.** The QM9 and ZINC tables have no standard deviations and don't say how many sampling runs were used. The DNA tables do.
8. **Molecular coverage:**
   - the vector field vs. quantile map comparison and the timing only cover QM9, not ZINC;
   - uniqueness and novelty, which are standard on these benchmarks, are not reported.

**Optional, but cheap and useful**

9. **An algorithm box for the generalized sampler.** One step would be: sample the weights (λ), transport with the quantile map, then apply the corrector (η). It helps reproducibility more than any paragraph.
10. **A short derivation backing two main-text sentences:**
    - "averaging pulls the iterates toward the interior": the covariance-deficit identity from earlier would do;
    - the factorization caveat: the two-position counterexample would do.
11. **A label check.** Before submission, check that every reference from the main text resolves, in particular:
    - `ap:proofs`, `ap:experiment_details` and `ap:sampling-params`;
    - `prop:contraction` and `eq:dfm-vf-general`;
    - `eq:C_vectorfield` vs `eq:vectorfield`.

I'd start with items 1 and 2: they are the two places where the paper currently claims more than the appendix shows. I can draft the contraction proof and the summary table template next.