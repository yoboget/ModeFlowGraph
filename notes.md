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