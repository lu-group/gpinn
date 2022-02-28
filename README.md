# gPINN: Gradient-enhanced physics-informed neural networks

The data and code for the paper [J. Yu, L. Lu, X. Meng, & G. E. Karniadakis. Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems. *arXiv preprint arXiv:2111.02801*, 2021](https://arxiv.org/abs/2111.02801).

## Code

- [Function approximation](src/function.py)
- Forward PDE problems
    - [Poisson equation in 1D](src/poisson_1d.py)
    - [Diffusion-reaction equation](src/diffusion_reaction.py)
    - [Poisson equation in 2D](src/poisson_2d.py)
- Inverse PDEs problems
    - Brinkman-Forchheimer model
        - [Case 1](src/brinkman_forchheimer_1.py)
        - [Case 2](src/brinkman_forchheimer_2.py)
    - [Diffusion-reaction system](src/diffusion_reaction_inverse.py)
- gPINN enhanced by RAR
    - [Burgers' equation](src/burgers.py)
    - [Allen-Cahn equation](src/allen_cahn.py)

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{yu2021gradient,
  title={Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems},
  author={Yu, Jeremy and Lu, Lu and Meng, Xuhui and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2111.02801},
  year={2021}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
