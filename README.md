# Physics-Informed Neural Networks for Navier-Stokes: TensorFlow 2, PyTorch, and Data-Parallel Versions

This repository contains three adaptations of the physics-informed neural network (PINN) for the
2D incompressible Navier-Stokes equations (flow past a cylinder), from
[Raissi et al.'s original PINNs repository](https://github.com/maziarraissi/PINNs).

> **Attribution:** This code adapts the original TensorFlow 1 implementation by Maziar Raissi,
> Paris Perdikaris, and George Em Karniadakis, released under the MIT License. The original
> copyright and license notices are kept in [LICENSE](LICENSE). The TensorFlow 2, PyTorch, and
> data-parallel versions here are my adaptations of that code, not the original implementation.

I worked on these as a research intern in the SEEM Lab at Rice University (advisor: Prof. Sahar Bakhshian).

## The problem

Given scattered measurements of the velocity field (u, v), the network learns the stream function
ψ(x, y, t) and the pressure p(x, y, t), with velocities obtained as u = ∂ψ/∂y and v = −∂ψ/∂x.
The loss combines the data mismatch with the Navier-Stokes residuals:

```
f_u = u_t + λ1 (u u_x + v u_y) + p_x − λ2 (u_xx + u_yy)
f_v = v_t + λ1 (u v_x + v v_y) + p_y − λ2 (v_xx + v_yy)
```

The unknown coefficients λ1 (convection) and λ2 (viscosity) are learned along with the network.
Their true values for this dataset are 1.0 and 0.01.

## What's in this repository

| Folder | Framework | Description |
| --- | --- | --- |
| [`main/tensorflow2/`](main/tensorflow2) | TensorFlow 2 | Migration of the original TensorFlow 1 code to TensorFlow 2 |
| [`main/pytorch/`](main/pytorch) | PyTorch | PyTorch reimplementation of the same model |
| [`main/tensorflow1_data_parallel/`](main/tensorflow1_data_parallel) | TensorFlow 1.x | Data-parallel training across multiple devices |
| [`Utilities/`](Utilities) | | Plotting helpers from the original repository |
| [`main/Data/`](main/Data) | | Where the dataset goes (see below) |

### TensorFlow 2 version
- Replaces TF1 sessions, placeholders, and `tf.gradients` with eager execution and `tf.GradientTape`
  for the first and second derivatives in the PDE residuals.
- Replaces the TF1 Adam optimizer with Keras Adam, and the TF1 `ScipyOptimizerInterface` L-BFGS-B
  with `tfp.optimizer.lbfgs_minimize` from TensorFlow Probability.
- Uses mean-squared (`reduce_mean`) instead of summed (`reduce_sum`) loss terms, and shorter training
  runs than the original (10,000 Adam iterations and up to 2,500 L-BFGS iterations).
- Keeps the original network architecture, stream-function formulation, and learnable λ1 and λ2.

### PyTorch version
- Reimplements the model with `torch.nn` and computes derivatives with `torch.autograd.grad`.
- Trains with Adam followed by `torch.optim.LBFGS` (strong Wolfe line search).
- Uses a 6×36 hidden-layer network.
- Runs on a GPU automatically when one is available.

### Data-parallel version (TensorFlow 1.x)
- Splits each training batch across devices (two GPUs by default), computes each device's loss and
  gradients separately, then averages the gradients before each Adam update.
- Followed by L-BFGS-B refinement, as in the original.
- Requires TensorFlow 1.x, since it uses sessions, placeholders, and `tf.contrib`.

## Getting started

1. **Download the data.** Get `cylinder_nektar_wake.mat` and `cylinder_nektar_t0_vorticity.mat` from the
   [original repository's `main/Data` folder](https://github.com/maziarraissi/PINNs/tree/master/main/Data)
   and put them in `main/Data/`.
2. **Install the requirements** for the version you want to run:
   - TensorFlow 2: `pip install tensorflow tensorflow-probability tf-keras numpy scipy matplotlib`
   - PyTorch: `pip install torch numpy scipy`
   - Data-parallel: TensorFlow 1.15 with Python 3.7 or earlier, plus `numpy scipy matplotlib`
3. **Run a script from its own folder**, since data and utility paths are relative:
   ```
   cd main/tensorflow2
   python NavierStokesTF2.py
   ```

The plotting code in `Utilities/plotting.py` renders text with LaTeX (`text.usetex`). If you don't have
LaTeX installed, set `"text.usetex": False` in that file.

## Citation

If you use this code, please cite the original PINNs paper:

```
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier}
}
```

## License

MIT License. See [LICENSE](LICENSE), which keeps the original copyright notice of Maziar Raissi.
