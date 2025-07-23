## Modified Custom Optimizer (RMSProp variant)

This model is a fork of the custom 2nd‑order optimizer from [relu‑based‑2ndOrder‑convergence](https://github.com/YunInSung/relu-based-2ndOrder-convergence), with the following changes:

- **Removed Hessian‑based dynamic learning rate computed from batch loss** now using a fixed learning rate 
- **Applied standard RMSProp gradient scaling**   
- Hyperparameters adjusted to:
  - `diff = 0.3`
  - `square = 5`
  - **Learning rate** fixed in the range **7.51e‑5 – 1e‑4**
- **Performance**: Even on a simple MLP architecture, this configuration consistently outperforms the original optimizer across training loss and convergence speed.

