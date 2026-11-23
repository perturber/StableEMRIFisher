# Fisher matrix vs MCMC

`exec.py` checks the Fisher approximation for one injected source. It computes the Fisher matrix with StableEMRIFisher, samples the posterior of the same (noiseless) signal with the [Eryn](https://github.com/mikekatz04/Eryn) sampler, and compares the two covariances.

## Run

```bash
python exec.py injection_config_kerr_prograde_nonplunging_withresponse.py
```

Eryn is installed automatically (`eryn==1.2.6`) if it is missing.

## Injection config

Each `injection_config_*.py` file describes one run:

- **Source and model:** the FEW waveform model, the source parameters, the observation time `T` and sampling interval `dt`, and whether the LISA response is used (1st-generation TDI, A and E channels).
- **`DeltaT`:** p0 is chosen so that the source plunges at `T + DeltaT` (negative: during the observation).
- **`desired_snr`:** the distance is set to give this SNR.
- **Fisher matrix:** the derivative type and order, and the parameters to measure.
- **Sampler:** prior width, temperatures, walkers, steps, burn-in, move weights and seed. The prior is uniform, ±`prior_factor` Fisher standard deviations around the injection, clipped to the absolute limits in `prior_bounds` (e.g. the model's valid range of a spin). Half the moves (by default) are Eryn's stretch move, the rest a Gaussian move with the Fisher covariance.
- **`save_dir`:** the results folder, created next to the config file.

To add a run, copy a config, change it, and give it its own `save_dir`.

## Output

In `save_dir`:

- a copy of the config file;
- `Fisher.h5` and `stable_deltas.txt` from StableEMRIFisher;
- `<tag>.h5`: the Eryn chains at all temperatures, plus a `fisher` group with the Fisher matrix, covariance, injection and SNR;
- `<tag>.png`: a corner plot of the offsets from the injection: the MCMC posterior (cold chain) as filled 1σ and 2σ regions, the Fisher Gaussian as 1σ and 2σ ellipses, and σ_MCMC/σ_Fisher above each 1D panel. `python plot_results.py <tag>.h5` redraws it from a finished run.

`<tag>` names the model, response, derivative, `T`, `dt`, `DeltaT`, SNR and source parameters. The script also prints, for each parameter, the MCMC and Fisher standard deviations and their ratio.

## Metrics

`python metrics.py` summarises every run in the subfolders (or the `.h5` files given): the KL divergence and 2-Wasserstein distance between the MCMC posterior and the Fisher Gaussian (both as Gaussians, all parameters jointly), and per parameter the width ratio, the offset of the mean, the 1-Wasserstein distance of the marginal (sensitive to its shape) and a 1D KL divergence. All are computed where the Fisher Gaussian is a standard normal, so they are dimensionless. Each run is compared with a noise floor: the same metrics for as many independent draws of the Fisher Gaussian as the chain has effective samples. Each run's report is printed and saved as `metrics.txt` in its folder; with several runs, a summary table follows.

## Fisher convergence

`python fisher_convergence.py <injection_config>.py` recomputes the Fisher matrix of a finished run with the inspiral tolerance `err` = 1e-11 and 1e-13 and the stencil order `der_order` = 4 and 8, plus a fifth case (`err` = 1e-13, `der_order` = 8) whose step-size search for the masses and orbital parameters reaches down to 1e-12 of their value with 16 trial steps (default: 1e-9, 8 steps), and compares each with the run's MCMC samples. Cases already computed are reused. The injection is read from the run's `.h5` file, so all four describe exactly the source the MCMC sampled. Results go to `<save_dir>/fisher_convergence/`: a folder per case (`Fisher.h5`, `stable_deltas.txt` and the step-size stability plots), a corner plot per case, and `summary.txt` with σ_MCMC/σ_Fisher per parameter and the joint KL and W₂ of each case.
