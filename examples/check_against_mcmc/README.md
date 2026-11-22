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
- `<tag>.png`: a corner plot of the posterior (cold chain) against the Fisher Gaussian.

`<tag>` names the model, response, derivative, `T`, `dt`, `DeltaT`, SNR and source parameters. The script also prints, for each parameter, the MCMC and Fisher standard deviations and their ratio.
