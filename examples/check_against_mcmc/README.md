# Fisher matrix vs MCMC

`exec.py` computes the Fisher matrix of one injected source with StableEMRIFisher, samples the posterior of the same noiseless signal with [Eryn](https://github.com/mikekatz04/Eryn), and compares the two.

```bash
python exec.py injection_config_kerr_prograde_nonplunging_withresponse.py
```

Each `injection_config_*.py` sets the waveform model, source parameters, observation time, LISA response, Fisher settings and sampler settings, and the results folder `save_dir`. The source plunges `DeltaT` years after the end of the observation (negative: during it), and its distance gives the SNR `desired_snr`. Priors are uniform over ±`prior_factor` Fisher standard deviations around the injection, clipped to `prior_bounds`.

The results folder holds a copy of the config, the Fisher matrix (`Fisher.h5`, `stable_deltas.txt`), the chains (`<tag>.h5`), a corner plot of the MCMC posterior against the Fisher Gaussian (`<tag>.png`) and `metrics.txt`.

- `python exec.py <injection_config>.py --fisher-only` recomputes the Fisher matrix of a finished run and updates its results, keeping the chains.
- `python plot_results.py <tag>.h5` redraws the corner plot.
- `python metrics.py` writes `metrics.txt` to every results folder: KL divergence and Wasserstein distances between the MCMC posterior and the Fisher Gaussian, with the noise floor of a finite chain.
