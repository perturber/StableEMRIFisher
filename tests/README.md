# StableEMRIFisher tests

Run the tests right after installing StableEMRIFisher, before you try the examples. They check that the installation works and that the results are correct on your machine.

From the root of the repository:

```bash
pytest            # fast tests (the default): about a minute on a laptop CPU
pytest -m slow    # fast and slow tests: adds the step-size search, plots and the LISA response
```

The first run also downloads FEW's data files, which needs an internet connection and can take a few minutes.

## Reading the output

The run starts with a report of your environment:

```
StableEMRIFisher 1.1.0 | FEW 2.0.0 | numpy 2.5.4 | Python 3.12.16 | Darwin arm64
GPU:      CuPy not installed. Linux x86_64 with an NVIDIA GPU: pip install -e ".[cuda12x]" (or ".[cuda11x]")
Response: lisaanalysistools 1.2.0, fastlisaresponse 1.1.17
```

- **GPU:** whether a GPU is available to both CuPy and FEW. Each test that can run on a GPU runs once on the CPU and once on the GPU. Without a GPU, the GPU runs are skipped.
- **Response:** whether the LISA response packages (`lisaanalysistools` and `fastlisaresponse`) are installed and can be loaded together, and whether they are the versions the examples need. Without them, the response tests are skipped.

The run ends with a list of what was **not tested**, with the command to install what is missing. A skipped GPU or response test is not a failure: it only means that part of StableEMRIFisher was not checked on your machine.

## What is tested

Wherever possible, results are compared with values known exactly in theory, not with numbers stored from an earlier run. Exact values don't change when FEW is updated, and a failure points directly at the problem.

The waveform tests use FEW's `FastKerrEccentricEquatorialFlux` model, with a short (0.01 yr) signal at SNR ≈ 40.

### Installation (`test_environment.py`)

- The installed version matches `pyproject.toml`. After changing the version, reinstall with `pip install -e .`.
- The citation example in the README and `CITATION.cff` name the version in `pyproject.toml`. After changing the version, update the version there too (the DOI stays the same).
- The public modules import.
- A GPU that CuPy can use also has a FEW CUDA backend.
- The LISA response packages load together in one process. The macOS wheels from PyPI crash when loaded together; see the README.

### Inner products (`test_inner_product.py`, `test_noise.py`)

- A sinusoid at a Fourier frequency in white noise has exactly (h|h) = A² N dt / S₀, and its SNR is the square root.
- Sinusoids at different frequencies, and sine and cosine at the same frequency, are orthogonal.
- The inner product is symmetric and linear, scales as 1/PSD, sums over channels, and gives an overlap of 1 for a signal with itself.
- A complex waveform h₊ − i h× gives the same result as the two channels [h₊, h×].
- `fmin`, `fmax`, `freq_mask` and `window` select and weight the right frequencies and samples.
- The Fisher matrix is the matrix of inner products of the derivatives.
- `generate_psd` returns one PSD per channel of the waveform, and rejects too few PSDs.

### Derivatives

- **Finite differences** (`test_finite_difference.py`): every stencil (central, forward, backward; orders 2 to 8) differentiates polynomials up to its order exactly, and the error of an order-n stencil shrinks as stepⁿ.
- **Angles** (`test_angles.py`): the analytic derivatives of the viewing angle and of the antenna patterns match finite differences.
- **Derivative interface** (`test_derivative_base.py`): the direct derivative of a toy waveform with known derivatives is correct for every stencil, and invalid input raises errors. These tests don't need FEW.
- **FEW waveforms** (`test_derivatives.py`):
  - The waveform the stable derivative builds internally is identical to FEW's (to 10⁻¹²), whichever way the modes are chosen: FEW's default, a call-time threshold, an explicit mode list, or a threshold given at initialization (which FEW ignores).
  - The stable and direct derivatives agree for every Fisher parameter of the model.
  - With `fix_modes` (the default), the direct derivative uses the modes of the reference waveform, even at a step where FEW would drop a mode.

### Fisher matrix (`test_fisher.py`)

- The distance element is exactly Γ_dist,dist = SNR² / dist², because the waveform is proportional to 1 / dist.
- The matrix is symmetric, reproducible, and equal to the inner products of the derivatives.
- The Fisher matrices from the stable and direct derivatives agree.
- Without the LISA response, exactly two combinations of the sky angles (qS, phiS, qK, phiK) are degenerate: the waveform depends on them only through the viewing angle and the polarization angle. All other parameters are constrained.
- The output files (Fisher matrix, derivatives, step sizes) are written and match the results.
- The plunge check shortens the observation time only when the body plunges.
- `fishinv` and `check_fisher` (degenerate, non-contributing and negative cases) behave as documented.

### GPU (`test_gpu.py`)

The GPU gives the same inner products, waveform and Fisher matrix as the CPU. These tests run only on a machine with a GPU.

### Slow tests

- **Step-size search** (`test_stepsize.py`): the search over all parameters converges (relative error below 1%), the Fisher matrix with the chosen steps satisfies the checks above, the steps reproduce it, and the stability and covariance plots are written.
- **LISA response** (`test_response.py`): with the A and E TDI channels, Γ_dist,dist = SNR² / dist² still holds exactly, and the stable and direct derivatives agree through the response.

## For contributors

- Every test is marked `fast` or `slow` (exactly one), with `pytestmark = pytest.mark.fast` at the top of the file or a mark on the test. A test without a mark stops the run with an error.
- Tests that can run on a GPU take the `backend` fixture. They then run on the CPU and, when available, on the GPU.
- The waveform models are listed in `cases.py`. To test a new model, add an entry: the tests use that model's default Fisher parameters automatically. A model the installed FEW doesn't provide is skipped, with the FEW version it needs.
- Shared fixtures and the environment report are in `conftest.py`.
