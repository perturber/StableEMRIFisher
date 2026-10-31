"""Stable Fisher matrices for EMRI signals.

`StableEMRIFisher` runs the pipeline: it generates the waveform and its noise
PSD, computes the SNR, chooses stable finite-difference step sizes and
assembles the Fisher matrix, with optional covariance, plots and file output.
The pieces live elsewhere:

- `derivatives`: the waveform model and its derivatives (`DerivativeBase`).
- `fisher.stepsize`: the step-size search.
- `fisher.defaults`: default parameters, step-size grids and stencil bounds.
- `utils.analysis`: inner products, Fisher matrix and degeneracy checks.
- `utils.noise`: noise models and PSDs.
"""

import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import cupy as cp
except ImportError:
    cp = None

from few.waveform import GenerateEMRIWaveform

from .defaults import (
    STENCIL_BOUNDS,
    default_delta_grid,
    default_param_names,
    stencil_kind,
)
from stableemrifisher.derivatives import (
    DerivativeBase,
    DirectDerivative,
    get_derivative_class,
)
from .stepsize import stable_deltas
from stableemrifisher.utils.analysis import (
    optimal_snr,
    check_fisher,
    fisher_matrix_from_coefficients,
    fourier_coefficients,
    inner_product,
)
from stableemrifisher.utils.noise import generate_psd, resolve_noise_model
from stableemrifisher.utils.plot import plot_covariance_ellipses, plot_stability
from stableemrifisher.utils.utils import check_if_plunging

logger = logging.getLogger("stableemrifisher")


def _pick(value, default):
    """`value`, or `default` if `value` is None."""
    return default if value is None else value


class StableEMRIFisher:
    """Compute stable Fisher matrices for EMRI signals.

    The instance is configured once (waveform model, response, noise, derivative
    and output settings) and then called with the source parameters::

        sef = StableEMRIFisher(waveform_class=..., T=1.0, dt=10.0)
        Fisher = sef(wave_params, param_names=[...])

    Unless step sizes are given, they are chosen by `find_stable_deltas`.

    Attributes set by a call:

    - `deltas` (step size per parameter), `relerr` (relative error of each
      diagonal element at the chosen step), `fisher_check` (degeneracy
      report, see `utils.analysis.check_fisher`), `covariance` (if
      CovEllipse) and `derivatives` (if keep_derivatives).
    - `waveform` (channels x N), `PSD_funcs`, `SNR2` and `T` (observation
      time in years, shortened if the object plunges).

    Other attributes:

    - `derivative`: the `DerivativeBase` instance that builds the waveform
      model and computes its derivatives; `deriv_type`: its class.
    - `waveform_generator`: generator of the detector waveform (response
      included, if any), owned by `derivative`.
    - `noise_model`, `noise_kwargs`, `channels`: noise configuration.
    - `minmax`: per-parameter thresholds for one-sided stencils, see
      `defaults.STENCIL_BOUNDS`.
    """

    def __init__(
        self,
        *,
        waveform_class: Type,
        waveform_class_kwargs: Optional[Dict[str, Any]] = None,
        waveform_generator: Type = GenerateEMRIWaveform,
        waveform_generator_kwargs: Optional[Dict[str, Any]] = None,
        ResponseWrapper: Optional[Type] = None,
        ResponseWrapper_kwargs: Optional[Dict[str, Any]] = None,
        noise_model: Optional[Callable] = None,
        noise_kwargs: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None,
        deriv_type: Union[str, Type[DerivativeBase]] = "stable",
        use_gpu: bool = False,
        dt: float = 10.0,
        T: float = 1.0,
        # Fisher matrix computation defaults
        der_order: int = 2,
        Ndelta: int = 8,
        CovEllipse: bool = False,
        stability_plot: bool = False,
        save_derivatives: bool = False,
        filename: Optional[str] = None,
        plunge_check: bool = True,
        keep_derivatives: bool = False,
        waveform_kwargs: Optional[Dict[str, Any]] = None,
        degeneracy_rtol: float = 1e-12,
    ) -> None:
        """Configure the waveform model, response, noise and derivative.

        Args:
            waveform_class: Uninitialized FEW waveform class.
            waveform_class_kwargs: Optional kwargs for the waveform class.
            waveform_generator: Uninitialized waveform generator class,
                defaults to `few.GenerateEMRIWaveform`.
            waveform_generator_kwargs: Optional kwargs for the generator.
            ResponseWrapper: Uninitialized response wrapper class, defaults to
                None (no response).
            ResponseWrapper_kwargs: kwargs for the response wrapper. Its
                "Tobs" and "dt" override `T` and `dt`.
            noise_model: Noise PSD function. See
                `utils.noise.resolve_noise_model` for the defaults.
            noise_kwargs: Noise model kwargs.
            channels: Channel names.
            deriv_type: Derivative to use: "stable" (`StableDerivative`),
                "direct" (`DirectDerivative`), or any `DerivativeBase` subclass.
            use_gpu: Use CuPy arrays.
            dt: Sampling interval in seconds (without a response).
            T: Observation time in years (without a response).
            der_order: Finite-difference order for derivatives.
            Ndelta: Number of trial step sizes in the step-size search.
            CovEllipse: Compute the covariance matrix and plot its ellipses.
            stability_plot: Plot the step-size search for each parameter.
            save_derivatives: Save the derivatives to HDF5 (needs `filename`).
            filename: Output directory. None for no file output.
            plunge_check: Shorten T if the compact object plunges.
            keep_derivatives: Keep the derivatives in `self.derivatives`.
            waveform_kwargs: Default kwargs for waveform generation.
            degeneracy_rtol: Relative eigenvalue below which the (normalized)
                Fisher matrix is reported as degenerate, see
                `utils.analysis.check_fisher`.

        Raises:
            ValueError: If `deriv_type` is not a known name or a
                `DerivativeBase` subclass.
        """
        self.use_gpu = use_gpu
        if self.use_gpu and cp is None:
            logger.warning("CuPy not found; disabling GPU acceleration.")
            self.use_gpu = False

        # =============== setup waveform kwargs ================
        # for keys in both dicts, the generator kwargs win
        waveform_generator_kwargs = {
            **(waveform_class_kwargs or {}),
            **(waveform_generator_kwargs or {}),
        }
        self.waveform_generator_kwargs = waveform_generator_kwargs

        ResponseWrapper_kwargs = dict(ResponseWrapper_kwargs or {})
        if "waveform_gen" in ResponseWrapper_kwargs:
            logger.warning(
                "ResponseWrapper_kwargs should not "
                "contain 'waveform_gen'. It will be set automatically."
            )
            ResponseWrapper_kwargs.pop("waveform_gen")

        # ================== Initialize the derivative ==================
        # The derivative builds and owns the waveform model (with the response,
        # if any), which is also used here to generate the waveform.
        deriv_class = get_derivative_class(deriv_type)
        if not deriv_class.supports_waveform(waveform_class):
            logger.warning(
                "%s does not support the %s waveform model. "
                "Switching to DirectDerivative.",
                deriv_class.__name__,
                waveform_class.__name__,
            )
            deriv_class = DirectDerivative
        self.deriv_type = deriv_class
        self.derivative = deriv_class(
            waveform_class=waveform_class,
            waveform_generator=waveform_generator,
            waveform_generator_kwargs=waveform_generator_kwargs,
            ResponseWrapper=ResponseWrapper,
            ResponseWrapper_kwargs=ResponseWrapper_kwargs,
            use_gpu=self.use_gpu,
        )
        self.waveform_generator = self.derivative.waveform_generator
        self.has_ResponseWrapper = self.derivative.has_response

        # trajectory module and function for plunge checks
        base_waveform = self.derivative.base_generator.waveform_generator
        self.traj_module = base_waveform.inspiral_generator
        self.traj_module_func = base_waveform.inspiral_kwargs["func"]

        if self.has_ResponseWrapper:
            self.T = ResponseWrapper_kwargs["Tobs"]
            self.dt = ResponseWrapper_kwargs["dt"]
        else:
            self.T = T
            self.dt = dt
        # requested observation time; self.T may be shortened per call
        self._T_requested = self.T

        # ================ Noise model =======================
        self.noise_model, self.noise_kwargs, self.channels = resolve_noise_model(
            noise_model,
            noise_kwargs,
            channels,
            ResponseWrapper_kwargs=(
                ResponseWrapper_kwargs if self.has_ResponseWrapper else None
            ),
            use_gpu=self.use_gpu,
        )

        self.minmax = {key: list(value) for key, value in STENCIL_BOUNDS.items()}

        # Fisher matrix computation defaults. Each call can override them; the
        # settings in use are stored on the instance (self.order, ...).
        self._defaults = dict(
            order=der_order,
            Ndelta=Ndelta,
            CovEllipse=CovEllipse,
            stability_plot=stability_plot,
            save_derivatives=save_derivatives,
            keep_derivatives=keep_derivatives,
            plunge_check=plunge_check,
            filename=filename,
        )
        self.order = der_order
        self.Ndelta = Ndelta
        self.kind = "central"
        self.CovEllipse = CovEllipse
        self.stability_plot = stability_plot
        self.save_derivatives = save_derivatives
        self.plunge_check = plunge_check
        self.keep_derivatives = keep_derivatives
        self.degeneracy_rtol = degeneracy_rtol
        self.waveform_kwargs = dict(waveform_kwargs or {})
        self.filename = filename
        self.suffix = None
        self.window = None
        self.fmin = None
        self.fmax = None
        self.freq_mask = None
        self.delta_range = {}

        # set by __call__
        self.wave_params = {}
        self.traj_params = {}
        self.wave_params_list = []
        self.param_names: Optional[List[str]] = None
        self.npar: Optional[int] = None
        self.current_waveform_kwargs: Optional[Dict[str, Any]] = None
        self.waveform = None
        self.PSD_funcs = None
        self.SNR2 = None
        self.deltas: Optional[Dict[str, float]] = None
        self.relerr: Optional[Dict[str, float]] = None
        self.fisher_check = None
        self.stats_for_nerds = False
        self.derivatives = None
        self.covariance = None

    def __call__(
        self,
        wave_params: Dict[str, float],
        add_param_args: Optional[Dict[str, Any]] = None,
        waveform_kwargs: Optional[Dict[str, Any]] = None,
        window: Optional[Any] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        freq_mask: Optional[np.ndarray] = None,
        param_names: Optional[List[str]] = None,
        deltas: Optional[Dict[str, float]] = None,
        der_order: Optional[int] = None,
        kind: Optional[str] = None,
        Ndelta: Optional[int] = None,
        delta_range: Optional[Dict[str, List[float]]] = None,
        CovEllipse: Optional[bool] = None,
        stability_plot: Optional[bool] = None,
        save_derivatives: Optional[bool] = None,
        keep_derivatives: Optional[bool] = None,
        plunge_check: Optional[bool] = None,
        filename: Optional[str] = None,
        suffix: Optional[str] = None,
        stats_for_nerds: bool = False,
    ) -> np.ndarray:
        """Compute the Fisher matrix at the given source parameters.

        Steps:
            1) Shorten the observation time if the object plunges (optional).
            2) Generate the waveform and PSDs and compute the SNR.
            3) Choose stable step sizes, unless `deltas` is given.
            4) Compute the Fisher matrix and report degeneracies.
            5) Optionally compute the covariance matrix and plot it.

        Arguments left as None take the value given at initialization.

        Args:
            wave_params: Source parameters, in the order the waveform
                generator expects them (m1, m2, a, p0, e0, xI0, dist, qS, phiS,
                qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 for FEW models).
            add_param_args: Additional model parameters, appended to
                `wave_params` (and passed to the trajectory).
            waveform_kwargs: Additional kwargs for waveform generation.
            window: Window applied to the time series in inner products.
            fmin, fmax: Frequency range (Hz) of the inner products.
            freq_mask: Boolean frequency mask; overrides fmin and fmax.
            param_names: Fisher matrix parameters, in order. Defaults to
                `defaults.default_param_names` for the waveform model.
            deltas: Step size for every parameter in `param_names`. Skips the
                step-size search.
            der_order: Finite-difference order.
            kind: Default stencil kind, "central" (default), "forward" or
                "backward".
            Ndelta: Number of trial step sizes in the search.
            delta_range: Trial step sizes per parameter, replacing the default
                grid of `defaults.default_delta_grid`.
            CovEllipse: Compute the covariance matrix and plot its ellipses.
            stability_plot: Plot the step-size search.
            save_derivatives: Save the derivatives to HDF5.
            keep_derivatives: Keep the derivatives in `self.derivatives`.
            plunge_check: Shorten T if the object plunges.
            filename: Output directory.
            suffix: Suffix for output file names.
            stats_for_nerds: Print the step size chosen for each parameter,
                and its relative error, during the step-size search. For
                full debugging output, set the "stableemrifisher" logger to
                DEBUG: logging.getLogger("stableemrifisher").setLevel("DEBUG").

        Returns:
            np.ndarray: the Fisher matrix, shape (npar, npar). The covariance
            matrix (if CovEllipse) and the derivatives (if keep_derivatives)
            are stored in `self.covariance` and `self.derivatives`.
        """
        # per-call settings, falling back on those given at initialization
        defaults = self._defaults
        self.order = _pick(der_order, defaults["order"])
        self.Ndelta = _pick(Ndelta, defaults["Ndelta"])
        self.kind = _pick(kind, "central")
        self.window = window
        self.fmin = fmin
        self.fmax = fmax
        self.freq_mask = freq_mask
        if freq_mask is not None:
            # custom freq_mask overrides fmin and fmax.
            self.fmin = None
            self.fmax = None
        self.CovEllipse = _pick(CovEllipse, defaults["CovEllipse"])
        self.stability_plot = _pick(stability_plot, defaults["stability_plot"])
        self.save_derivatives = _pick(save_derivatives, defaults["save_derivatives"])
        self.keep_derivatives = _pick(keep_derivatives, defaults["keep_derivatives"])
        self.plunge_check = _pick(plunge_check, defaults["plunge_check"])
        self.stats_for_nerds = stats_for_nerds
        self.filename = _pick(filename, defaults["filename"])
        self.suffix = suffix
        self.delta_range = _pick(delta_range, {})
        self.derivatives = None
        self.covariance = None

        # parameters
        self.wave_params = wave_params
        if param_names is None:
            base_waveform = self.derivative.base_generator.waveform_generator
            logger.debug(
                "EMRI_ORBIT: %s, BACKGROUND: %s",
                base_waveform.descriptor,
                base_waveform.background,
            )
            param_names = default_param_names(base_waveform)
        self.param_names = list(param_names)
        self.npar = len(self.param_names)

        if deltas is not None:
            missing = [p for p in self.param_names if p not in deltas]
            if missing:
                logger.critical(
                    "deltas has no step size for %s. Assuming deltas = None.",
                    missing,
                )
                deltas = None
        self.deltas = dict(deltas) if deltas is not None else None
        self.relerr = None

        # trajectory params are the first six entries
        self.traj_params = dict(list(self.wave_params.items())[:6])

        # append any additional model parameters (optional)
        if add_param_args is not None:
            for k, v in add_param_args.items():
                self.wave_params[k] = v
                self.traj_params[k] = v

        self.wave_params_list = list(self.wave_params.values())

        # 1. Shorten the observation time if the small body plunges.
        self.T = self._T_requested
        if self.plunge_check:
            self.T = check_if_plunging(
                self.traj_module,
                self.T,
                *self.traj_params.values(),
                Phi_phi0=self.wave_params["Phi_phi0"],
                Phi_theta0=self.wave_params["Phi_theta0"],
                Phi_r0=self.wave_params["Phi_r0"],
                chop_inspiral_time=6.0,
                dt=self.dt,
            )

        # per-call waveform kwargs on top of the defaults; dt and T always set
        self.current_waveform_kwargs = {
            **self.waveform_kwargs,
            **(waveform_kwargs or {}),
            "dt": self.dt,
            "T": self.T,
        }

        # 2. Waveform, PSDs and SNR
        rho = self.compute_snr(
            *self.wave_params_list,
            window=self.window,
            fmin=self.fmin,
            fmax=self.fmax,
            freq_mask=self.freq_mask,
            use_gpu=self.use_gpu,
            **self.current_waveform_kwargs,
        )
        self.SNR2 = rho**2
        logger.info("Waveform Generated. SNR: %s", rho)
        if rho <= 20.0:
            logger.critical(
                "The optimal source SNR is <= 20. "
                "The Fisher approximation may not be valid!"
            )

        # derivatives are computed at the injection
        self.derivative.set_reference(
            self.wave_params, self.waveform, **self.current_waveform_kwargs
        )

        if self.filename is not None:
            os.makedirs(self.filename, exist_ok=True)

        # 3. Step sizes
        if self.deltas is None:
            start = time.time()
            self.find_stable_deltas()
            logger.debug(
                "Time taken to compute stable deltas is %s seconds",
                time.time() - start,
            )

        # 4. Fisher matrix
        start = time.time()
        Fisher = self.compute_fisher()
        logger.debug("Time taken to compute FM is %s seconds", time.time() - start)

        # 5. Covariance matrix
        if self.CovEllipse:
            if not self.fisher_check["positive_definite"]:
                logger.error(
                    "Covariance not computed: the Fisher matrix is not "
                    "positive definite (see the messages above)."
                )
            else:
                self.covariance = np.linalg.inv(Fisher)
                path = self._output_path("covariance_ellipses", ".png")
                if path is not None:
                    plot_covariance_ellipses(
                        self.covariance, self.param_names, self.wave_params, filename=path
                    )
                else:
                    plot_covariance_ellipses(self.covariance, self.param_names, self.wave_params)
                    plt.show()

        return Fisher

    def compute_snr(
        self,
        *waveform_args,
        window=None,
        fmin=None,
        fmax=None,
        freq_mask=None,
        use_gpu=False,
        **waveform_kwargs,
    ):
        """Generate the waveform and the PSDs, then compute the optimal SNR.

        Sets `self.waveform` (shape (channels, N); without a response,
        h+ - i hx is split into [h+, hx]) and `self.PSD_funcs`.

        Args:
            *waveform_args: Waveform parameters, in generator order.
            window, fmin, fmax, freq_mask: as in `utils.analysis.inner_product`.
            use_gpu: Use CuPy arrays.
            **waveform_kwargs: kwargs for the waveform generator; must
                include dt.

        Returns:
            float: The optimal SNR.
        """
        try:
            dt = waveform_kwargs["dt"]
        except KeyError as e:
            raise ValueError(f"waveform_kwargs must include {e}.") from e

        self.waveform = self.derivative.generate(*waveform_args, **waveform_kwargs)
        logger.debug("waveform shape: %s", self.waveform.shape)

        self.PSD_funcs = generate_psd(
            waveform=self.waveform,
            dt=dt,
            noise_PSD=self.noise_model,
            channels=self.channels,
            noise_kwargs=self.noise_kwargs,
            use_gpu=self.use_gpu,
        )

        logger.debug("Computing SNR for parameters: %s", waveform_args)
        return optimal_snr(
            self.waveform,
            self.PSD_funcs,
            dt=dt,
            window=window,
            fmin=fmin,
            fmax=fmax,
            freq_mask=freq_mask,
            use_gpu=use_gpu,
        )

    def find_stable_deltas(self):
        """Choose the step size of each parameter with `stepsize.stable_deltas`.

        The trial steps come from `delta_range`, or else from
        `defaults.default_delta_grid`. Sets `self.deltas` and `self.relerr`,
        plots the search if `stability_plot`, and saves the step sizes if an
        output directory is set.
        """
        grids = {
            p: (
                self.delta_range[p]
                if p in self.delta_range
                else default_delta_grid(p, self.wave_params[p], self.Ndelta)
            )
            for p in self.param_names
        }
        kinds = {p: self._stencil_kind(p) for p in self.param_names}

        self.deltas, self.relerr, curves = stable_deltas(
            self.derivative,
            grids,
            self._inner,
            self.order,
            kinds,
            verbose=self.stats_for_nerds,
        )

        if self.stability_plot:
            for param_name, (grid, Gamma, index) in curves.items():
                plot_stability(
                    grid,
                    Gamma,
                    stable_index=index,
                    param_name=param_name,
                    filename=self._output_path("stability", ".png", tag=param_name),
                )

        self.save_deltas()

    def save_deltas(self):
        """Write `self.deltas` to stable_deltas[_suffix].txt in the output directory."""
        path = self._output_path("stable_deltas", ".txt")
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as file:
                file.write(str(self.deltas))

    def compute_fisher(self):
        """Compute the derivatives and the Fisher matrix, and check it.

        Uses the step sizes in `self.deltas`. Each derivative is Fourier
        transformed once, as soon as it is computed, and the Fisher matrix is
        assembled from the Fourier coefficients (see
        `utils.analysis.fisher_matrix_from_coefficients`); the derivatives
        themselves are only kept if `keep_derivatives` or `save_derivatives`.
        Degeneracies are reported (see `utils.analysis.check_fisher`) but the
        matrix is not modified. The derivatives and the Fisher matrix are
        written to Fisher[_suffix].h5 if an output directory is set.

        Returns:
            np.ndarray: the Fisher matrix, shape (npar, npar).
        """
        path = self._output_path("Fisher", ".h5")
        keep = self.keep_derivatives or (self.save_derivatives and path is not None)

        coefficients = []
        derivatives = []
        with tqdm(
            total=self.npar,
            desc="Computing Fisher   ",
            unit="deriv",
            file=sys.stdout,
        ) as bar:
            for param_name in self.param_names:
                bar.set_postfix_str(param_name)
                derivative = self.derivative(
                    param_name,
                    self.deltas[param_name],
                    order=self.order,
                    kind=self._stencil_kind(param_name),
                )
                coefficients.append(
                    fourier_coefficients(
                        derivative,
                        self.dt,
                        window=self.window,
                        fmin=self.fmin,
                        fmax=self.fmax,
                        freq_mask=self.freq_mask,
                        use_gpu=self.use_gpu,
                    )
                )
                if keep:
                    derivatives.append(derivative)
                bar.update()
            bar.set_postfix_str("")

        if self.save_derivatives and path is not None:
            derivatives_save = (cp if self.use_gpu else np).asarray(derivatives)
            if self.use_gpu:
                derivatives_save = cp.asnumpy(derivatives_save)
            with h5py.File(path, "w") as f:
                f.create_dataset("derivatives", data=derivatives_save)

        Fisher = fisher_matrix_from_coefficients(
            coefficients, self.PSD_funcs, use_gpu=self.use_gpu
        )
        if self.keep_derivatives:
            self.derivatives = derivatives

        self._check_fisher(Fisher)

        if path is not None:
            mode = "a" if self.save_derivatives else "w"
            with h5py.File(path, mode) as f:
                f.create_dataset("Fisher", data=Fisher)

        return Fisher

    def _check_fisher(self, Fisher):
        """Report degeneracies of the Fisher matrix; sets `self.fisher_check`."""
        check = check_fisher(Fisher, self.param_names, rtol=self.degeneracy_rtol)
        for param_name in check["no_effect"]:
            logger.warning(
                "%s has no effect on the waveform (zero Fisher diagonal element). "
                "The Fisher matrix is not invertible.",
                param_name,
            )
        for params, rel_eigenvalue in check["degenerate"]:
            logger.warning(
                "Degenerate parameters: a combination of %s is unconstrained "
                "(relative eigenvalue %.1e). The Fisher matrix is not invertible.",
                ", ".join(params),
                rel_eigenvalue,
            )
        if check["negative"]:
            logger.critical(
                "Calculated Fisher is not positive "
                "semi-definite. "
                "Try lowering inspiral error tolerance "
                "or increasing the derivative order."
            )
        if check["positive_definite"]:
            logger.info("Calculated Fisher is positive-definite.")
        self.fisher_check = check
        return check

    def _inner(self, a, b):
        """Inner product with the PSDs and frequency settings of this call."""
        return inner_product(
            a,
            b,
            self.PSD_funcs,
            self.dt,
            window=self.window,
            fmin=self.fmin,
            fmax=self.fmax,
            freq_mask=self.freq_mask,
            use_gpu=self.use_gpu,
        )

    def _stencil_kind(self, param_name):
        """Stencil kind for a parameter, see `defaults.stencil_kind`."""
        return stencil_kind(
            param_name, self.wave_params[param_name], self.kind, self.minmax
        )

    def _output_path(self, stem, ext, tag=None):
        """Path of an output file, or None if there is no output directory.

        The name is stem[_suffix][_tag]ext.
        """
        if self.filename is None:
            return None
        name = stem
        if self.suffix is not None:
            name += f"_{self.suffix}"
        if tag is not None:
            name += f"_{tag}"
        return os.path.join(self.filename, name + ext)
