"""Stable EMRI Fisher-matrix utilities.

This module provides the `StableEMRIFisher` class to compute signal-to-noise
ratio (SNR), select numerically stable finite-difference step sizes ("stable
deltas"), and build Fisher information matrices for Extreme Mass Ratio
Inspirals (EMRIs) using the FEW toolkit. It supports optional LISA response
wrapping, basic plunge checks, optional GPU acceleration via CuPy, and
convenience plotting/saving utilities.

Key capabilities:
- SNR computation from time-domain waveforms and generated PSDs.
- Automatic search for finite-difference step sizes that stabilize the
    diagonal Fisher elements.
- Fisher matrix assembly using inner products across one or more channels.
- Optional covariance matrix computation and diagnostic plots.

Notes
-----
- The waveform generator can be either a FEW `GenerateEMRIWaveform` instance
    or a `ResponseWrapper` that applies a LISA response to a base waveform.
- GPU support is best-effort; when enabled, arrays are promoted to CuPy where
    possible and converted back to NumPy for persistence and linear algebra
    operations that require CPU.
"""

import os
import sys
import time
import logging
from typing import Optional, Union, Dict, List, Tuple, Any, Callable, Type

import numpy as np
import h5py
import matplotlib.pyplot as plt

try:
    import cupy as cp

    ArrayType = Union[np.ndarray, cp.ndarray]
except ImportError:
    cp = None
    ArrayType = np.ndarray

from few.utils.constants import YRSID_SI
from few.waveform import GenerateEMRIWaveform
from stableemrifisher.fisher.derivatives import derivative, handle_a_flip
from stableemrifisher.fisher.stablederivative import StableEMRIDerivative
from stableemrifisher.utils import inner_product, SNRcalc, generate_PSD
from stableemrifisher.noise import sensitivity_LWA, write_psd_file, load_psd_from_file
from stableemrifisher.plot import CovEllipsePlot, StabilityPlot

logger = logging.getLogger("stableemrifisher")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel("INFO")
logger.info("startup")


class StableEMRIFisher:
    """Compute stable Fisher matrices for EMRI signals.

    This class orchestrates waveform generation (with or without a response),
    SNR calculation, stable step-size selection for numerical derivatives, and
    Fisher matrix assembly. It also provides optional covariance matrix
    computation and plotting, and basic file output convenience.

    Typical usage:
        1) Instantiate with physical parameters and a waveform generator.
        2) Call the instance to compute the Fisher matrix (and covariance if
           requested). Stable deltas are estimated automatically unless
           provided.

    Attributes:
        waveform: Cached waveform (channels x N).
        waveform_generator: `few.GenerateEMRIWaveform` or `fastlisaresponse.ResponseWrapper`.
        channels: Channel names used to build PSDs and products.
        deltas: Per-parameter finite-difference step sizes. Computed if not provided.
        param_names: Parameter names corresponding to Fisher order.
        npar: Number of parameters in the Fisher matrix.
        SNR2: SNR squared of the current waveform (set after call).
        dt: Time step for waveform generation.
        T: Total evolution time in years.
        use_gpu: Whether to use GPU acceleration.
        deriv_type: Type of derivative calculation ("stable" or "direct").
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
        deriv_type: str = "stable",
        stats_for_nerds: bool = False,
        use_gpu: bool = False,
        dt: float = 10.0,
        T: float = 1.0,
        # Fisher matrix computation defaults
        der_order: int = 2,
        Ndelta: int = 8,
        CovEllipse: bool = False,
        stability_plot: bool = False,
        save_derivatives: bool = False,
        live_dangerously: bool = False,
        filename: Optional[str] = None,
        plunge_check: bool = True,
        return_derivatives: bool = False,
        waveform_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a Fisher-matrix computation for an EMRI configuration.

        This configuration-only initializer sets up waveform/noise backends,
        derivative settings, and I/O options. Physical EMRI parameters are
        provided later when invoking the instance via `__call__`.

        Args:
            waveform_class: Uninitialized waveform class.
            waveform_class_kwargs: Optional kwargs for the waveform class.
            waveform_generator: Uninitialized waveform model class, defaults to `few.GenerateEMRIWaveform`.
            waveform_generator_kwargs: Optional kwargs for the waveform model.
            ResponseWrapper: Uninitialized response wrapper class, defaults to `None`.
            ResponseWrapper_kwargs: Optional kwargs for the response wrapper class.
            noise_model: Noise PSD function.
            noise_kwargs: Noise model kwargs. Defaults to {"TDI": "TDI1"}.
            channels: Channels to use. Defaults to ["A","E"].
            deriv_type: Type of derivative calculation ("stable" or "direct").
                "stable" uses `StableEMRIDerivatives`, "direct" uses `derivative`.
            stats_for_nerds: Enable verbose DEBUG logging.
            use_gpu: Prefer CuPy for array ops where available.
            der_order: Finite-difference order for derivatives.
            Ndelta: Number of trial deltas in stability search.
            CovEllipse: If True, compute covariance and plots by default.
            stability_plot: If True, plot stability curves by default.
            save_derivatives: If True, save derivative stacks to HDF5 by default.
            live_dangerously: If True, skip stability search and use heuristics by default.
            plunge_check: If True, trim evolution time if plunge is detected by default.
            return_derivatives: If True, return derivatives along with Fisher matrix by default.
            waveform_kwargs: Default kwargs for waveform generation.

        Raises:
            ValueError: If `deriv_type` is not "stable" or "direct".
        """
        # placeholders for attributes configured at call-time
        self.waveform = None
        self.wave_params = {}
        self.traj_params = {}
        self.wave_params_list = []
        self.SNR2 = None
        self.PSD_funcs = None

        self.use_gpu = use_gpu
        if self.use_gpu and cp is None:
            logger.warning("CuPy not found; disabling GPU acceleration.")
            self.use_gpu = False

        if stats_for_nerds:
            logger.setLevel("DEBUG")

        # =============== setup waveform kwargs ================
        if waveform_class_kwargs is None:
            waveform_class_kwargs = {}

        if waveform_generator_kwargs is None:
            waveform_generator_kwargs = {**waveform_class_kwargs}
        else:
            waveform_generator_kwargs = {
                **waveform_class_kwargs,
                **waveform_generator_kwargs,
            }  # if the two dicts have the same keys, the key value in the right side dict is used.

        if ResponseWrapper_kwargs is None:
            ResponseWrapper_kwargs = {}
        elif "waveform_gen" in ResponseWrapper_kwargs:
            logger.warning(
                "ResponseWrapper_kwargs should not contain 'waveform_gen'. It will be set automatically."
            )
            ResponseWrapper_kwargs.pop("waveform_gen")

        # ================== Initialize StableEMRIDerivatives ==================
        self.deriv_type = deriv_type
        if self.deriv_type == "stable":
            waveform_derivative = StableEMRIDerivative(
                waveform_class=waveform_class,
                **waveform_generator_kwargs,  # to pass to GenerateEMRIWaveforms
            )
            self.waveform_derivative_kwargs = {}
            # some utility funcs from SED useful later
            self._deltas = waveform_derivative._deltas
            self._stencil = waveform_derivative._stencil
        elif self.deriv_type == "direct":
            waveform_derivative = derivative
            self.waveform_derivative_kwargs = {"use_gpu": self.use_gpu}
        else:
            raise ValueError("deriv_type must be 'stable' or 'direct'.")

        # ================== Initialize waveform model ==================
        waveform_generator = waveform_generator(
            waveform_class=waveform_class,
            **waveform_generator_kwargs,
        )
        self.waveform_generator_kwargs = waveform_generator_kwargs  # This is the waveform generator without response to generate waveforms.

        # trajectory module and function for plunge checks
        self.traj_module = waveform_generator.waveform_generator.inspiral_generator
        self.traj_module_func = waveform_generator.waveform_generator.inspiral_kwargs[
            "func"
        ]

        # ================ Initialize ResponseWrapper if provided ==================
        if ResponseWrapper is not None:
            self.waveform_generator = ResponseWrapper(
                waveform_generator, **ResponseWrapper_kwargs
            )  # waveform generator with LISA response.
            if self.deriv_type == "direct":
                self.derivative = waveform_derivative
                self.waveform_derivative_kwargs.update(
                    {"waveform_generator": self.waveform_generator}
                )  # direct derivative waveform_generator with response.
            else:
                response_for_derivative = ResponseWrapper(
                    waveform_derivative, **ResponseWrapper_kwargs
                )  # this is the response wrapper to apply LISA response to the waveform derivative.
                self.derivative = response_for_derivative  # stable derivative wrapped with response. No kwargs needed. !! Does not include derivative of the response itself. !!
            self.has_ResponseWrapper = True
            self.T = ResponseWrapper_kwargs["Tobs"]
            self.dt = ResponseWrapper_kwargs["dt"]
        else:
            self.waveform_generator = (
                waveform_generator  # waveform generator without LISA response.
            )
            self.derivative = waveform_derivative  # either stable or direct derivative without response.
            self.T = T
            self.dt = dt
            if self.deriv_type == "direct":
                self.waveform_derivative_kwargs.update(
                    {"waveform_generator": self.waveform_generator}
                )  # direct derivative waveform_generator without response.
            self.has_ResponseWrapper = False

        # ================ Initialise Noise Model if provide =======================
        if noise_model is None and self.has_ResponseWrapper is True:
            logger.info("No noise model provided but response has been provided")
            logger.info("Generating and loading default PSD file")
            run_direc = os.getcwd()
            if ResponseWrapper_kwargs["tdi"] == "2nd generation":
                PSD_filename = "tdi2_wo_background.npy"
                kwargs_PSD = {
                    "stochastic_params": [T * YRSID_SI]
                }  # We include the background
                write_psd_file(
                    model="scirdv1",
                    channels="AE",
                    tdi2=True,
                    include_foreground=False,
                    filename=run_direc + PSD_filename,
                    **kwargs_PSD,
                )
                logger.info("\nTDI2 A and E with stochastic background.")
            else:
                PSD_filename = "tdi1_wo_background.npy"
                kwargs_PSD = {
                    "stochastic_params": [T * YRSID_SI]
                }  # We include the background
                write_psd_file(
                    model="scirdv1",
                    channels="AE",
                    tdi2=False,
                    include_foreground=False,
                    filename=run_direc + PSD_filename,
                    **kwargs_PSD,
                )
                logger.info("\nTDI1 A and E with stochastic background.")

            if self.use_gpu:
                self.noise_model = load_psd_from_file(run_direc + PSD_filename, xp=cp)
            else:
                self.noise_model = load_psd_from_file(run_direc + PSD_filename, xp=np)
            self.noise_kwargs = {}
            self.channels = ["A", "E"]
        elif noise_model is None and self.has_ResponseWrapper is False:
            logger.warning("No noise model or response wrapper provided.")
            logger.warning("Defaulting to the sky-averaged sensitivity curve")

            self.noise_model = sensitivity_LWA
            self.noise_kwargs = {}
            self.channels = channels if channels is not None else ["I", "II"]
        else:
            self.noise_model = noise_model
            self.noise_kwargs = (
                noise_kwargs if noise_kwargs is not None else {"TDI": "TDI1"}
            )
            self.channels = channels if channels is not None else ["A", "E"]

        # Bounds for directional derivatives near edges
        self.minmax = {
            "a": [0.05, 0.95],
            "e0": [0.01, 0.7],
            "Phi_phi0": [0.1, 2 * np.pi * 0.9],
            "Phi_r0": [0.1, 2 * np.pi * 0.9],
            "Phi_theta0": [0.1, 2 * np.pi * 0.9],
            "qS": [0.1, np.pi * 0.9],
            "qK": [0.1, np.pi * 0.9],
            "phiS": [0.1, 2 * np.pi * 0.9],
            "phiK": [0.1, 2 * np.pi * 0.9],
        }

        # Initialize Fisher matrix computation configuration
        self.order = der_order
        self.Ndelta = Ndelta
        self.CovEllipse = CovEllipse
        self.stability_plot = stability_plot
        self.save_derivatives = save_derivatives
        self.live_dangerously = live_dangerously
        self.plunge_check = plunge_check
        self.return_derivatives = return_derivatives

        # Initialize default waveform kwargs
        if waveform_kwargs is not None:
            self.waveform_kwargs = {**waveform_kwargs}
        else:
            self.waveform_kwargs = {}

        # Initialize attributes that will be set in __call__
        self.param_names: Optional[List[str]] = None
        self.npar: Optional[int] = None
        self.deltas: Optional[Dict[str, float]] = None
        self.current_waveform_kwargs: Optional[Dict[str, Any]] = None
        self.delta_range: Optional[Dict[str, Tuple[float, float]]] = None

        # Per-call configuration (can be overridden in __call__)
        self.window = None
        self.fmin = None
        self.fmax = None
        self.filename = None
        self.suffix = None
        # self.filename = filename   # Always set per-call

    def __call__(
        self,
        m1: float,
        m2: float,
        a: float,
        p0: float,
        e0: float,
        xI0: float,
        dist: float,
        qS: float,
        phiS: float,
        qK: float,
        phiK: float,
        Phi_phi0: float,
        Phi_theta0: float,
        Phi_r0: float,
        add_param_args: Optional[Dict[str, Any]] = None,
        waveform_kwargs: Optional[Dict[str, Any]] = None,
        window: Optional[Union[np.ndarray, Any]] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        param_names: Optional[List[str]] = None,
        deltas: Optional[Dict[str, float]] = None,
        der_order: Optional[int] = None,
        Ndelta: Optional[int] = None,
        delta_range: Optional[Dict[str, List[float]]] = None,
        CovEllipse: Optional[bool] = None,
        stability_plot: Optional[bool] = None,
        save_derivatives: Optional[bool] = None,
        return_derivatives: Optional[bool] = None,
        live_dangerously: Optional[bool] = None,
        plunge_check: Optional[bool] = None,
        filename: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Run the full pipeline at specific EMRI parameters.

        Workflow:
            1) Build the waveform and compute SNR (stored via `self.SNR2`).
            2) If `self.deltas` is None and `live_dangerously` is False,
               search for stable step sizes; otherwise use heuristics.
            3) Compute the Fisher matrix via inner products of derivatives.
            4) Optionally compute the covariance matrix and generate plots.

        Args:
            m1: Primary mass (solar masses).
            m2: Secondary mass (solar masses).
            a: Spin parameter [0, 1).
            p0: Initial separation (gravitational radii).
            e0: Initial eccentricity [0, 1).
            xI0: Initial inclination cosine [-1, 1].
            dist: Luminosity distance (Gpc).
            qS: Sky location polar angle [0, π].
            phiS: Sky location azimuthal angle [0, 2π].
            qK: Spin direction polar angle [0, π].
            phiK: Spin direction azimuthal angle [0, 2π].
            Phi_phi0: Initial azimuthal phase [0, 2π].
            Phi_theta0: Initial polar phase [0, 2π].
            Phi_r0: Initial radial phase [0, 2π].
            dt: Time step for waveform generation (seconds).
            T: Total evolution time (years).
            add_param_args: Additional model parameters to append.
            waveform_kwargs: Additional kwargs for waveform generation.
            window: Optional window to apply on the waveform.
            fmin: Minimum frequency for inner products.
            fmax: Maximum frequency for inner products.
            param_names: Ordered parameter names for derivatives.
            deltas: Optional fixed step sizes for derivatives.
            der_order: Finite-difference order for derivatives.
            Ndelta: Number of trial deltas in stability search.
            delta_range: Custom per-parameter delta grids.
            CovEllipse: If True, compute covariance and plots.
            stability_plot: If True, plot stability curves.
            save_derivatives: If True, save derivative stacks to HDF5.
            return_derivatives: If True, return derivatives along with Fisher matrix.
            live_dangerously: If True, skip stability search and use heuristics.
            plunge_check: If True, trim evolution time if plunge is detected.
            filename: Output directory for files.
            suffix: Optional suffix for output filenames.

        Returns:
            Fisher matrix (npar x npar), or (Fisher, Covariance) if `CovEllipse` is True.

        Raises:
            ValueError: If `param_names` is None or empty.
        """

        # initialize deltas (can be provided up-front)
        if deltas is not None and len(deltas) != self.npar:
            logger.critical(
                "Length of deltas array should be equal to length of param_names.\nAssuming deltas = None."
            )
            deltas = None
        self.deltas = deltas  # Use deltas == None as a Flag

        # Use defaults from __init__ but allow per-call overrides
        self.order = der_order if der_order is not None else self.order
        self.Ndelta = Ndelta if Ndelta is not None else self.Ndelta
        self.window = window  # Always set per-call
        self.fmin = fmin  # Always set per-call
        self.fmax = fmax  # Always set per-call
        self.CovEllipse = CovEllipse if CovEllipse is not None else self.CovEllipse
        self.stability_plot = (
            stability_plot if stability_plot is not None else self.stability_plot
        )
        self.save_derivatives = (
            save_derivatives if save_derivatives is not None else self.save_derivatives
        )
        self.filename = filename  # Always set per-call
        self.suffix = suffix  # Always set per-call
        self.live_dangerously = (
            live_dangerously if live_dangerously is not None else self.live_dangerously
        )
        self.plunge_check = (
            plunge_check if plunge_check is not None else self.plunge_check
        )
        self.return_derivatives = (
            return_derivatives
            if return_derivatives is not None
            else self.return_derivatives
        )

        # merge per-call waveform kwargs with existing defaults
        call_waveform_kwargs = dict(
            self.waveform_kwargs
        )  # Start with defaults from __init__
        if waveform_kwargs is not None:
            call_waveform_kwargs.update(
                waveform_kwargs
            )  # Override with per-call kwargs

        # ensure dt and T are passed to waveform generator
        call_waveform_kwargs.update({"dt": self.dt, "T": self.T})

        # Store for use throughout this call
        self.current_waveform_kwargs = call_waveform_kwargs

        # optional custom delta grids per parameter
        if delta_range is None:
            self.live_dangerously = True
            self.delta_range = {}
        else:
            self.delta_range = delta_range

        # initialize parameter dictionaries for this call
        self.wave_params = {
            "m1": m1,
            "m2": m2,
            "a": a,
            "p0": p0,
            "e0": e0,
            "xI0": xI0,
            "dist": dist,
            "qS": qS,
            "phiS": phiS,
            "qK": qK,
            "phiK": phiK,
            "Phi_phi0": Phi_phi0,
            "Phi_theta0": Phi_theta0,
            "Phi_r0": Phi_r0,
        }

        # initialize parameter name list
        if param_names is None:
            EMRI_ORBIT = (
                self.waveform_generator.waveform_gen.waveform_generator.descriptor
            )
            BACKGROUND = (
                self.waveform_generator.waveform_gen.waveform_generator.background
            )
            if EMRI_ORBIT == "eccentric equatorial" and BACKGROUND == "Kerr":
                param_names = [
                    "m1",
                    "m2",
                    "a",
                    "p0",
                    "e0",
                    "dist",
                    "qS",
                    "phiS",
                    "qK",
                    "phiK",
                    "Phi_phi0",
                    "Phi_r0",
                ]
            elif EMRI_ORBIT == "eccentric equatorial" and BACKGROUND == "Schwarzschild":
                param_names = [
                    "m1",
                    "m2",
                    "a",
                    "p0",
                    "e0",
                    "dist",
                    "qS",
                    "phiS",
                    "qK",
                    "phiK",
                    "Phi_phi0",
                    "Phi_r0",
                ]

        self.param_names = param_names
        self.npar = len(self.param_names)

        # trajectory params are the first six entries
        self.traj_params = dict(list(self.wave_params.items())[:6])

        # append any additional model parameters (optional)
        if add_param_args is not None:
            for k, v in add_param_args.items():
                self.wave_params[k] = v
                self.traj_params[k] = v

        self.wave_params_list = list(self.wave_params.values())

        # # Redefine final time if small body is plunging. More stable FMs.
        if self.plunge_check:
            final_time = self.check_if_plunging()
            self.T = final_time / YRSID_SI  # Years
            self.current_waveform_kwargs.update({"T": self.T})

        rho = self.SNRcalc_SEF(
            fmin=self.fmin,
            fmax=self.fmax,
            window=self.window,
            use_gpu=self.use_gpu,
            *self.wave_params_list,
            **self.current_waveform_kwargs,
        )

        self.SNR2 = rho**2

        logger.info("Waveform Generated. SNR: %s", rho)

        if rho <= 20.0:
            logger.critical(
                "The optimal source SNR is <= 20. The Fisher approximation may not be valid!"
            )

        # update derivative kwargs
        if self.deriv_type == "direct":
            self.waveform_derivative_kwargs.update(
                {
                    "parameters": self.wave_params,
                    "waveform": self.waveform,
                    "order": self.order,
                    "waveform_kwargs": self.current_waveform_kwargs,
                }
            )
        else:
            self.waveform_derivative_kwargs.update(
                {
                    "parameters": self.wave_params,
                    "order": self.order,
                    **self.current_waveform_kwargs,
                }
            )

        # making parent folder
        if self.filename is not None:
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)

        # 1. If deltas not provided, calculating the stable deltas
        if not self.live_dangerously:
            if self.deltas is None:
                start = time.time()
                self.Fisher_Stability()  # Attempts to compute stable delta values.
                end = time.time() - start
                logger.info("Time taken to compute stable deltas is %s seconds", end)
        else:
            logger.debug("You have elected for dangerous living, I like it. ")
            fudge_factor_intrinsic = (
                3
                * (self.wave_params["m2"] / self.wave_params["m1"])
                * (1 / self.SNR2) ** (1 / 2)
            )
            delta_intrinsic = fudge_factor_intrinsic * np.array(
                [self.wave_params["m1"], self.wave_params["m2"], 1.0, 1.0, 1.0, 1.0]
            )
            danger_delta_dict = dict(zip(self.param_names[0:7], delta_intrinsic))
            delta_dict_final_params = dict(
                zip(self.param_names[6:14], np.array(8 * [1e-6]))
            )
            danger_delta_dict.update(delta_dict_final_params)

            self.deltas = danger_delta_dict
            self.save_deltas()

        # 2. Given the deltas, we calculate the Fisher Matrix
        start = time.time()
        Fisher = self.FisherCalc()
        end = time.time() - start
        logger.info("Time taken to compute FM is %s seconds", end)

        # 3. If requested, calculate the covariance Matrix
        if self.CovEllipse:
            covariance = np.linalg.inv(Fisher)
            if self.filename is not None:
                if self.suffix is not None:
                    CovEllipsePlot(
                        covariance,
                        self.param_names,
                        self.wave_params,
                        filename=os.path.join(
                            self.filename, f"covariance_ellipses_{self.suffix}.png"
                        ),
                    )
                else:
                    CovEllipsePlot(
                        covariance,
                        self.param_names,
                        self.wave_params,
                        filename=os.path.join(self.filename, "covariance_ellipses.png"),
                    )
            else:
                CovEllipsePlot(covariance, self.param_names, self.wave_params)
                plt.show()
            return Fisher, covariance

        return Fisher

    def SNRcalc_SEF(
        self,
        *waveform_args,
        window=None,
        fmin=None,
        fmax=None,
        use_gpu=False,
        **waveform_kwargs,
    ):
        """Generate waveform and PSDs, then compute the optimal SNR.

        The waveform is obtained from `self.waveform_generator` using the
        parameters provided. If no response wrapper is used and a
        1D waveform is returned (h+ - i hx), it is replicated across the
        configured channels with equal weighting. Per-channel PSDs are then
        generated and the multi-channel SNR is computed.

        Returns:
            float: The optimal SNR of the current configuration.
        """
        # generate PSD
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        try:
            dt = waveform_kwargs["dt"]
        except KeyError as e:
            raise ValueError(f"waveform_kwargs must include {e}.") from e

        self.waveform = xp.asarray(
            self.waveform_generator(*waveform_args, **waveform_kwargs)
        )

        # If no response is provided and waveform of the form h+ - ihx, create copies equivalent to the number of channels.
        if not self.has_ResponseWrapper:
            self.waveform = xp.asarray([self.waveform.real, -self.waveform.imag])
        ### HEREAFTER, THE WAVEFORM HAS SHAPE (NCHANNELS, N) ###

        logger.debug("wave ndim: %s", self.waveform.ndim)
        # Generate PSDs
        self.PSD_funcs = generate_PSD(
            waveform=self.waveform,
            dt=dt,
            noise_PSD=self.noise_model,
            channels=self.channels,
            noise_kwargs=self.noise_kwargs,
            use_gpu=self.use_gpu,
        )

        # Compute SNR
        logger.info("Computing SNR for parameters: %s", waveform_args)

        return SNRcalc(
            self.waveform,
            self.PSD_funcs,
            dt=dt,
            window=window,
            fmin=fmin,
            fmax=fmax,
            use_gpu=use_gpu,
        )

    def check_if_plunging(self):
        """Check for plunge and return an adjusted evolution time (seconds).

        A short inspiral termination compared to the requested duration is a
        proxy for plunge. If detected, the final time is trimmed by six hours
        to improve numerical stability of subsequent Fisher calculations.

        Returns:
            float: Final evolution time in seconds (possibly reduced).
        """
        # Compute trajectory

        traj_vals = list(handle_a_flip(self.traj_params).values())
        t_traj = self.traj_module(
            *traj_vals,
            Phi_phi0=self.wave_params["Phi_phi0"],
            Phi_theta0=self.wave_params["Phi_theta0"],
            Phi_r0=self.wave_params["Phi_r0"],
            T=self.T,
            dt=self.dt,
        )[0]

        if (
            t_traj[-1] < self.T * YRSID_SI - 1.0
        ):  # 1.0 is a buffer because self.traj_module can produce trajectories slightly smaller than T*YRSID_SI even if not plunging!
            logger.warning("Body is plunging! Expect instabilities.")
            final_time = t_traj[-1] - 6 * 60 * 60  # Remove 6 hours of final inspiral
            logger.warning(
                "Removed last 6 hours of inspiral. New evolution time: %s years",
                final_time / YRSID_SI,
            )
        else:
            logger.info("Body is not plunging, Fisher should be stable.")
            final_time = self.T * YRSID_SI
        return final_time

    # defining Fisher_Stability function, generates self.deltas
    def Fisher_Stability(self):
        """Search per-parameter finite-difference steps that stabilize Gamma_ii.

        For each parameter in `self.param_names`, scan a geometric grid of
        trial step sizes (or use a user-provided grid via `delta_range`). For
        each trial delta, compute the derivative of the waveform and evaluate
        the corresponding diagonal Fisher element, Gamma_ii. A step size is
        selected by minimizing the relative change between successive Gamma_ii
        values. Results are stored in `self.deltas`. Optionally, stability
        plots are generated.

        Side effects:
            - Sets `self.deltas` to a dict[param_name] -> float.
            - Saves a `stable_deltas*.txt` file if `self.filename` is set.
        """
        if not self.use_gpu:
            xp = np
        else:
            xp = cp
        logger.info("calculating stable deltas...")
        Ndelta = self.Ndelta
        deltas = {}
        relerr_min = {}

        for param_name in self.param_names:

            try:
                delta_init = self.delta_range[param_name]

            except KeyError:

                # If a specific parameter equals zero, then consider stepsizes around zero.
                if self.wave_params[param_name] == 0.0:
                    delta_init = np.geomspace(1e-4, 1e-9, Ndelta)

                # Compute Ndelta number of delta values to compute derivative. Testing stability.
                elif param_name == "m1" or param_name == "m2":
                    delta_init = np.geomspace(
                        1e-4 * self.wave_params[param_name],
                        1e-9 * self.wave_params[param_name],
                        Ndelta,
                    )
                elif (
                    param_name == "a"
                    or param_name == "p0"
                    or param_name == "e0"
                    or param_name == "xI0"
                ):
                    delta_init = np.geomspace(
                        1e-4 * self.wave_params[param_name],
                        1e-9 * self.wave_params[param_name],
                        Ndelta,
                    )
                else:
                    delta_init = np.geomspace(
                        1e-1 * self.wave_params[param_name],
                        1e-6 * self.wave_params[param_name],
                        Ndelta,
                    )

            Gamma = []

            relerr_flag = False
            for k, delta_k in enumerate(delta_init):

                if param_name in list(self.minmax.keys()):
                    if (
                        self.wave_params[param_name]
                        <= self.minmax[param_name][0]
                    ):
                        kind = "forward"
                    elif (
                        self.wave_params[param_name]
                        > self.minmax[param_name][1]
                    ):
                        kind = "backward"
                    else:
                        kind = "central"
                else:
                    kind = "central"

                if param_name == "dist":
                    del_k = xp.asarray(
                        self.derivative(
                            *self.wave_params_list,
                            param_to_vary=param_name,
                            delta=delta_k,
                            kind=kind,
                            **self.waveform_derivative_kwargs,
                        )
                    )

                    relerr_flag = True
                    deltas["dist"] = 0.0
                    relerr_min["dist"] = 0.0
                    break

                if (param_name in ["Phi_phi0", "Phi_theta0", "Phi_r0"]) & (
                    self.deriv_type == "stable"
                ):
                    # derivatives are analytically available
                    del_k = xp.asarray(
                        self.derivative(
                            *self.wave_params_list,
                            param_to_vary=param_name,
                            delta=delta_k,
                            kind=kind,
                            **self.waveform_derivative_kwargs,
                        )
                    )
                    relerr_flag = True
                    deltas[param_name] = 0.0
                    relerr_min[param_name] = 0.0
                    break

                if (
                    (param_name in ["qS", "phiS", "qK", "phiK"])
                    & (self.deriv_type == "stable")
                    & (self.has_ResponseWrapper)
                ):
                    # cannot calculate derivative of the response-wrapped waveform with respect to 
                    # the angles for the stable deriv_type, so we use the direct derivative method.
                    if len(delta_init) == 1:
                        relerr_flag = True
                        deltas[param_name] = delta_k
                        break
                    deltas_grid = self._deltas(delta_k, self.order, kind=kind)
                    Rh_temp = xp.zeros(
                        (len(deltas_grid), len(self.waveform), len(self.waveform[0])),
                        dtype=xp.complex128,
                    )  # Ngrid x Nchannels x Nsamples
                    # calculate dR_dx
                    for dd, delt in enumerate(deltas_grid):
                        parameters_in = self.waveform_derivative_kwargs[
                            "parameters"
                        ].copy()
                        parameters_in[param_name] += float(
                            delt
                        )  # theta is of the same order as the other angles, so we use the same deltas.
                        parameters_in_list = list(parameters_in.values())
                        # get the ylms for this theta
                        Rh_temp[dd] = xp.asarray(
                            self.waveform_generator(
                                *parameters_in_list, **self.current_waveform_kwargs
                            )
                        )  # R[h] on the stencil grid

                    del_k = self._stencil(
                        Rh_temp, delta=delta_k, order=self.order, kind=kind
                    )  # derivative of R[h]

                else:
                    deltas[param_name] = delta_k
                    del_k = xp.asarray(
                        self.derivative(
                            *self.wave_params_list,
                            param_to_vary=param_name,
                            delta=delta_k,
                            kind=kind,
                            **self.waveform_derivative_kwargs,
                        )
                    )
                    if len(delta_init) == 1:
                        relerr_flag = True

                if not self.has_ResponseWrapper:
                    # If the derivative is 1D
                    del_k = xp.asarray([del_k.real, -del_k.imag])

                # Calculating the Fisher Elements
                Gammai = inner_product(
                    del_k,
                    del_k,
                    self.PSD_funcs,
                    self.dt,
                    window=self.window,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    use_gpu=self.use_gpu,
                )
                logger.debug("Gamma_ii for %s: %s", param_name, Gammai)
                if np.isnan(Gammai):
                    Gamma.append(0.0)  # handle nan's
                    logger.warning(
                        "NaN type encountered during Fisher calculation! Replacing with 0.0."
                    )
                else:
                    Gamma.append(Gammai)

            if not relerr_flag:
                if self.use_gpu:
                    Gamma = xp.asnumpy(xp.array(Gamma))
                else:
                    Gamma = xp.array(Gamma)

                if (Gamma[1:] == 0.0).all():  # handle non-contributing parameters
                    relerr = list(np.ones(len(Gamma) - 1))
                else:
                    relerr = []
                    for m in range(1, len(Gamma)):
                        if Gamma[m - 1] == 0.0:  # handle partially null contributors
                            relerr.append(1.0)
                        else:
                            relerr.append(np.abs(Gamma[m] - Gamma[m - 1]) / Gamma[m])

                logger.debug(relerr)

                relerr_min_i = relerr.index(min(relerr))

                logger.debug(relerr_min_i)

                if relerr[relerr_min_i] >= 0.01:
                    logger.warning(
                        "minimum relative error is greater than 1%% for %s. Fisher may be unstable!",
                        param_name,
                    )

                deltas_min_i = (
                    relerr_min_i + 1
                )  # +1 because relerr grid starts from Gamma_i index of 1 (not zero)
                deltas[param_name] = delta_init[deltas_min_i].item()
                relerr_min[param_name] = relerr[
                    relerr_min_i
                ]  # save the relerr minima. these can be used as error estimates on the FIM

                if self.stability_plot:
                    if self.filename is not None:
                        if self.suffix is not None:
                            StabilityPlot(
                                delta_init,
                                Gamma,
                                stable_index=deltas_min_i,
                                param_name=param_name,
                                filename=os.path.join(
                                    self.filename,
                                    f"stability_{self.suffix}_{param_name}.png",
                                ),
                            )
                        else:
                            StabilityPlot(
                                delta_init,
                                Gamma,
                                stable_index=deltas_min_i,
                                param_name=param_name,
                                filename=os.path.join(
                                    self.filename,
                                    f"stability_{param_name}.png",
                                ),
                            )
                    else:
                        StabilityPlot(
                            delta_init,
                            Gamma,
                            stable_index=deltas_min_i,
                            param_name=param_name,
                        )

        logger.debug("stable deltas: %s", deltas)

        self.deltas = deltas
        self.save_deltas()

    def save_deltas(self):
        """Persist the currently selected `self.deltas` to disk (if configured).

        Writes a small text file into `self.filename` containing the string
        representation of the `self.deltas` dictionary. If no output directory
        is configured, this function does nothing.
        """
        if self.filename is not None:
            if self.suffix is not None:
                with open(
                    f"{self.filename}/stable_deltas_{self.suffix}.txt",
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    file.write(str(self.deltas))
            else:
                with open(
                    f"{self.filename}/stable_deltas.txt",
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    file.write(str(self.deltas))

    # defining FisherCalc function, returns Fisher
    def FisherCalc(self):
        """Assemble the Fisher matrix using numerically differentiated waveforms.

        Uses the per-parameter step sizes in `self.deltas` to compute
        finite-difference derivatives of the waveform and evaluates inner
        products across channels using the precomputed PSD functions. The
        Fisher matrix is symmetrized, checked for degeneracies, and validated
        for positive definiteness (or semi-definiteness) via its inverse.

        Side effects:
            - Optionally saves derivative stacks and Fisher matrix to HDF5
              files in `self.filename`.

        Returns:
            numpy.ndarray: The Fisher matrix of shape (npar, npar).
        """
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        logger.info("calculating Fisher matrix...")

        Fisher = np.zeros((self.npar, self.npar), dtype=np.float64)
        dtv = []
        for i in range(self.npar):

            if self.param_names[i] in list(self.minmax.keys()):
                if (
                    self.wave_params[self.param_names[i]]
                    <= self.minmax[self.param_names[i]][0]
                ):
                    kind = "forward"
                elif (
                    self.wave_params[self.param_names[i]]
                    > self.minmax[self.param_names[i]][1]
                ):
                    kind = "backward"
                else:
                    kind = "central"
            else:
                kind = "central"

            if (
                (self.param_names[i] in ["qS", "phiS", "qK", "phiK"])
                & (self.deriv_type == "stable")
                & (self.has_ResponseWrapper)
            ):
                # cannot calculate derivative of the response-wrapped waveform with respect to the angles for the stable deriv_type, so we use the direct derivative method.
                deltas_grid = self._deltas(
                    self.deltas[self.param_names[i]], self.order, kind=kind
                )
                Rh_temp = xp.zeros(
                    (len(deltas_grid), len(self.waveform), len(self.waveform[0])),
                    dtype=self.waveform.dtype,
                )  # Ngrid x Nchannels x Nsamples
                # calculate dR_dx
                for dd, delt in enumerate(deltas_grid):
                    parameters_in = self.waveform_derivative_kwargs["parameters"].copy()
                    parameters_in[self.param_names[i]] += float(
                        delt
                    )  # theta is of the same order as the other angles, so we use the same deltas.
                    parameters_in_list = list(parameters_in.values())
                    # get the ylms for this theta
                    Rh_temp[dd] = xp.asarray(
                        self.waveform_generator(
                            *parameters_in_list, **self.current_waveform_kwargs
                        )
                    )  # R[h] on the stencil grid
                dtv_i = self._stencil(
                    Rh_temp,
                    delta=self.deltas[self.param_names[i]],
                    order=self.order,
                    kind=kind,
                )  # derivative of R[h]

            else:
                dtv_i = xp.asarray(
                    self.derivative(
                        *self.wave_params_list,
                        param_to_vary=self.param_names[i],
                        delta=self.deltas[self.param_names[i]],
                        kind=kind,
                        **self.waveform_derivative_kwargs,
                    )
                )

            if not self.has_ResponseWrapper:
                # If the derivative is 1D
                dtv_i = xp.asarray([dtv_i.real, -dtv_i.imag])

            dtv.append(dtv_i)

        logger.info("Finished derivatives")

        if self.save_derivatives:
            dtv_save = xp.asarray(dtv)
            if self.use_gpu:
                dtv_save = xp.asnumpy(dtv_save)
            if not self.filename is None:
                if not self.suffix is None:
                    with h5py.File(
                        f"{self.filename}/Fisher_{self.suffix}.h5", "w"
                    ) as f:
                        f.create_dataset("derivatives", data=dtv_save)
                else:
                    with h5py.File(f"{self.filename}/Fisher.h5", "w") as f:
                        f.create_dataset("derivatives", data=dtv_save)

        for i in range(self.npar):
            for j in range(i, self.npar):
                if self.use_gpu:
                    Fisher[i, j] = np.float64(
                        xp.asnumpy(
                            inner_product(
                                dtv[i],
                                dtv[j],
                                self.PSD_funcs,
                                self.dt,
                                window=self.window,
                                fmin=self.fmin,
                                fmax=self.fmax,
                                use_gpu=self.use_gpu,
                            ).real
                        )
                    )
                else:
                    Fisher[i, j] = np.float64(
                        (
                            inner_product(
                                dtv[i],
                                dtv[j],
                                self.PSD_funcs,
                                self.dt,
                                window=self.window,
                                fmin=self.fmin,
                                fmax=self.fmax,
                                use_gpu=self.use_gpu,
                            ).real
                        )
                    )

                # Exploiting symmetric property of the Fisher Matrix
                Fisher[j, i] = Fisher[i, j]

        # Check for degeneracies
        diag_elements = np.diag(Fisher)

        if 0 in diag_elements:
            logger.critical("Nasty. We have a degeneracy. Can't measure a parameter")
            degen_index = np.argwhere(diag_elements == 0)[0][0]
            Fisher[degen_index, degen_index] = 1.0

        # Check for positive-definiteness
        if "m1" in self.param_names:
            Fisher_inv = np.linalg.inv(Fisher)
        else:
            Fisher_inv = np.linalg.inv(Fisher)

        if (np.linalg.eigvals(Fisher_inv) < 0.0).any():
            logger.critical(
                "Calculated Fisher is not positive semi-definite. Try lowering inspiral error tolerance or increasing the derivative order."
            )
        else:
            logger.info("Calculated Fisher is *atleast* positive-definite.")

        if self.filename is None:
            pass
        else:
            if self.save_derivatives:
                mode = "a"  # append
            else:
                mode = "w"  # write new
            if self.suffix is not None:
                with h5py.File(f"{self.filename}/Fisher_{self.suffix}.h5", mode) as f:
                    f.create_dataset("Fisher", data=Fisher)
            else:
                with h5py.File(f"{self.filename}/Fisher.h5", mode) as f:
                    f.create_dataset("Fisher", data=Fisher)
        if self.return_derivatives is True:
            return dtv, Fisher
        else:
            return Fisher
