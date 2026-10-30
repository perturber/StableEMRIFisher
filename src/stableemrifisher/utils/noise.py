"""
This module provides noise modeling functionality for LISA (Laser Interferometer Space Antenna) gravitational wave detector.

Key components:
- Sensitivity functions for LISA using long-wavelength approximation and TDI (Time-Delay Interferometry) configurations
- Power spectral density (PSD) generation and management for different noise models and channel configurations
- File I/O utilities for saving and loading pre-computed PSD data with interpolation capabilities
- Support for both CPU and GPU backends via configurable array libraries

"""

import logging
import os

import numpy as np
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import YRSID_SI

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger("stableemrifisher")

# lisatools (pip: lisaanalysistools) is an optional dependency. write_psd_file and load_psd
# are only defined if it imports and a test sensitivity evaluation succeeds.
try:
    from lisatools.sensitivity import (
        get_sensitivity,
        A1TDISens,
        E1TDISens,
        T1TDISens,
        LISASens,
    )
    from lisatools.utils.constants import lisaLT

    _test_psd = get_sensitivity(
        np.array([1e-3, 1e-2]), sens_fn=A1TDISens, model="scirdv1", return_type="PSD"
    )
    if not np.all(np.isfinite(_test_psd)):
        raise RuntimeError("lisatools test sensitivity evaluation returned non-finite values")
    del _test_psd

    LISATOOLS_AVAILABLE = True
except Exception:
    LISATOOLS_AVAILABLE = False


def sensitivity_lwa(f, galactic_confusion=False, **kwargs):
    """
    LISA sensitivity function in the long-wavelength approximation (https://arxiv.org/pdf/1803.01944.pdf).

    args:
        f (float): LISA-band frequency of the signal

    Returns:
        The output sensitivity strain Sn(f)
    """

    # Defining supporting functions
    L = 2.5e9  # m
    fstar = 19.09e-3  # Hz

    P_OMS = (1.5e-11**2) * (1 + (2e-3 / f) ** 4)  # Hz-1
    P_acc = (3e-15**2) * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)  # Hz-1

    # S_c changes depending on signal duration (Equation 14 in 1803.01944)
    # for 1 year
    alpha = 0.171
    beta = 292
    kappa = 1020
    gamma = 1680
    fk = 0.00215
    # log10_Sc = (np.log10(9)-45) -7/3*np.log10(f) -(f*alpha + beta*f*np.sin(kappa*f))*np.log10(np.e) + np.log10(1 + np.tanh(gamma*(fk-f))) #Hz-1

    A = 9e-45

    sensitivity = (10 / (3 * L**2)) * (
            P_OMS + 4 * (P_acc) / ((2 * np.pi * f) ** 4)
        ) * (1 + 6 * f**2 / (10 * fstar**2))

    if galactic_confusion:
        Sc = (
            A
            * f ** (-7 / 3)
            * np.exp(-(f**alpha) + beta * f * np.sin(kappa * f))
            * (1 + np.tanh(gamma * (fk - f)))
        )

        return sensitivity + Sc

    return sensitivity


def generate_psd(
    waveform,
    dt,
    noise_PSD=sensitivity_lwa,
    channels=None,
    noise_kwargs=None,
    use_gpu=False,
):
    """
    Evaluate the noise PSD of each channel on the frequency grid of a waveform.

    The PSD is evaluated at the positive frequencies of the real FFT of the
    waveform, np.fft.rfftfreq(N, dt)[1:] (the f = 0 bin is left out), which is
    the grid `inner_product` expects.

    `noise_PSD` is called once per dict in `noise_kwargs` (or once, if it is a
    single dict). Each call may return one channel (a 1D array) or several
    channels at once (a 2D array, one row per channel, like the interpolant
    from `load_psd_from_file`). A single channel in total is used for every
    channel; otherwise there must be at least one PSD per channel, and the
    first len(channels) are used.

    Args:
        waveform (array): the waveform, shape (N_channels, N). Only its
            length N is used.
        dt (float): sampling interval of the waveform in seconds.
        noise_PSD (callable): function of frequency (Hz) returning the
            one-sided PSD. Default `sensitivity_lwa`.
        channels (list): channel names; only their number is used.
            Default ["A", "E"].
        noise_kwargs (dict or list of dict): kwargs passed to `noise_PSD`.
            Default {"TDI": "TDI1"}.
        use_gpu (bool): return CuPy arrays. Default False.

    Returns:
        list of arrays: the PSD of each channel, each of length N // 2.

    Raises:
        ValueError: if `noise_PSD` gives fewer PSDs than there are channels.
    """
    if channels is None:
        channels = ["A", "E"]
    if noise_kwargs is None:
        noise_kwargs = {"TDI": "TDI1"}
    xp = cp if use_gpu else np

    # Extract fourier frequencies
    length = waveform.shape[-1]
    freq = xp.fft.rfftfreq(length) / dt

    if use_gpu:
        freq_np = freq.get()
    else:
        freq_np = freq

    kwargs_list = noise_kwargs if isinstance(noise_kwargs, list) else [noise_kwargs]
    PSD = []
    for kwargs in kwargs_list:
        psd = xp.asarray(noise_PSD(freq_np[1:], **kwargs))
        # a 2D result holds one channel per row
        PSD.extend(list(psd) if psd.ndim == 2 else [psd])

    if len(PSD) == 1:
        PSD = PSD * len(channels)
    if len(PSD) < len(channels):
        raise ValueError(
            f"noise_PSD gave {len(PSD)} PSDs for {len(channels)} channels "
            f"({channels})."
        )
    return PSD[: len(channels)]


def load_psd_from_file(psd_file, xp=np):
    """
    Load the PSD from a file and return an interpolant.

    Parameters
    ----------
    psd_file : str
        The name of the file to load the PSD from.
    xp : module
        The module to use for array operations. Default is np.

    Returns
    -------
    psd_clipped : function
        A function that takes a frequency and returns the PSD at that frequency.
    """

    psd_in = np.load(psd_file).T
    freqs, values = psd_in[0], np.atleast_2d(psd_in[1:])

    # convert to cupy if needed
    freqs = xp.asarray(freqs)
    values = xp.asarray(values)

    backend = "cpu" if xp is np else "gpu"
    logger.debug("Using %s backend for PSD interpolation", backend)

    min_psd = np.min(
        values[:, freqs < 1e-2], axis=1
    )  # compatible with both tdi 1 and tdi 2
    max_psd = np.max(values, axis=1)
    psd_interp = CubicSplineInterpolant(freqs, values, force_backend=backend)

    def psd_clipped(f, **kwargs):
        f = xp.clip(f, 0.00001, 1.0)

        out = xp.array(
            [
                xp.clip(xp.atleast_2d(psd_interp(f))[i], min_psd[i], max_psd[i])
                for i in range(len(values))
            ]
        )
        return xp.squeeze(
            out
        )  # remove the extra dimension if there is only one channel

    return psd_clipped


if LISATOOLS_AVAILABLE:

    def write_psd_file(
        model="scirdv1",
        channels="AET",
        tdi2=True,
        include_foreground=False,
        filename="example_psd.npy",
        **kwargs,
    ):
        """
        Write a PSD file for a given model.

        Parameters
        ----------
        model : str
            The noise model to use. Default is 'scirdv1'.
        channels : str
            The channels to include in the PSD. Default is 'AET'. if None, the sensitivity curve without projections is computed.
        tdi2 : bool
            Whether to use Second generation TDI. Default is True.
        include_foreground : bool
            Whether to include the foreground noise. Default is False. This is just an extra check, the actual
            argument is in the kwargs.
        filename : str
            The name of the file to save the PSD to. Default is 'example_psd.npy'.
        **kwargs : dict
            Additional keyword arguments to pass to the PSD generation function.
        """

        assert channels in [
            None,
            "A",
            "AE",
            "AET",
        ], "channels must be None, 'A', 'AE', or 'AET'"
        if include_foreground:
            assert (
                "stochastic_params" in kwargs.keys()
            ), "`stochastic_params = List(Tobs) [s]` must be provided if include_foreground is True"

        freqs = np.linspace(0, 1, 100001)[1:]

        if channels is None:
            sens_fns = [LISASens]

            default_kwargs = dict(return_type="PSD", average=False)

        elif "A" in channels:
            sens_fns = [A1TDISens]
            if "E" in channels:
                sens_fns.append(E1TDISens)
            if "T" in channels:
                sens_fns.append(T1TDISens)

            default_kwargs = dict(
                return_type="PSD",
            )

        updated_kwargs = default_kwargs | kwargs

        Sn = [
            get_sensitivity(freqs, sens_fn=sens_fn, model=model, **updated_kwargs)
            for sens_fn in sens_fns
        ]

        if tdi2:
            x = 2.0 * np.pi * lisaLT * freqs
            tdi_factor = 4 * np.sin(2 * x) ** 2
            Sn = [sens * tdi_factor for sens in Sn]

        Sn = np.array(Sn)
        np.save(filename, np.vstack((freqs, Sn)).T)

    def load_psd(
        logger,
        model="scirdv1",
        channels="AET",
        tdi2=True,
        include_foreground=False,
        filename="example_psd.npy",
        xp=np,
        **kwargs,
    ):
        """
        Load the PSD from a file and returns an interpolant. If the file does not exist, it will be created.

        Parameters
        ----------
        model : str
            The noise model to use. Default is 'scirdv1'.
        channels : str
            The channels to include in the PSD. Default is 'AET'.
        tdi2 : bool
            Whether to use Second generation TDI. Default is True.
        include_foreground : bool
            Whether to include the foreground noise. Default is False. This is just an extra check, the actual
            argument is in the kwargs.
        filename : str
            The name of the file to save the PSD to. Default is 'example_psd.npy'.
        xp : module
            The module to use for array operations. Default is np.
        **kwargs : dict
            Additional keyword arguments to pass to the PSD generation function.

        Returns
        -------
        psd_clipped : function
            A function that takes a frequency and returns the PSD at that frequency
        """
        if filename is None or filename == "None":
            tdi_gen = "tdi2" if tdi2 else "tdi1"
            foreground = "wd" if include_foreground else "no_wd"
            filename = f"noise_psd_{model}_{channels}_{tdi_gen}_{foreground}.npy"
        if not os.path.exists(filename):
            logger.warning("PSD file %s does not exist. Creating it now.", filename)
            write_psd_file(
                model=model,
                channels=channels,
                tdi2=tdi2,
                include_foreground=include_foreground,
                filename=filename,
                **kwargs,
            )

        logger.info("Loading PSD from %s", filename)
        return load_psd_from_file(filename, xp=xp)


def resolve_noise_model(
    noise_model=None,
    noise_kwargs=None,
    channels=None,
    ResponseWrapper_kwargs=None,
    use_gpu=False,
):
    """
    Noise model, its kwargs and the channel names StableEMRIFisher uses.

    - A `noise_model` is used as given, with `noise_kwargs` defaulting to
      {"TDI": "TDI1"} and `channels` to ["A", "E"].
    - Without a noise model and without a response: the sky-averaged
      long-wavelength sensitivity `sensitivity_lwa` on channels ["I", "II"].
    - Without a noise model but with a response (not recommended): a PSD is
      generated with lisatools (scirdv1 noise, A and E channels, the TDI
      generation from `ResponseWrapper_kwargs["tdi"]`, galactic foreground for
      the observation time `ResponseWrapper_kwargs["Tobs"]`), written to the
      current directory and loaded back. It is built independently of the
      rest of the response configuration, which it may not match.

    Args:
        noise_model (callable, optional): function of frequency returning
            the one-sided PSD.
        noise_kwargs (dict or list of dict, optional): kwargs for noise_model.
        channels (list of str, optional): channel names.
        ResponseWrapper_kwargs (dict, optional): kwargs of the response
            wrapper, None if no response is used.
        use_gpu (bool): load the generated PSD as CuPy arrays.

    Returns:
        tuple: (noise_model, noise_kwargs, channels).
    """
    if noise_model is not None:
        if noise_kwargs is None:
            noise_kwargs = {"TDI": "TDI1"}
        if channels is None:
            channels = ["A", "E"]
        return noise_model, noise_kwargs, channels

    if ResponseWrapper_kwargs is None:
        logger.warning("No noise model or response wrapper provided.")
        logger.warning("Defaulting to the sky-averaged sensitivity curve")
        return sensitivity_lwa, {}, channels if channels is not None else ["I", "II"]

    if not LISATOOLS_AVAILABLE:
        raise ImportError(
            "Generating the default PSD requires lisatools (pip install lisaanalysistools). "
            "Install it, or pass a noise_model explicitly."
        )

    tdi2 = ResponseWrapper_kwargs.get("tdi") == "2nd generation"
    Tobs = ResponseWrapper_kwargs["Tobs"]
    logger.warning(
        "NOT RECOMMENDED: no noise_model was given, so a default PSD is generated "
        "(lisatools scirdv1 noise, channels A and E, %s TDI, galactic foreground "
        "for Tobs = %s yr). It is built independently of the ResponseWrapper "
        "configuration used for the waveform (TDI channels, orbits, noise model), "
        "which it may not match; a mismatch, e.g. tdi_chan='AET' (three channels) "
        "against this two-channel PSD, gives wrong results or errors later on. "
        "Pass a noise_model (with noise_kwargs and channels) consistent with "
        "the response instead.",
        "2nd generation" if tdi2 else "1st generation",
        Tobs,
    )

    PSD_filename = os.path.join(
        os.getcwd(),
        f"tdi{2 if tdi2 else 1}_with_foreground.npy",
    )
    write_psd_file(
        model="scirdv1",
        channels="AE",
        tdi2=tdi2,
        include_foreground=True,
        filename=PSD_filename,
        stochastic_params=[Tobs * YRSID_SI],
    )
    logger.info("Default PSD written to %s", PSD_filename)

    noise_model = load_psd_from_file(PSD_filename, xp=cp if use_gpu else np)
    return noise_model, {}, ["A", "E"]
