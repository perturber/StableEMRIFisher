"""
This module provides noise modeling functionality for LISA (Laser Interferometer Space Antenna) gravitational wave detector.

Key components:
- Sensitivity functions for LISA using long-wavelength approximation and TDI (Time-Delay Interferometry) configurations
- Power spectral density (PSD) generation and management for different noise models and channel configurations
- File I/O utilities for saving and loading pre-computed PSD data with interpolation capabilities
- Support for both CPU and GPU backends via configurable array libraries

"""

from few.summation.interpolatedmodesum import CubicSplineInterpolant
from lisatools.sensitivity import (
    get_sensitivity,
    A1TDISens,
    E1TDISens,
    T1TDISens,
    LISASens,
)
from lisatools.utils.constants import lisaLT

import numpy as np
import os


def sensitivity_LWA(f):
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
    Sc = (
        A
        * f ** (-7 / 3)
        * np.exp(-(f**alpha) + beta * f * np.sin(kappa * f))
        * (1 + np.tanh(gamma * (fk - f)))
    )
    sensitivity = (10 / (3 * L**2)) * (
        P_OMS + 4 * (P_acc) / ((2 * np.pi * f) ** 4)
    ) * (1 + 6 * f**2 / (10 * fstar**2)) + Sc
    return sensitivity


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
    print(f"Using {backend} backend for PSD interpolation")

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
        logger.warning(f"PSD file {filename} does not exist. Creating it now.")
        write_psd_file(
            model=model,
            channels=channels,
            tdi2=tdi2,
            include_foreground=include_foreground,
            filename=filename,
            **kwargs,
        )

    logger.info(f"Loading PSD from {filename}")
    return load_psd_from_file(filename, xp=xp)
