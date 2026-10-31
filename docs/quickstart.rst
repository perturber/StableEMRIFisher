Quick Start Guide
=================

StableEMRIFisher (SEF) does two things:

- **Robust numerical derivatives** of EMRI waveforms with respect to their parameters.
- **Fisher matrices** built from these derivatives, for fast estimates of parameter uncertainties.

This guide shows how to compute a Fisher matrix, how to control the finite-difference step sizes, how to include the LISA response, and how to compute a waveform derivative on its own. The `example notebooks <https://github.com/perturber/StableEMRIFisher/tree/main/examples>`_ cover each topic in more detail.

The examples below run on a CPU; set ``use_gpu=True`` (and remove the FEW backend lines) to run on an NVIDIA GPU.

Computing a Fisher Matrix
-------------------------

``StableEMRIFisher`` takes a FEW waveform class (not an instance) and its keyword arguments, the observation time ``T`` (years) and the sampling interval ``dt`` (seconds). It is then called with the source parameters, and returns the Fisher matrix over the parameters in ``param_names``.

.. code-block:: python

    import numpy as np

    import few
    from few.waveform import FastKerrEccentricEquatorialFlux

    from stableemrifisher.fisher import StableEMRIFisher

    # run FEW on the CPU
    cfg_set = few.get_config_setter(reset=True)
    cfg_set.enable_backends("cpu")

    sef = StableEMRIFisher(
        waveform_class=FastKerrEccentricEquatorialFlux,
        waveform_class_kwargs=dict(
            inspiral_kwargs=dict(err=1e-11),  # trajectory integrator tolerance
            mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
        ),
        T=0.01,  # observation time, years
        dt=10.0,  # sampling interval, seconds
        der_order=4,  # order of the finite-difference stencils
        use_gpu=False,
    )

    wave_params = dict(
        m1=1e6,  # primary mass, solar masses
        m2=10.0,  # secondary mass, solar masses
        a=0.9,  # primary spin
        p0=10.0,  # initial semi-latus rectum
        e0=0.4,  # initial eccentricity
        xI0=1.0,  # cosine of the initial inclination
        dist=0.05,  # luminosity distance, Gpc
        qS=0.2,  # sky location, polar angle
        phiS=0.8,  # sky location, azimuthal angle
        qK=1.6,  # spin orientation, polar angle
        phiK=1.5,  # spin orientation, azimuthal angle
        Phi_phi0=2.0,  # initial azimuthal phase
        Phi_theta0=0.0,  # initial polar phase
        Phi_r0=3.0,  # initial radial phase
    )

    param_names = ["m1", "m2", "a", "p0", "e0", "dist", "Phi_phi0", "Phi_r0"]

    Fisher = sef(wave_params, param_names=param_names)

    # the inverse of the Fisher matrix approximates the covariance matrix
    sigma = np.sqrt(np.diag(np.linalg.inv(Fisher)))
    for name, s in zip(param_names, sigma):
        print(f"{name:9s} = {wave_params[name]:.6g} +/- {s:.3g}")

Without a ``ResponseWrapper`` or ``noise_model``, SEF uses the sky-averaged LISA sensitivity curve on the two polarisations :math:`h_+` and :math:`h_\times`. Without the response, the waveform depends on the four sky and spin angles (``qS``, ``phiS``, ``qK``, ``phiK``) only through two combinations, so they are left out of ``param_names`` here: SEF would report them as degenerate (see below).

Choosing Step Sizes
-------------------

For each parameter, SEF tries ``Ndelta`` finite-difference step sizes and keeps the one where the diagonal Fisher element :math:`\Gamma_{ii}` is most stable. The trial steps come from a default grid, or from ``delta_range``. Parameters with analytic derivatives (the distance and, for the stable derivative, the initial phases) need no step size.

.. code-block:: python

    Ndelta = 8
    delta_range = dict(
        m1=np.geomspace(1e2, 1e-3, Ndelta),
        m2=np.geomspace(1e-3, 1e-8, Ndelta),
        a=np.geomspace(1e-4, 1e-9, Ndelta),
    )

    Fisher = sef(
        wave_params,
        param_names=["m1", "m2", "a"],
        Ndelta=Ndelta,
        delta_range=delta_range,
        stability_plot=True,  # plot Gamma_ii against the step size
        stats_for_nerds=True,  # print the chosen step size of each parameter
    )

    print(sef.deltas)  # chosen step size of each parameter
    print(sef.relerr)  # relative change of Gamma_ii at the chosen step

``stability_plot=True`` plots :math:`\Gamma_{ii}` against the step size for each parameter; the chosen step lies on the plateau where it is stable. ``stats_for_nerds=True`` prints the chosen step size and the relative error of :math:`\Gamma_{ii}`. For full debugging output, set the package logger to DEBUG: ``logging.getLogger("stableemrifisher").setLevel("DEBUG")``.

To skip the search, pass a step size for every parameter, for example those found in a previous run:

.. code-block:: python

    Fisher = sef(wave_params, param_names=["m1", "m2", "a"], deltas=sef.deltas)

Results and Checks
------------------

Besides returning the Fisher matrix, a call stores:

- ``sef.deltas`` and ``sef.relerr``: the step sizes and the relative errors of the diagonal elements.
- ``sef.fisher_check``: whether the Fisher matrix is positive definite, and any degenerate parameters. SEF reports degeneracies (for example parameters that do not affect the waveform, or combinations of them) as warnings and leaves the matrix unchanged.
- ``sef.covariance``: the covariance matrix, if called with ``CovEllipse=True``, which also plots the covariance ellipses.
- ``sef.derivatives``: the waveform derivatives, if called with ``keep_derivatives=True``.
- ``sef.SNR2``: the squared optimal SNR of the source.

With ``filename`` (an output directory), the Fisher matrix, the step sizes and, with ``save_derivatives=True``, the derivatives are also written to files.

With the LISA Response
----------------------

The LISA response from ``fastlisaresponse`` is given as the ``ResponseWrapper`` class with its keyword arguments, together with a noise model for the TDI channels it returns. Here, first-generation TDI on equal-arm-length orbits returns the A and E channels, and the noise model gives the LISA PSD of each channel from ``lisatools``:

.. code-block:: python

    from fastlisaresponse import ResponseWrapper
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens

    T = 0.01  # observation time, years
    dt = 10.0  # sampling interval, seconds

    ResponseWrapper_kwargs = dict(
        Tobs=T,
        dt=dt,
        index_lambda=8,  # position of phiS in the waveform arguments
        index_beta=7,  # position of qS in the waveform arguments
        t0=10000.0,  # seconds at each end of the signal where the response is unreliable
        remove_garbage="zero",  # zero the signal within t0 of each end
        flip_hx=True,  # FEW returns h+ - i hx
        is_ecliptic_latitude=False,  # qS is a polar angle, not a latitude
        orbits=EqualArmlengthOrbits(),
        order=20,  # order of the Lagrange interpolation of the time delays
        tdi="1st generation",
        tdi_chan="AE",
    )

    channels = [A1TDISens, E1TDISens]

    sef_response = StableEMRIFisher(
        waveform_class=FastKerrEccentricEquatorialFlux,
        waveform_class_kwargs=dict(
            inspiral_kwargs=dict(err=1e-11),
            mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
        ),
        ResponseWrapper=ResponseWrapper,
        ResponseWrapper_kwargs=ResponseWrapper_kwargs,
        noise_model=get_sensitivity,
        noise_kwargs=[{"sens_fn": channel} for channel in channels],  # one dict per channel
        channels=channels,
        der_order=4,
        use_gpu=False,
    )

    param_names = ["m1", "m2", "a", "p0", "e0", "dist", "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_r0"]
    Fisher = sef_response(wave_params, param_names=param_names)

With the response, the sky and spin angles can be measured. If no ``noise_model`` is given with a response, SEF generates a default PSD, but warns that it may not match the response configuration; passing a noise model consistent with the response, as above, is recommended. Any callable whose first argument is the frequency can be used as a noise model (see ``custom_psd_tutorial.ipynb``).

Waveform Derivatives
--------------------

The derivative classes can also be used on their own, without building a Fisher matrix. ``StableDerivative`` (``deriv_type="stable"``) computes the stable derivative; ``DirectDerivative`` (``deriv_type="direct"``) takes finite differences of the full waveform. Both build the waveform model from a FEW waveform class, take the reference point with ``set_reference``, and return derivatives with one row per channel (here :math:`[\partial h_+, \partial h_\times]`).

.. code-block:: python

    import matplotlib.pyplot as plt

    from stableemrifisher.derivatives import StableDerivative

    derivative = StableDerivative(
        waveform_class=FastKerrEccentricEquatorialFlux,
        waveform_generator_kwargs=dict(
            inspiral_kwargs=dict(err=1e-11),
            mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
        ),
    )
    derivative.set_reference(wave_params, T=0.01, dt=10.0)

    # derivative with respect to m1, with a 4th-order central stencil
    dh_dm1 = derivative("m1", 1e-1, order=4, kind="central")

    time_hours = np.arange(dh_dm1.shape[-1]) * 10.0 / 3600
    plt.plot(time_hours, dh_dm1[0])
    plt.xlabel("time [hours]")
    plt.ylabel(r"$\partial h_+ / \partial m_1$")
    plt.show()

See ``derivatives_tutorial.ipynb`` for comparisons of the stable derivative with direct finite differences, with and without the LISA response.
