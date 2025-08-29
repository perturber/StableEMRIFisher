Quick Start Guide
=================

The key features of the `StableEMRIFisher` package is twofold 

- **Robust Numerical Derivatives:** Cheap and Robust EMRI waveform derivatives 
- **Fisher Matrix Computations** Fisher matrix calculations for accelerated parameter inference. 

This quick-start guide will walk you through computing numerical derivatives and ultimately Fisher matrices.

Numerical Derivatives of Waveforms
----------------------------------

Here's an example of compute a parameter derivative of a detector frame Kerr Eccentric Equatorial waveform model

.. code-block:: python

    # Import relevant EMRI packages
    from few.waveform import(
        GenerateEMRIWaveform,
        FastKerrEccentricEquatorialFlux
    )

    from few.utils.constants import YRSID_SI

    # Import StableEMRIDerivative
    from stableemrifisher.fisher.stablederivative import StableEMRIDerivative
    
    import matplotlib.pyplot as plt
    import numpy as np

    # Define Waveform params
    wave_params = {
        "m1": 1e6,      # Primary mass (solar masses)
        "m2": 1e1,      # Secondary mass (solar masses)
        "a": 0.9,      # Spin parameter
        "p0": 10,     # Initial semi-latus rectum 
        "e0": 0.4,    # Initial eccentricity
        "xI0": 1.0,   # Initial inclination (equatorial orbits)
        "dist": 1.0,  # Distance to source 
        "qS": 0.2,    # Sky location polar angle
        "phiS": 0.8,  # Sky location azimuthal angle
        "qK": 1.6,    # Orientation spin vector polar angle
        "phiK": 1.5,  # Orientation spin vector azimuthal angle
        "Phi_phi0": 2.0, # Initial phase in phi0
        "Phi_theta0": 0.0, # Initial phase in theta0
        "Phi_r0": 3.0, # Initial phase in r0
    }

    # waveform class setup
    waveform_class_kwargs = {
        "inspiral_kwargs": {
            "err": 1e-11,
        },
        "mode_selector_kwargs": {
            "mode_selection_threshold": 1e-5
        },
    }

    waveform_generator_kwargs = {"return_list": False, "frame": "detector"}


    EMRI_deriv = StableEMRIDerivative(
        FastKerrEccentricEquatorialFlux,
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=waveform_generator_kwargs,
    )

    T = 0.01
    dt = 10.0
    kwargs = {"T": T, "dt": dt}
    compute_stable_deriv = EMRI_deriv(
        parameters=wave_params,
        param_to_vary="m1",
        delta=1e-1,
        order=4,
        kind="central",
        **kwargs,
    )

    t = np.arange(0, T * YRSID_SI, dt)
    plt.plot(t, compute_stable_deriv.real, c="blue", label="h_p: m1 partial derivative")
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Derivative", fontsize=16)
    plt.title("EMRI Derivative with respect to m1", fontsize=16)
    plt.grid(True)
    plt.show()

The code block above computes the numerical derivative of the waveform with respect to the primary mass `m1` using a 4th order central finite difference stencil with a step size of `delta=1e-1`. The resulting derivative is plotted as a function of time.

Fisher Matrix 
-------------

Detector Frame Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~

Extract parameter uncertainties from the Fisher matrix:

.. code-block:: python

   # Import relevant EMRI packages
    from few.waveform import (
        GenerateEMRIWaveform,
        FastKerrEccentricEquatorialFlux,
    )
    # Import StableEMRIFisher
    from stableemrifisher.fisher import StableEMRIFisher

    import numpy as np
    # Waveform params
    dt = 5.0
    T = 0.01
    wave_params = {
        "m1": 1e6,
        "m2": 1e1,
        "a": 0.9,
        "p0": 10,
        "e0": 0.4,
        "xI0": 1.0,
        "dist": 1.0,
        "qS": 0.2,
        "phiS": 0.8,
        "qK": 1.6,
        "phiK": 1.5,
        "Phi_phi0": 2.0,
        "Phi_theta0": 0.0,
        "Phi_r0": 3.0,
    }

    # waveform class setup
    waveform_class_kwargs = {
        "inspiral_kwargs": {
            "err": 1e-11,
        },
        "mode_selector_kwargs": {
            "mode_selection_threshold": 1e-5
        },
    }

    # waveform generator setup
    waveform_generator = GenerateEMRIWaveform
    waveform_generator_kwargs = {"return_list": False, 
                                "frame": "detector"}


    der_order = 4 # Order 4 stencil
    Ndelta = 8 # Number of finite difference steps

    # Initialise Fisher matrix class
    # use latest KerrEccentricEquatorial waveform model
    # with GenerateEMRIWaveform interface 

    sef = StableEMRIFisher(
        waveform_class=FastKerrEccentricEquatorialFlux,
        waveform_class_kwargs=waveform_class_kwargs,
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=waveform_generator_kwargs,
        dt=dt,
        T=T,
        der_order=der_order, 
        Ndelta=Ndelta,
        deriv_type="stable",
    )

    # Specify what parameters to compute Fisher matrix for
    param_names = [
        "m1",
        "m2",
        "a",
    ]

    # User can specify their own delta values to compute FM
    # More advanced techniques to determine best value of 
    # finite difference deltas will be discussed later
    deltas = np.array([1e-1, 1e-6, 1e-7])

    # Compute Fisher matrix
    fisher_matrix = sef(
        wave_params,  
        param_names=param_names, 
        deltas=deltas, 
    )

    # Compute parameter covariance matrix -- inverse of FM
    param_cov = np.linalg.inv(fisher_matrix)

    for k, item in enumerate(param_names):
        print(
            "Precision measurement in param {} is {}".format(
                item, param_cov[k, k] ** (1 / 2)
            )
        )

One can use more advanced features with StableEMRIFisher to ensure 
convergence of the numerical derivatives. 

.. code-block:: python

    sef = StableEMRIFisher(
        waveform_class=FastKerrEccentricEquatorialFlux,
        waveform_class_kwargs=waveform_class_kwargs,
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=waveform_generator_kwargs,
        dt=dt,
        T=T,
        stability_plot = True,
        stats_for_nerds = True,
        return_derivatives = True,
        der_order=der_order, 
        Ndelta=Ndelta,
        deriv_type="stable",
    )
    # Specify what parameters to compute Fisher matrix for
    param_names = [
        "m1",
        "m2",
        "a",
    ]

    # StableEMRIFisher computes numerical derivatives and 
    # Fisher based scalars to identify the optimal finite difference
    # step size within the intervals set below
    delta_range = {
        "m1": np.geomspace(1e2, 1e-3, Ndelta),
        "m2": np.geomspace(1e-3, 1e-8, Ndelta),
        "a": np.geomspace(1e-4, 1e-9, Ndelta),
    }

    # Compute Fisher matrix
    param_derivs, fisher_matrix = sef(
                wave_params,  
                param_names=param_names, 
                delta_range=delta_range
                )
    
    # Compute parameter covariance matrix -- inverse of FM
    param_cov = np.linalg.inv(fisher_matrix)

    for k, item in enumerate(param_names):
        print(
            "Precision measurement in param {} is {}".format(
                item, param_cov[k, k] ** (1 / 2)
            )
        )

In the code block above, setting `stability_plot = True` shows a plot of the Fisher scalars :math:`\Gamma_{ii}` as a function of finite difference step size. The optimal step size is chosen to be in the plateau region where the Fisher scalars are stable.
The argument `stats_for_nerds = True` enables additional output which can be useful for debugging and understanding the behavior of the numerical derivatives.

With the Response Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the time-domain Response function alongside state-of-the-art Power Spectral Densities (PSDs) for either first/second generation TDI variables to compute the Fisher Matrix. 

.. code-block:: python

    # Import relevant EMRI packages
    from few.waveform import (
        GenerateEMRIWaveform,
        FastKerrEccentricEquatorialFlux,
    )
    from stableemrifisher.fisher import StableEMRIFisher

    from fastlisaresponse import ResponseWrapper             # Response
    from lisatools.detector import EqualArmlengthOrbits

    import numpy as np
    # Waveform params
    dt = 5.0
    T = 0.01
    wave_params = {
        "m1": 1e6,
        "m2": 1e1,
        "a": 0.9,
        "p0": 10,
        "e0": 0.4,
        "xI0": 1.0,
        "dist": 1.0,
        "qS": 0.2,
        "phiS": 0.8,
        "qK": 1.6,
        "phiK": 1.5,
        "Phi_phi0": 2.0,
        "Phi_theta0": 0.0,
        "Phi_r0": 3.0,
    }


    ####=======================True Responsed waveform==========================
    waveform_class = FastKerrEccentricEquatorialFlux
    waveform_class_kwargs = {
        "inspiral_kwargs": {
            "err": 1e-11,
        },
        "mode_selector_kwargs": {"mode_selection_threshold": 1e-5},
    }
    # waveform generator setup
    waveform_generator = GenerateEMRIWaveform
    waveform_generator_kwargs = {"return_list": False, "frame": "detector"}
    # Response function set up
    USE_GPU = False
    tdi_kwargs = dict(
        orbits=EqualArmlengthOrbits(use_gpu=USE_GPU),
        order=25,
        tdi="2nd generation",
        tdi_chan="AE",
    )  

    INDEX_LAMBDA = 8
    INDEX_BETA = 7

    # with longer signals we care less about this
    t0 = 20000.0  # throw away on both ends when our orbital information is weird

    ResponseWrapper_kwargs = dict(
        Tobs = T,
        dt = dt,
        index_lambda = INDEX_LAMBDA,
        index_beta = INDEX_BETA,
        t0 = t0,
        flip_hx = True,
        use_gpu=USE_GPU,
        is_ecliptic_latitude=False,
        remove_garbage="zero",
        **tdi_kwargs
    )

    der_order = 4
    Ndelta = 8
    stability_plot = False
    sef = StableEMRIFisher(waveform_class=waveform_class, 
                        waveform_class_kwargs=waveform_class_kwargs,
                        waveform_generator=waveform_generator,
                        waveform_generator_kwargs=waveform_generator_kwargs,
                        ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=ResponseWrapper_kwargs,
                        stats_for_nerds = True, use_gpu = USE_GPU,
                            T = T, dt = dt,
                            der_order = der_order,
                            Ndelta = Ndelta,
                            stability_plot = stability_plot,
                            return_derivatives = False,
                        deriv_type='stable')

    param_names = ['m1','m2','a']

    delta_range = dict(
        m1 = np.geomspace(1e3, 1e-5, Ndelta),
        m2 = np.geomspace(1e-2, 1e-8, Ndelta),
        a = np.geomspace(1e-5, 1e-9, Ndelta),
    )

    fisher_matrix = sef(wave_params, param_names = param_names, 
                            delta_range = delta_range,
                            filename=None,
                            live_dangerously = False)


    param_cov = np.linalg.inv(fisher_matrix)

    for k, item in enumerate(param_names):
        print("Precision measurement in param {} is {}".format(item, param_cov[k,k]**(1/2)))

