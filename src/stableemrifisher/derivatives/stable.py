"""Semi-analytic ("stable") derivative of the EMRI waveform."""

import inspect
import logging

import numpy as np
from few.waveform import GenerateEMRIWaveform
from few.trajectory.ode.circ1pat1r import TrajectoryCirc1PAT1R
from few.utils.constants import Gpc, MRSUN_SI
from tqdm import tqdm

from ..utils.deriv_utils.deriv_angles import viewing_angle_partials, fplus_fcross_derivs
from .base import DerivativeBase
from .finite_difference import check_stencil, offsets, apply_stencil

logger = logging.getLogger("stableemrifisher")

# call kwargs of a FEW waveform that it passes to its mode selector
MODE_SELECTION_KWARGS = ("mode_selection", "include_minus_mkn", "mode_selection_threshold")


class _StableWaveformDerivative(GenerateEMRIWaveform):
    """
    Computes the stable derivative of the waveform without the response.

    Inherits from the GenerateEMRIWaveform class of FEW to reuse its trajectory,
    amplitude and summation modules, and replaces __call__ with the derivative
    calculation. Used internally by StableDerivative.
    """

    def __init__(
        self,
        waveform_class,
        *args,
        frame: str = "detector",
        return_list=False,
        flip_output=False,
        **kwargs,
    ):
        super().__init__(
            waveform_class=waveform_class,
            *args,
            frame=frame,
            return_list=return_list,
            flip_output=flip_output,
            **kwargs,
        )  # initialize GenerateEMRIWaveform
        self.cache = None  # initialize waveform cache

    def __getattr__(self, name):
        # get_attributes from self.waveform_generator if not found in GenerateEMRIWaveform
        return getattr(self.waveform_generator, name)

    def __call__(
        self,
        *args,  # dummy to fool FastLISAResponse
        parameters: dict,
        param_to_vary: str,
        delta: float,
        order: int,
        kind: str,
        **kwargs,
    ):
        """
        generate the waveform derivative for the given parameter
        """

        self.delta = delta
        self.order = order
        self.kind = kind

        check_stencil(self.order, self.kind)

        try:
            T = kwargs["T"]
        except KeyError:
            T = 1.0

        try:
            dt = kwargs["dt"]
        except KeyError:
            dt = 10.0

        try:
            batch_size = kwargs["batch_size"]
        except KeyError:
            batch_size = -1

        try:
            show_progress = kwargs["show_progress"]
        except KeyError:
            show_progress = False

        # construct deltas for derivative
        self.deltas = offsets(self.delta, self.order, self.kind)

        # how we proceed depends on the parameter we are differentiating with respect to
        if param_to_vary not in parameters:
            raise ValueError(
                f"Parameter '{param_to_vary}' not in parameters dictionary."
            )

        # obtain source frame angles
        theta_source, phi_source = self._get_viewing_angles(
            parameters["qS"], parameters["phiS"], parameters["qK"], parameters["phiK"]
        )

        # mode-selection kwargs are consumed here, as in FEW, and not passed on
        selector_kwargs = self._mode_selector_kwargs(kwargs)

        keys_exclude = ["T", "dt", "batch_size", "show_progress", *MODE_SELECTION_KWARGS]
        kwargs_remaining = {
            key: value for key, value in kwargs.items() if key not in keys_exclude
        }

        # get waveform
        if self.cache is None or parameters != self.cache["parameters"]:

            t, y = self._trajectory_from_parameters(parameters, T)
            fixed_amp_args = self._fixed_amplitude_args()

            self.cache = {
                "t": t,
                "y": y,
                "parameters": parameters,
                "coefficients": self.inspiral_generator.integrator_spline_coeff,
                # Phi_phi, Phi_theta, Phi_r: the summation sums m, k and n phases
                "phase_coefficients": self.xp.asarray(
                    self.inspiral_generator.integrator_spline_phase_coeff
                ),
                "phase_coefficients_t": self.inspiral_generator.integrator_spline_t,
            }

            amps_here = self._amplitudes_from_trajectory(
                parameters,
                t=t,
                y=y,
                fixed_amp_args=fixed_amp_args,
                theta_source=float(theta_source),
                phi_source=phi_source,
                cache=True,
                selector_kwargs=selector_kwargs,
            )

            # create waveform at injection

            waveform_source = self._create_waveform_in_batches(
                self.cache["t"],
                amps_here,  # actually teuk_amps * Ylms_in
                self.cache["dummy_ylms"],  # a bunch of ones
                self.cache["phase_coefficients_t"],
                self.cache["phase_coefficients"],
                self.cache["ls_all"],
                self.cache["ms_all"],
                self.cache["ks_all"],
                self.cache["ns_all"],
                dt=dt,
                T=T,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs_remaining,
            )

            self.cache["waveform_source"] = waveform_source

        self.cache["Npad"] = 0

        # now calculate the derivatives with respect to the chosen parameters

        # distance
        if param_to_vary == "dist":
            waveform_derivative_source = (
                -self.cache["waveform_source"] / parameters["dist"]
            )

        # phases
        elif param_to_vary in ["Phi_phi0", "Phi_theta0", "Phi_r0"]:
            # factor of -1j*m is applied to each amplitude
            modified_amps = self._modify_amplitudes_for_initial_phase_derivative(
                param_to_vary
            )

            # create waveform derivative
            waveform_derivative_source = self._create_waveform_in_batches(
                self.cache["t"],
                modified_amps,
                self.cache["dummy_ylms"],
                self.cache["phase_coefficients_t"],
                self.cache["phase_coefficients"],
                self.cache["ls_all"],
                self.cache["ms_all"],
                self.cache["ks_all"],
                self.cache["ns_all"],
                dt=dt,
                T=T,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs_remaining,
            )

        # sky angles (SSB)
        elif param_to_vary in ["qS", "phiS", "qK", "phiK"]:
            # finite differencing of the ylms w.r.t. theta, then chain rule partial h / partial theta * partial theta / partial angle
            modified_amps = self._modify_amplitudes_for_angle_derivative(
                parameters,
                param_to_vary,
                theta_source=theta_source,
                phi_source=phi_source,
            )

            waveform_derivative_source = self._create_waveform_in_batches(
                self.cache["t"],
                modified_amps,
                self.cache["dummy_ylms"],
                self.cache["phase_coefficients_t"],
                self.cache["phase_coefficients"],
                self.cache["ls_all"],
                self.cache["ms_all"],
                self.cache["ks_all"],
                self.cache["ns_all"],
                dt=dt,
                T=T,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs_remaining,
            )

        # traj params
        else:
            # if you've reached this point, a derivative w.r.t one of the trajectory parameters is requested.
            # trajectory must be modified

            use_gpu = self.waveform_generator.backend.uses_cupy

            # get trajectories
            y_interps = self.xp.full(
                (len(self.deltas), self.cache["t"].size, len(self.cache["y"])),
                self.xp.nan,
            )  # trajectory for each of the finite difference deltas
            t_interp = self.cache["t"].copy()  # CHANGED
            if use_gpu:
                t_interp_np = t_interp.get()  # Changed!
            else:
                t_interp_np = t_interp

            fixed_amp_args_steps = []  # amplitude args stored by each trajectory
            for k, delt in enumerate(self.deltas):
                parameters_in = parameters.copy()
                parameters_in[param_to_vary] += delt  # perturb by finite-difference
                t, y = self._trajectory_from_parameters(parameters_in, T)
                fixed_amp_args_steps.append(self._fixed_amplitude_args())
                # re-interpolate onto the time-step grid for the injection trajectory
                # t_interp_np = np.asarray(t_interp) # Changed!

                if self.xp.around(t_interp[-1], 5) > self.xp.around(
                    t[-1], 5
                ):  # check plunge. We round to five decimal places to avoid numerical precision errors (which sometimes happen otherwise).
                    logger.warning(
                        "Perturbed trajectory plunges earlier: t_interp[-1] = %s, "
                        "t_traj[-1] = %s",
                        t_interp[-1],
                        t[-1],
                    )

                    mask_notplunging = (
                        t_interp < t[-1]
                    )  # for all t_interp < t[-1], the perturbed trajectory is still not plunging
                    # t_interp_np = t_interp[mask_notplunging].get() # CHANGED
                    # t_interp_np = np.asarray(t_interp[mask_notplunging]) # CHANGED
                    if use_gpu:
                        t_interp_np = t_interp[mask_notplunging].get()
                    else:
                        t_interp_np = t_interp[mask_notplunging]

                y_interp = self.xp.asarray(
                    self.inspiral_generator.inspiral_generator.eval_integrator_spline(
                        t_interp_np
                    ).T
                )  # any unfilled elements due to plunging trajectories assume nans. # CHANGED
                y_interps[k, : len(t_interp_np)] = y_interp.T  # CHANGED

            # In the case of plunge, some trajectories will be shorter than others. They appear as NaN's in the y_interps array
            nans = self.xp.isnan(y_interps)
            if nans.any():
                max_ind = int(self.xp.where(nans.sum(2).sum(0) > 0)[0].min())
            else:
                max_ind = int(y_interp.shape[1])

            # modify size of the trajectories and phases accordingly
            t_interp = self.cache["t"][:max_ind]
            y_interps = y_interps[:, :max_ind, :]
            phases_steps = y_interps[:, :, 3:6]  # Phi_phi, Phi_theta, Phi_r
            phase_coefficients = self.xp.asarray(
                self.cache["phase_coefficients"][: max_ind - 1, :]
            )
            phase_t = self.cache["phase_coefficients_t"][:max_ind]

            # finite differencing the phases
            dPhi_fund_dx = apply_stencil(
                phases_steps, self.delta, self.order, self.kind, xp=self.xp
            )
            # project the fundamental phases up to the full mode index space
            dPhi_dx = (
                dPhi_fund_dx[:, 0, None] * self.cache["ms_all"][None, :]
                + dPhi_fund_dx[:, 1, None] * self.cache["ks_all"][None, :]
                + dPhi_fund_dx[:, 2, None] * self.cache["ns_all"][None, :]
            )

            # get amplitude derivative
            amps_steps = self.xp.zeros(
                (len(self.deltas), max_ind, len(self.cache["ls_all"])),
                dtype=self.xp.complex128,
            )

            for k, delt in enumerate(self.deltas):

                parameters_in = parameters.copy()
                parameters_in[param_to_vary] += delt  # perturb by finite-difference
                amps_here = self._amplitudes_from_trajectory(
                    parameters_in,
                    t_interp,
                    y_interps[k].T,
                    fixed_amp_args=fixed_amp_args_steps[k],
                    theta_source=float(theta_source),
                    phi_source=phi_source,
                    cache=False,
                )  # remember, this function multiplies by Ylmns!
                amps_steps[k] = amps_here

            # finite differencing the amplitudes
            dAmp_dx = apply_stencil(
                amps_steps, self.delta, self.order, self.kind, xp=self.xp
            )  # actually d teuk_amps/dx * Ylmns

            # defining effective amplitudes = dAdx - i A dPhidx
            # In the detector frame the amplitudes scale as 1 / dist_dimensionless,
            # which depends on m1 and m2 through mu. The amplitudes at each stencil
            # step are divided by the dist_dimensionless of that step, so dAmp_dx
            # already includes this dependence.
            wave_amps = self.cache["teuk_modes_with_ylms"][:max_ind]
            effective_amps = dAmp_dx - 1j * wave_amps * dPhi_dx

            # create derivative
            waveform_derivative_source = self._create_waveform_in_batches(
                t_interp,
                effective_amps,  # actually effective amplitudes * Ylmns
                self.cache["dummy_ylms"],  # just a bunch of ones
                phase_t,
                phase_coefficients,
                self.cache["ls_all"],
                self.cache["ms_all"],
                self.cache["ks_all"],
                self.cache["ns_all"],
                dt=dt,
                T=T,
                batch_size=batch_size,
                show_progress=show_progress,
                **kwargs_remaining,
            )

            # pad with zeroes if required to get back to the waveform length
            if max_ind < self.cache["t"].size:
                self.cache["Npad"] = (
                    self.cache["waveform_source"].size - waveform_derivative_source.size
                )  # length of zero padding also cached
                waveform_derivative_source = self.xp.concatenate(
                    (
                        waveform_derivative_source,
                        self.xp.zeros(
                            self.cache["Npad"], dtype=waveform_derivative_source.dtype
                        ),
                    ),
                    axis=0,
                )

        # waveform derivative obtained in the source frame. Now we must transform to detector (SSB) frame
        # and apply antenna patterns (and its derivatives in case of qS, phiS, qK, phiK)

        if self.waveform_generator.frame == "source":
            waveform_source_flipped = -1 * self.cache["waveform_source"]
            waveform_derivative_source *= -1

        # decompose to plus and cross
        (waveform_derivative_source_plus, waveform_derivative_source_cross) = (
            waveform_derivative_source.real,
            -waveform_derivative_source.imag,
        )

        if self.frame == "source":
            if self.return_list is False:
                return (
                    waveform_derivative_source_plus
                    - 1j * waveform_derivative_source_cross
                )
            else:
                return [
                    waveform_derivative_source_plus,
                    waveform_derivative_source_cross,
                ]

        # detector frame requested; apply antenna pattern
        (waveform_derivative_detector_plus, waveform_derivative_detector_cross) = (
            self._to_SSB_frame(
                hp=waveform_derivative_source_plus,
                hc=waveform_derivative_source_cross,
                qS=parameters["qS"],
                phiS=parameters["phiS"],
                qK=parameters["qK"],
                phiK=parameters["phiK"],
            )
        )

        # if derivative is with respect to one of the angles, also need derivatives with antenna pattern
        if param_to_vary in ["qS", "phiS", "qK", "phiK"]:

            Fderivs = fplus_fcross_derivs(
                parameters["qS"],
                parameters["phiS"],
                parameters["qK"],
                parameters["phiK"],
                with_respect_to=param_to_vary,
            )
            dFplusIdx = Fderivs[f"dFplusI/d{param_to_vary}"]
            dFplusIIdx = Fderivs[f"dFplusII/d{param_to_vary}"]
            dFcrossIdx = Fderivs[f"dFcrossI/d{param_to_vary}"]
            dFcrossIIdx = Fderivs[f"dFcrossII/d{param_to_vary}"]

            waveform_derivative_detector_plus += (
                dFplusIdx * waveform_source_flipped.real
                + dFcrossIdx * (-waveform_source_flipped.imag)
            )

            waveform_derivative_detector_cross += (
                dFplusIIdx * waveform_source_flipped.real
                + dFcrossIIdx * (-waveform_source_flipped.imag)
            )

        if self.return_list is False:
            return (
                waveform_derivative_detector_plus
                - 1j * waveform_derivative_detector_cross
            )
        else:
            return [
                waveform_derivative_detector_plus,
                waveform_derivative_detector_cross,
            ]

    def clear_cache(self):
        self.cache = None  # reset cache

    def _mode_selector_kwargs(self, kwargs):
        """
        kwargs for the mode selector, resolved as FEW's waveform resolves them.

        The mode-selection options are call kwargs of the FEW waveform, with
        the defaults of its `_generate_waveform`. The mode selector falls back
        on the options set at initialization (through `mode_selector_kwargs`)
        only for those that are None.

        Args:
            kwargs (dict): kwargs of the waveform call.

        Returns:
            dict: mode-selection kwargs for `self.mode_selector`.
        """
        defaults = inspect.signature(self.waveform_generator._generate_waveform).parameters
        return {
            name: kwargs.get(name, defaults[name].default)
            for name in MODE_SELECTION_KWARGS
        }

    def _fixed_amplitude_args(self):
        """
        Scalar amplitude arguments stored by the trajectory module when it runs.

        FEW's waveform reads them from the trajectory module right after the
        trajectory, so this must also be called right after
        `_trajectory_from_parameters`. Only Circ1PAT1R has any (nu, chi2).

        Returns:
            tuple: the scalar amplitude arguments.
        """
        func = self.inspiral_generator.func
        if isinstance(func, TrajectoryCirc1PAT1R):
            return (func.args["nu"], func.args["chi2"])
        return ()

    def _amplitude_args(self, fixed_amp_args, y):
        """
        Extra arguments of the amplitude module, after (a, p, e, xI), as FEW's
        waveform builds them.

        Args:
            fixed_amp_args (tuple): from `_fixed_amplitude_args`.
            y (array): trajectory without time; rows beyond the first six
                (p, e, xI and the three phases) are extra evolved quantities.

        Returns:
            list: the extra amplitude arguments.
        """
        add_inspiral_args = list(y[6:])
        if isinstance(self.inspiral_generator.func, TrajectoryCirc1PAT1R):
            deltaM, delta_chit1 = add_inspiral_args
            nu, chi2 = fixed_amp_args
            return [nu, chi2, delta_chit1, deltaM]
        return add_inspiral_args

    def _online_mode_selection_args(self, t):
        """
        Fundamental frequencies along the trajectory for the mode selector, as
        FEW's waveform computes them. Uses the trajectory computed last.

        Args:
            t (array): time steps of the trajectory.

        Returns:
            dict or None: the frequencies f_phi, f_theta and f_r (None if the
            mode selector keeps all modes).
        """
        if self.mode_selector.mode_selection == "all":
            return None
        freqs = self.inspiral_generator.inspiral_generator.eval_integrator_derivative_spline(
            self.xp.asnumpy(t) if hasattr(self.xp, "asnumpy") else t, order=1
        )[:, 3:6] / 2 / np.pi
        return dict(f_phi=freqs[:, 0], f_theta=freqs[:, 1], f_r=freqs[:, 2])

    def _trajectory_from_parameters(self, parameters, T):
        """
        calculate the inspiral trajectory over time T (years) for a given set of parameters.

        Args:
            parameters (dict): dictionary of parameters with the param name as key and its value as value.
            T (float): time (in years) for the inspiral trajectory
        Returns:
            t (np.ndarray): time steps (in seconds) of the trajectory
            y (np.ndarray): evolving parameters of the trajectory along the time grid
        """

        add_parameters = []
        for key, value in parameters.items():
            if key not in [
                "m1",
                "m2",
                "a",
                "p0",
                "e0",
                "xI0",
                "Phi_phi0",
                "Phi_theta0",
                "Phi_r0",
                "dist",
                "qS",
                "phiS",
                "qK",
                "phiK",
            ]:

                add_parameters.append(value)

        traj = self.inspiral_generator(
            parameters["m1"],
            parameters["m2"],
            parameters["a"],
            parameters["p0"],
            parameters["e0"],
            parameters["xI0"],
            *add_parameters,  # any extra trajectory parameters
            Phi_phi0=parameters["Phi_phi0"],
            Phi_theta0=parameters["Phi_theta0"],
            Phi_r0=parameters["Phi_r0"],
            T=T,
            **self.inspiral_kwargs,
        )  # generate the trajectory

        # convert for gpus
        traj = self.xp.asarray([self.xp.asarray(traj[i]) for i in range(len(traj))])

        t = traj[0]
        y = traj[1:]

        return t, y

    def _amplitudes_from_trajectory(
        self,
        parameters,
        t,
        y,
        fixed_amp_args,
        theta_source,
        phi_source,
        cache=False,
        selector_kwargs=None,
    ):
        """
        calculate the amplitudes (times the ylms) of the modes along the trajectory.

        With cache=True (at the reference point), the modes are selected as
        FEW's waveform selects them: the mode selector computes the amplitudes
        and ylms from the trajectory, and the selected (l, m, k, n) modes are
        cached. With cache=False (at a stencil step), only the amplitudes of
        the cached modes are computed, as FEW's mode selector does for a mode
        list, and multiplied by the cached ylms.

        Args:
            parameters (dict): dictionary of trajectory parameters
            t (np.ndarray): array of time steps for trajectory
            y (np.ndarray): array of evolving parameters in the trajectory at time steps
            fixed_amp_args (tuple): from `_fixed_amplitude_args`, for these parameters
            theta_source (np.float): polar angle in source frame
            phi_source (np.float): azimuthal angle in source frame
            cache (bool): whether to select and cache the modes (True) or not (False)
            selector_kwargs (dict): mode selector kwargs from
                `_mode_selector_kwargs`, used if cache is True
        Returns:
            Teukolsky amplitudes times Ylms, for m >= 0 and then m < 0
        """

        # if detector frame, scale by distance in the amplitude module as well.
        if (
            self.frame == "detector"
        ):  # assuming waveform generator is always in source frame
            mu = (
                parameters["m1"]
                * parameters["m2"]
                / (parameters["m1"] + parameters["m2"])
            )
            dist_dimensionless = (parameters["dist"] * Gpc) / (mu * MRSUN_SI)
        else:
            dist_dimensionless = 1.0

        amp_args = self._amplitude_args(fixed_amp_args, y)

        if cache:
            # select the modes as FEW's waveform does: the mode selector computes
            # the amplitudes and ylms and returns those of the selected modes
            teuk_modes_in, ylms_in, ls, ms, ks, ns = self.mode_selector(
                t,
                parameters["a"],
                y[0],
                y[1],
                y[2],
                theta_source,
                phi_source,
                *amp_args,
                online_mode_selection_args=self._online_mode_selection_args(t),
                **selector_kwargs,
            )
            teuk_modes_in = self.xp.asarray(teuk_modes_in)

            # we don't use mode symmetry: the m < 0 modes are added explicitly,
            # A_{l,-m,-k,-n} = (-1)^(l+k) conj(A_{lmkn})
            m0mask = ms != 0
            conj_sign = 1 - 2 * ((ls + ks) % 2)
            teuk_modes_in = self.xp.concatenate(
                (
                    teuk_modes_in,
                    conj_sign[m0mask] * self.xp.conj(teuk_modes_in[:, m0mask]),
                ),
                axis=1,
            )

            ylms_in = self.xp.concatenate(
                (ylms_in[: ls.size], ylms_in[ls.size :][m0mask]), axis=0
            )

            self.cache["ls_all"] = self.xp.concatenate((ls, ls[m0mask]), axis=0)
            self.cache["ms_all"] = self.xp.concatenate((ms, -ms[m0mask]), axis=0)
            self.cache["ks_all"] = self.xp.concatenate((ks, -ks[m0mask]), axis=0)
            self.cache["ns_all"] = self.xp.concatenate((ns, -ns[m0mask]), axis=0)

            self.cache["ls"] = ls.copy()
            self.cache["ms"] = ms.copy()
            self.cache["ks"] = ks.copy()
            self.cache["ns"] = ns.copy()

            self.cache["mode_selection"] = [
                (int(l), int(m), int(k), int(n)) for l, m, k, n in zip(ls, ms, ks, ns)
            ]

            # indices of the selected modes in the amplitude module, as FEW's
            # mode selector finds them for a mode list
            self.cache["keep_inds"] = self.mode_selector.modeinds_map[ls, ms, ks, ns]

            self.cache["teuk_modes"] = teuk_modes_in / dist_dimensionless
            self.cache["teuk_modes_with_ylms"] = self.cache["teuk_modes"] * ylms_in
            self.cache["ylms_in"] = ylms_in
            self.cache["dummy_ylms"] = self.xp.ones(
                2 * teuk_modes_in.shape[1], dtype=self.xp.complex128
            )
            self.cache["dummy_ylms"][teuk_modes_in.shape[1] :] = 0.0
            self.cache["m0mask"] = m0mask
            self.cache["conj_sign"] = conj_sign

            return self.cache["teuk_modes_with_ylms"]

        else:
            # same modes as at the reference point, only their amplitudes change
            teuk_modes_in = self.xp.asarray(
                self.amplitude_generator(
                    parameters["a"],
                    y[0],
                    y[1],
                    y[2],
                    *amp_args,
                    specific_modes=self.cache["keep_inds"],
                )
            )

            m0mask = self.cache["m0mask"]
            teuk_modes_in = self.xp.concatenate(
                (
                    teuk_modes_in,
                    self.cache["conj_sign"][m0mask]
                    * self.xp.conj(teuk_modes_in[:, m0mask]),
                ),
                axis=1,
            )  # get the negative m modes as well

            ylms_in = self.cache[
                "ylms_in"
            ].copy()  # already calculated in the first run, so just copy it

            return teuk_modes_in * ylms_in / dist_dimensionless

    def _create_waveform_in_batches(
        self,
        t,
        amps,
        ylms,
        phase_t,
        phase_coeffs,
        ls,
        ms,
        ks,
        ns,
        dt,
        T,
        batch_size=-1,
        show_progress=False,
        **kwargs,
    ):
        """
        A wrapper for self.create_waveform that handles batching over the time axis.

        Args:
            All arguments are the same as those expected by the underlying
            summation module (e.g., InterpolatedModeSum).

        Returns:
            The full, assembled waveform as a single array.
        """
        # split into batches
        if batch_size == -1 or self.allow_batching is False:
            inds_split_all = [self.xp.arange(len(t))]
        else:
            split_inds = []
            i = 0
            while i < len(t):
                i += batch_size
                if i >= len(t):
                    break
                split_inds.append(i)

            inds_split_all = self.xp.split(self.xp.arange(len(t)), split_inds)

        # select tqdm if user wants to see progress
        iterator = enumerate(inds_split_all)
        iterator = (
            tqdm(iterator, desc="time batch", total=len(inds_split_all))
            if show_progress
            else iterator
        )

        for i, inds_in in iterator:
            # get subsection of the arrays for each batch
            t_batch = t[inds_in]
            amps_batch = amps[inds_in]

            # The phase information (spline coefficients) and mode lists are not
            # time-dependent, so they are passed through unmodified.
            waveform_batch = self.create_waveform(
                t_batch,
                amps_batch,
                ylms,
                phase_t,
                phase_coeffs,
                l_arr=ls,
                m_arr=ms,
                k_arr=ks,
                n_arr=ns,
                dt=dt,
                T=T,
                **kwargs,
            )

            if i == 0:
                waveform = waveform_batch
            else:
                waveform = self.xp.concatenate((waveform, waveform_batch))

        return waveform

    def _modify_amplitudes_for_initial_phase_derivative(self, param_to_vary):
        """
        calculates modified amplitudes for phase derivatives.

        Args:
            param_to_vary (string): one of "Phi_phi0", "Phi_theta0", "Phi_r0"
        returns
            modified amplitudes = -i (mode_index) A
        """

        if param_to_vary == "Phi_phi0":
            factor = -1j * self.cache["ms_all"]
        elif param_to_vary == "Phi_theta0":
            factor = -1j * self.cache["ks_all"]
        elif param_to_vary == "Phi_r0":
            factor = -1j * self.cache["ns_all"]

        modified_amps = self.cache["teuk_modes_with_ylms"] * factor[None, :]
        return modified_amps

    def _modify_amplitudes_for_angle_derivative(
        self, parameters, param_to_vary, theta_source, phi_source
    ):
        """
        calculates modified amplitudes for angle derivatives (qS, phiS, qK, phiK)

        Args:
            parameters (dict): model parameters
            param_to_vary (str): one of qS, phiS, qK, phiK: parameter with respect to which to calculate the derivative
            theta_source (float): polar angle in source frame
            phi_source (float): azimuthal angle in source frame
        Returns:
            modified amplitudes = A * partial Y_lm / partial theta * partial_theta / partial kappa where kappa is the param_to_vary
        """

        ylm_temp = self.xp.zeros(
            (len(self.deltas), self.cache["ls_all"].size), dtype=self.xp.complex128
        )

        # first calculate dylm_dtheta
        for k, delt in enumerate(self.deltas):

            theta_source_perturb = theta_source + float(delt)
            phi_source_perturb = (
                phi_source  # no delta in phi_source because phi_source is fixed!
            )
            # get the ylms for this theta
            ylm_temp[k] = self.ylm_gen(
                self.cache["ls_all"],
                self.cache["ms_all"],
                theta_source_perturb,
                phi_source_perturb,
            )

        dYlm_dtheta = apply_stencil(
            ylm_temp, self.delta, self.order, self.kind, xp=self.xp
        )

        # now calculate dtheta_dx where x is the param_to_vary
        if param_to_vary == "qS":
            key_dtheta_dx = "del theta_src / del qS"
        elif param_to_vary == "phiS":
            key_dtheta_dx = "del theta_src / del phi_S"
        elif param_to_vary == "qK":
            key_dtheta_dx = "del theta_src / del qK"
        elif param_to_vary == "phiK":
            key_dtheta_dx = "del theta_src / del phi_K"

        dtheta_dx = viewing_angle_partials(
            parameters["qS"], parameters["phiS"], parameters["qK"], parameters["phiK"]
        )[key_dtheta_dx]

        # modify the amplitudes by the derivative of the Ylms
        modified_amps = self.cache["teuk_modes"] * dYlm_dtheta[None, :] * dtheta_dx

        return modified_amps


class StableDerivative(DerivativeBase):
    """Semi-analytic derivative of the EMRI waveform.

    Instead of re-generating the waveform at each stencil offset, the
    trajectory and the mode amplitudes are differentiated numerically and the
    waveform derivative is assembled mode by mode. The derivatives with respect
    to the distance and the initial phases are analytic.

    With a response, the response is applied to the derivative of h. This
    leaves out the derivative of the response itself, which depends only on
    the sky angles, so for qS, phiS, qK and phiK the response-wrapped waveform
    R[h] is differentiated by finite differences instead.
    """

    supports_response = True
    analytic_params = frozenset({"dist", "Phi_phi0", "Phi_theta0", "Phi_r0"})
    sky_params = ("qS", "phiS", "qK", "phiK")

    def __init__(
        self,
        waveform_class,
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=None,
        ResponseWrapper=None,
        ResponseWrapper_kwargs=None,
        use_gpu=False,
    ):
        super().__init__(
            waveform_class,
            waveform_generator=waveform_generator,
            waveform_generator_kwargs=waveform_generator_kwargs,
            ResponseWrapper=ResponseWrapper,
            ResponseWrapper_kwargs=ResponseWrapper_kwargs,
            use_gpu=use_gpu,
        )
        if waveform_generator_kwargs is None:
            waveform_generator_kwargs = {}
        if ResponseWrapper_kwargs is None:
            ResponseWrapper_kwargs = {}

        # FEW loads all of a model's amplitude data (gigabytes for the Kerr
        # models) when the waveform is built, so the engine reuses the waveform
        # of the base generator instead of building a second one
        waveform = getattr(self.base_generator, "waveform_generator", None)
        self.engine = _StableWaveformDerivative(
            waveform_class=(
                waveform_class if waveform is None else lambda *args, **kwargs: waveform
            ),
            **waveform_generator_kwargs,
        )
        if self.has_response:
            # applies the response to the derivative of h
            self._engine_call = ResponseWrapper(self.engine, **ResponseWrapper_kwargs)
        else:
            self._engine_call = self.engine

    @classmethod
    def supports_waveform(cls, waveform_class):
        # the semi-analytic derivative needs a mode-by-mode waveform model
        return waveform_class.__name__ != "Pn5AAKWaveform"

    @property
    def npad(self):
        if self.engine.cache is None:
            return 0
        return self.engine.cache["Npad"]

    def clear_cache(self):
        self.engine.clear_cache()

    def set_reference(self, parameters, waveform=None, **waveform_kwargs):
        # the cached trajectory and modes belong to the previous reference point
        self.clear_cache()
        super().set_reference(parameters, waveform, **waveform_kwargs)

    def compute(self, param_to_vary, delta, order, kind):
        if self.has_response and param_to_vary in self.sky_params:
            return self._finite_difference(param_to_vary, delta, order, kind)
        # the parameters are also passed positionally, for the response wrapper
        return self._engine_call(
            *self.parameters.values(),
            parameters=self.parameters,
            param_to_vary=param_to_vary,
            delta=delta,
            order=order,
            kind=kind,
            **self.waveform_kwargs,
        )

