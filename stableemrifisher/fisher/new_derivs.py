from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform
from few.waveform.base import SphericalHarmonicWaveformBase
from few.utils.constants import Gpc, MRSUN_SI, YRSID_SI
import numpy as np
import matplotlib.pyplot as plt

class StableEMRIDerivative(GenerateEMRIWaveform):
    """
        inherits from the GenerateEMRIWaveform class of FEW, adds functions for derivative calculation and a fresh __call__ method.
    """
    def __init__(
        self,
        waveform_class,
        *args,
        frame: str = "detector",
        return_list = False,
        flip_output = False,
        **kwargs,
    ):
        super().__init__(waveform_class=waveform_class, 
                         *args, 
                         frame=frame, 
                         return_list=return_list, 
                         flip_output=flip_output, 
                         **kwargs) #initialize GenerateEMRIWaveform
        self.cache = None #initialize waveform cache

    def __getattr__(self, name):
        # get_attributes from self.waveform_generator if not found in GenerateEMRIWaveform
        return getattr(self.waveform_generator, name)

    def __call__(
        self,
        m1,
        m2,
        a,
        p0,
        e0,
        x0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
        *add_args,
        **kwargs, #should have parameters (dict), param_name (str), deltas (np.ndarray), stencil_fun (object) 
    ):
        """
        generate the waveform derivative for the given parameter
        """

        try:
            parameters = kwargs['parameters']
            deriv_parameter = kwargs['deriv_parameter']
            deltas = kwargs['deltas']
            stencil_fun = kwargs['stencil_fun']
            
        except KeyError as e:
            raise ValueError(f"kwargs must include {e}")

        # how we proceed depends on the parameter we are differentiating with respect to
        if deriv_parameter not in parameters:
            raise ValueError(f"Parameter '{deriv_parameter}' not in parameters dictionary.")

        try:
            T = kwargs['T']
        except KeyError:
            T = 1.0

        try:
            dt = kwargs['dt']
        except KeyError:
            dt = 10.0

        #obtain source frame angles
        theta_source, phi_source = self._get_viewing_angles(parameters['qS'],
                                                            parameters['phiS'],
                                                            parameters['qK'],
                                                            parameters['phiK'])
        parameters['theta_source'] = theta_source
        parameters['phi_source'] = phi_source

        keys_exclude = ['parameters', 'deriv_parameter', 'deltas', 'stencil_fun', 'T', 'dt']
        kwargs_remaining = {key: value for key, value in kwargs.items() if key not in keys_exclude}

        #get waveform
        if self.cache is None or parameters != self.cache['parameters']:
            t, y = self._trajectory_from_parameters(parameters, T)

            self.cache = {
                't':t,
                'y':y,
                'parameters':parameters,
                'coefficients':self.inspiral_generator.integrator_spline_coeff,
                'phase_coefficients':self.xp.asarray(self.inspiral_generator.integrator_spline_phase_coeff)[:, [0,2], :],
                'phase_coefficients_t':self.xp.asarray(self.inspiral_generator.integrator_spline_t),
            }

            amps_here = self._amplitudes_from_trajectory(parameters, t = t, y = y, cache=True, **kwargs_remaining)

            #create waveform at injection
            waveform_source = self.create_waveform(
                self.cache['t'],
                amps_here, #actually teuk_amps * Ylms_in
                self.cache['dummy_ylms'], #a bunch of ones
                self.cache['phase_coefficients_t'],
                self.cache['phase_coefficients'],
                self.cache['ls_all'],
                self.cache['ms_all'],
                #self.cache['ks_all'], #no inclination in FEW 2.0 :(
                self.cache['ns_all'],
                dt=dt,
                T=T,
                **kwargs_remaining,
            )

            self.cache['waveform_source'] = waveform_source #source_frame waveform

        # now calculate the derivatives with respect to the chosen parameters

        # distance
        if deriv_parameter == 'dist':
            waveform_derivative_source = -self.cache['waveform_source'] / parameters['dist']

        # phases
        elif deriv_parameter in ['Phi_phi0', 'Phi_theta0', 'Phi_r0']:
            # factor of -1j*m is applied to each amplitude
            modified_amps = self._modify_amplitudes_for_initial_phase_derivative(deriv_parameter)
            
            # create waveform derivative
            waveform_derivative_source = self.create_waveform(
                self.cache['t'],
                modified_amps,
                self.cache['dummy_ylms'],
                self.cache['phase_coefficients_t'],
                self.cache['phase_coefficients'],
                self.cache['ls_all'],
                self.cache['ms_all'],
                #self.cache['ks_all'],
                self.cache['ns_all'],
                dt=dt,
                T=T,
                **kwargs_remaining,
            )

        # sky angles (SSB)
        elif deriv_parameter in ['qS', 'phiS', 'qK', 'phiK']:
            # finite differencing of the ylms w.r.t. theta, then chain rule partial h / partial theta * partial theta / partial angle
            modified_amps = self._modify_amplitudes_for_angle_derivative(parameters, deriv_parameter, deltas, stencil_fun)
            
            waveform_derivative_source = self.create_waveform(
                    self.cache['t'],
                    modified_amps,
                    self.cache['dummy_ylms'],
                    self.cache['phase_coefficients_t'],
                    self.cache['phase_coefficients'],
                    self.cache['ls_all'],
                    self.cache['ms_all'],
                    #self.cache['ks_all'],
                    self.cache['ns_all'],
                    dt=dt,
                    T=T,
                    **kwargs_remaining,
                )

        # traj params
        else:
            #if you've reached this point, a derivative w.r.t one of the trajectory parameters is requested.
            #trajectory must be modified
            #get trajectories
            y_interps = self.xp.full((len(deltas), self.cache['t'].size, len(self.cache['y'])), self.xp.nan) #trajectory for each of the finite difference deltas
            
            for k, delt in enumerate(deltas):
                parameters_in = parameters.copy()
                parameters_in[deriv_parameter] += delt #perturb by finite-difference
                t, y = self._trajectory_from_parameters(parameters_in, T)
                #re-interpolate onto the time-step grid for the injection trajectory
                t_interp = self.cache['t'].copy()

                if t_interp[-1] > t[-1]: #the perturbed trajectory is plunging. Add NaN's at the end!
                    mask_notplunging = t_interp < t[-1] #for all t_interp < t[-1], the perturbed trajectory is still not plunging
                    t_interp = t_interp[mask_notplunging]
                
                y_interp = self.xp.asarray(self.inspiral_generator.inspiral_generator.eval_integrator_spline(t_interp).T) #any unfilled elements due to plunging trajectories assume nans.
                y_interps[k,:len(t_interp)] = y_interp.T

                #print("y_interps[k][xI] (before removing nans): ", y_interps[k,:,2])
                
            # In the case of plunge, some trajectories will be shorter than others. They appear as NaN's in the y_interps array
            nans = self.xp.isnan(y_interps)
            #print("nans:", nans)
            if nans.any():
                max_ind = self.xp.where(nans.sum(2).sum(0) > 0)[0].min()
            else:
                max_ind = y_interp.shape[1]
            #print("max_ind: ", max_ind, "y_interp.shape: ", y_interp.shape[1])
            
            # modify size of the trajectories and phases accordingly
            t_interp = self.cache['t'][:max_ind]   
            y_interps = y_interps[:, :max_ind, :]
            phases_steps = y_interps[:,:,3:6] #Phi_phi, Phi_theta, Phi_r
            phase_coefficients = self.xp.asarray(self.cache['phase_coefficients'][:max_ind - 1, :])
            phase_t = self.xp.asarray(self.cache['phase_coefficients_t'][:max_ind])
            
            #finite differencing the phases
            dPhi_fund_dx = stencil_fun(phases_steps, deltas[1] - deltas[0])
            #project the fundamental phases up to the full mode index space
            dPhi_dx = (dPhi_fund_dx[:,0,None] * self.cache['ms_all'][None,:] + 
                       #dPhi_fund_dx[:,1,None] * self.cache['ks_all'][None,:] + #no inclination in FEW 2.0
                       dPhi_fund_dx[:,2,None] * self.cache['ns_all'][None,:])

            #get amplitude derivative
            amps_steps = self.xp.zeros((len(deltas), max_ind,  len(self.cache['ls_all'])), dtype=self.xp.complex128)

            for k, delt in enumerate(deltas):

                #### DO WE NEED TO ITERATE? NOT IN OG CODE FROM CHRISTIAN ###########
                parameters_in = parameters.copy()
                parameters_in[deriv_parameter] += delt #perturb by finite-difference
                #####################################################################
                #print("y_interps[k][xI]: ", y_interps[k].T[2])
                amps_here = self._amplitudes_from_trajectory(parameters_in, t_interp, y_interps[k].T, cache=False, **kwargs_remaining) #remember, this function multiplies by Ylmns!
                amps_steps[k] = amps_here

            #finite differencing the amplitudes
            dAmp_dx = stencil_fun(amps_steps, deltas[1] - deltas[0]) #actually d teuk_amps/dx * Ylmns

            #defining effective amplitudes = dAdx - i A dPhidx
            wave_amps = self.cache['teuk_modes_with_ylms'][:max_ind]
            effective_amps = dAmp_dx - 1j * wave_amps * dPhi_dx

            if deriv_parameter in ['m1', 'm2']: #additional term due to chain rule of dist_dimensionless
                mu = parameters['m1'] * parameters['m2'] / (parameters['m1'] + parameters['m2'])
                M = paramaters['m1'] + parameters['m2']
                if deriv_parameter == 'm1': 
                    dmu_dm = parameters['m2'] ** 2 / (M ** 2)
                else:
                    dmu_dm = parameters['m1'] ** 2 / (M ** 2)
                effective_amps += wave_amps / mu * dmu_dm

            #create derivative
            waveform_derivative_source = self.create_waveform(
                t_interp,
                effective_amps, #actually effective amplitudes * Ylmns
                self.cache['dummy_ylms'], #just a bunch of ones
                phase_t,
                phase_coefficients,
                self.cache['ls_all'],
                self.cache['ms_all'],
                #self.cache['ks_all'],
                self.cache['ns_all'],
                dt=dt,
                T=T,
                **kwargs_remaining,
            )

            # pad with zeroes if required to get back to the waveform length
            if max_ind < self.cache['t'].size:
                waveform_derivative_source = np.concatenate(
                    (
                        waveform_derivative_source, 
                        np.zeros((self.cache['waveform'].size - waveform_derivative_source.size), 
                                 dtype=waveform_derivative_source.dtype)
                    ), 
                axis=0)

        #waveform derivative obtained in the source frame. Now we must transform to detector (SSB) frame 
        #and apply antenna patterns (and its derivatives in case of qS, phiS, qK, phiK)

        if self.waveform_generator.frame == "source":
            waveform_derivative_source *= -1

        #decompose to plus and cross
        (waveform_derivative_source_plus, 
         waveform_derivative_source_cross) = (waveform_derivative_source.real,
                                              -waveform_derivative_source.imag)

        if self.frame == "source":
            if self.return_list is False:
                return waveform_derivative_source_plus - 1j * waveform_derivative_source_cross
            else:
                return [waveform_derivative_source_plus, waveform_derivative_source_cross]

        #detector frame requested; apply antenna pattern
        (waveform_derivative_detector_plus, 
         waveform_derivative_detector_cross) = self._to_SSB_frame(hp = waveform_derivative_source_plus, 
                                                                  hc = waveform_derivative_source_cross,
                                                                  qS = parameters['qS'], phiS = parameters['phiS'], 
                                                                  qK = parameters['qK'], phiK = parameters['phiK'])

        #if derivative is with respect to one of the angles, also need derivatives with antenna pattern
        if deriv_parameter in ['qS', 'phiS', 'qK', 'phiK']:
            antenna_derivs = fplus_fcros_derivs(qS, phiS, qK, phiK, with_respect_to=deriv_parameter)

            waveform_derivative_detector_plus += (antenna_derivs[f'dFplusI/d{deriv_parameter}'] * waveform_derivative_source_plus +
                                                  antenna_derivs[f'dFcrossI/d{deriv_parameter}'] * waveform_derivative_source_cross)

            waveform_derivative_detector_cross += (antenna_derivs[f'dFplusII/d{deriv_parameter}'] * waveform_derivative_source_plus +
                                                  antenna_derivs[f'dFcrossII/d{deriv_parameter}'] * waveform_derivative_source_cross)
        
        if self.return_list is False:
            return waveform_derivative_detector_plus - 1j * waveform_derivative_detector_cross
        else:
            return [waveform_derivative_detector_plus, waveform_derivative_detector_cross]
    
    def clear_cache(self):
        self.cache = None #reset cache
            
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
        
        traj = self.inspiral_generator(
            parameters['m1'],
            parameters['m2'],
            parameters['a'],
            parameters['p0'],
            parameters['e0'],
            parameters['xI0'],
            Phi_phi0 = parameters['Phi_phi0'],
            Phi_theta0 = parameters['Phi_theta0'],
            Phi_r0 = parameters['Phi_r0'],
            T = T,
            **self.inspiral_kwargs,
        ) #generate the trajectory
    
        t = traj[0]
        y = traj[1:]
    
        #convert for gpus
        t = self.xp.asarray(t)
        y = self.xp.asarray(y)
    
        return t, y

    def _amplitudes_from_trajectory(self, parameters, t, y, cache=False, **kwargs):
        """
        calculate the amplitudes (and ylms) from the trajectory.

        Args:
            parameters (dict): dictionary of trajectory parameters
            t (np.ndarray): array of time steps for trajectory
            y (np.ndarray): array of evolving parameters in the trajectory at time steps
            cache (bool): whether to cache info (True) or not (False)
        Returns:
            Teukolsky amplitudes times Ylms
        """

        #if detector frame, scale by distance in the amplitude module as well.
        if self.frame == 'detector':
            mu = parameters['m1'] * parameters['m2'] / (parameters['m1'] + parameters['m2'])
            dist_dimensionless = (parameters['dist'] * Gpc) / (mu * MRSUN_SI)
        else:
            dist_dimensionless = 1.0 
            
        if cache:
            mode_selection = None
        else:
            mode_selection = self.cache['mode_selection']

        #get teuk amplitudes, ylms, ls, ms, ks, and ns from the mode_selector module.

        #ylms
        ylms = self.ylm_gen(self.unique_l, self.unique_m, parameters['theta_source'], parameters['phi_source']).copy()[self.inverse_lm]
        
        # amplitudes
        teuk_modes = self.xp.asarray(
            self.amplitude_generator(parameters['a'], *y[:3])
        )

        modeinds = [self.l_arr, 
                    self.m_arr,
                    self.n_arr]
        modeinds_map = self.special_index_map_arr
        
        (
            teuk_modes_in,
            ylms_in,
            self.ls,
            self.ms,
            self.ns,
        ) = self.mode_selector(
            teuk_modes,
            ylms,
            modeinds,
            mode_selection = mode_selection, #None in first pass, but selects a given set of modes in subsequent passes
            modeinds_map = modeinds_map, #only used when mode_selection is a list.
            **kwargs
        )

        if cache:
            #we don't use mode symmetry
            m0mask = self.ms != 0
            teuk_modes_in = self.xp.concatenate(
                (teuk_modes_in, (-1)**(self.ls[m0mask])*self.xp.conj(teuk_modes_in[:, m0mask])), axis=1
            )

            ylms_in = self.xp.concatenate(
                (ylms_in[:self.ls.size], ylms_in[self.ls.size:][m0mask]), axis=0
            )

            self.cache['ls_all'] = self.xp.concatenate(
                (self.ls, self.ls[m0mask]),
                axis=0
            )
            self.cache['ms_all'] = self.xp.concatenate(
                (self.ms, -self.ms[m0mask]),
                axis=0
            )

            ######### NO INCLINATION IN FEW 2.0 :( ################
            #self.cache['ks_all'] = self.xp.concatenate(
            #    (self.ks, -self.ks[m0mask]),
            #    axis=0
            #)
            #######################################################
            
            self.cache['ns_all'] = self.xp.concatenate(
                (self.ns, -self.ns[m0mask]),
                axis=0
            )

            self.cache['ls'] = self.ls.copy()
            self.cache['ms'] = self.ms.copy()
            #self.cache['ks'] = self.ks.copy()
            self.cache['ns'] = self.ns.copy()

            self.cache['mode_selection'] = [
                (l, 
                 m, 
                 n) for l, m, n in zip(
                    self.cache['ls'], 
                    self.cache['ms'], 
                    self.cache['ns']
                )
            ]

            self.cache['teuk_modes'] = teuk_modes_in / dist_dimensionless
            self.cache['teuk_modes_with_ylms'] = self.cache['teuk_modes'] * ylms_in
            self.cache['ylms_in'] = ylms_in
            self.cache['dummy_ylms'] = self.xp.ones(2 * teuk_modes_in.shape[1], dtype=self.xp.complex128)
            self.cache['dummy_ylms'][teuk_modes_in.shape[1]:] = 0.0
            self.cache['m0mask'] = m0mask

            return self.cache['teuk_modes_with_ylms']
                
        else:
            teuk_modes_in = self.xp.concatenate(
                (teuk_modes_in, (-1)**(self.ls[self.cache['m0mask']])*self.xp.conj(teuk_modes_in[:, self.cache['m0mask']])), axis=1
            )

            ylms_in = self.xp.concatenate(
                (ylms_in[:self.ls.size], ylms_in[self.ls.size:][self.cache['m0mask']]), axis=0
            )

            return teuk_modes_in * ylms_in / dist_dimensionless

    def _modify_amplitudes_for_initial_phase_derivative(self, deriv_parameter):
        """ 
        calculates modified amplitudes for phase derivatives.

        Args:
            deriv_parameter (string): one of "Phi_phi0", "Phi_theta0", "Phi_r0"
        returns
            modified amplitudes = -i (mode_index) A
        """

        if deriv_parameter == 'Phi_phi0':
            factor = -1j * self.cache['ms_all']
        elif deriv_parameter == 'Phi_theta0':
            raise NotImplementedError #no inclination in FEW 2.0
        elif deriv_parameter == 'Phi_r0':
            factor = -1j * self.cache['ns_all']

        modified_amps = self.cache['teuk_modes_with_ylms'] * factor[None, :]
        return modified_amps

    @staticmethod
    def _viewing_angle_partials(qS, phiS, qK, phiK):
        """
        Compute partial derivatives of theta_src = arccos(-R · S) with respect to 
        detector-frame angles.

        Parameters
        ----------
        qS : float
            Polar angle theta_S (radians)
        phiS : float  
            Azimuthal angle phi_S (radians)
        qK : float
            Polar angle theta_K (radians)
        phiK : float
            Azimuthal angle phi_K (radians)

        Returns
        -------
        dict
            Dictionary containing:
            - "theta_src": The viewing angle theta_src
            - "del theta_src / del phi_S": Partial derivative w.r.t. phi_S
            - "del theta_src / del qS ": Partial derivative w.r.t. qS
            - "del theta_src / del qK": Partial derivative w.r.t. qK
            - "del theta_src / del phi_K": Partial derivative w.r.t. phi_K

        Raises
        ------
        ValueError
            If sin(theta_src) is zero (derivatives undefined at theta_src = 0 or π)

        Notes
        -----
        Uses u = cos(theta_src) = -[sin qS sin qK cos(phiS-phiK) + cos qS cos qK].
        For numerical stability, sin(theta_src) is computed from u via sqrt(1 - u²).
        """
        sS, cS = np.sin(qS), np.cos(qS)
        sK, cK = np.sin(qK), np.cos(qK)

        # u = cos theta_S = -(R . S) in FEW
        u = -(sS * sK * np.cos(phiS - phiK) + cS * cK)

        theta_src = np.arccos(u)
        sin_theta_src_sq = 1.0 - u * u
        sin_theta_src = np.sqrt(np.maximum(0.0, sin_theta_src_sq))

        # Handle singular geometry: sin(theta_src) == 0
        # (theta_src == 0 or pi) -> derivatives undefined
        if sin_theta_src == 0.0:
            raise ValueError("sin(theta_src) is zero, derivatives are undefined.")
        else:
            # ∂u/∂x pieces from the algebra
            du_dphiS =  sS * sK * np.sin(phiS - phiK)
            du_dqS   =  sS * cK - cS * sK * np.cos(phiS - phiK)
            du_dqK   = -sS * cK * np.cos(phiS - phiK) + cS * sK
            du_dphiK = -sS * sK * np.sin(phiS - phiK)

            inv_sin = 1.0 / sin_theta_src # Blows up as theta_src = 0,pi. Careful. 
            # d theta_src / d x = -(1/sin theta_src) * du/dx
            d_theta_d_phiS = -inv_sin * du_dphiS
            d_theta_d_qS   = -inv_sin * du_dqS
            d_theta_d_qK   = -inv_sin * du_dqK
            d_theta_d_phiK = -inv_sin * du_dphiK

        return {
            "theta_src": theta_src,
            "del theta_src / del phi_S": d_theta_d_phiS,
            "del theta_src / del qS ": d_theta_d_qS,      # key as requested
            "del theta_src / del qK": d_theta_d_qK,
            "del theta_src / del phi_K": d_theta_d_phiK,
        }


    @staticmethod
    def _fplus_fcros_derivs(qS, phiS, qK, phiK,
                        with_respect_to=None):
        """
        Compute psi, (FplusI,FcrosI,FplusII,FcrosII) and their partial derivatives.

        Parameters
        ----------
        qS, phiS, qK, phiK : float
            Detector-frame angles.
        hp, hc : float, optional
            Polarizations before rotation. If given, rotated strains are returned.
        d_hp, d_hc : dict, optional
            Dictionaries mapping x -> derivative of hp/hc w.r.t x.
        with_respect_to : list of str, optional
            Subset of parameters to differentiate with respect to.
            Allowed values: 'qS','phiS','qK','phiK'.
            If None, defaults to all four.

        Returns
        -------
        dict
            Contains psi, FplusI, FcrosI, FplusII, FcrosII, and only the
            requested derivatives.
        """
        cS, sS = np.cos(qS), np.sin(qS)
        cK, sK = np.cos(qK), np.sin(qK)
        dphi = phiS - phiK
        cdp, sdp = np.cos(dphi), np.sin(dphi)

        # u, v, psi
        u = cS * sK * cdp - cK * sS
        v = sK * sdp
        psi = -np.arctan2(u, v)

        c2p, s2p = np.cos(2*psi), np.sin(2*psi)
        FplusI, FcrosI = c2p, -s2p
        FplusII, FcrosII = s2p, c2p

        # du/dx, dv/dx
        du = {
            'qS':   -sS * sK * cdp - cK * cS,
            'phiS': -cS * sK * sdp,
            'qK':    cS * cK * cdp + sK * sS,
            'phiK':  cS * sK * sdp,
        }
        dv = {
            'qS':    0.0,
            'phiS':  sK * cdp,
            'qK':    cK * sdp,
            'phiK': -sK * cdp,
        }

        denom = u*u + v*v

        out = {
            'psi': psi,
            'FplusI': FplusI, 'FcrosI': FcrosI,
            'FplusII': FplusII, 'FcrosII': FcrosII,
        }

        if with_respect_to is None:
            with_respect_to = ['qS','phiS','qK','phiK']

        # Loop only over requested derivatives
        for x in with_respect_to:
            dpsi_x = - (v*du[x] - u*dv[x]) / denom
            # I-arm
            out[f'dFplusI/d{x}']  = -2.0 * s2p * dpsi_x
            out[f'dFcrosI/d{x}']  = -2.0 * c2p * dpsi_x
            # II-arm
            out[f'dFplusII/d{x}'] =  2.0 * c2p * dpsi_x
            out[f'dFcrosII/d{x}'] = -2.0 * s2p * dpsi_x
        return out

    def _modify_amplitudes_for_angle_derivative(self, parameters, deriv_parameter, deltas, stencil_fun):
        """
        calculates modified amplitudes for angle derivatives (qS, phiS, qK, phiK)

        Args:
            parameters (dict): model parameters
            deriv_parameter (str): one of qS, phiS, qK, phiK: parameter with respect to which to calculate the derivative
            deltas (np.ndarray): grid of deltas for stencil_fun
            stencil_fun (Callable): maps deltas to finite differences
        Returns:
            modified amplitudes = A * partial Y_lm / partial theta * partial_theta / partial kappa where kappa is the deriv_parameter
        """

        ylm_temp = np.zeros((len(deltas), self.cache['ls_all'].size), dtype=np.complex128)

        #first calculate dylm_dtheta
        for k, delt in enumerate(deltas):
            parameters_in = parameters.copy()
            parameters_in['theta_source'] += delt
            # get the ylms for this theta
            ylm_temp[k] = self.ylm_gen(self.cache['ls_all'], self.cache['ms_all'], parameters_in['theta_source'], parameters_in['phi_source'])

        dYlm_dtheta = stencil_fun(ylm_temp, deltas[1] - deltas[0])

        #now calculate dtheta_dkappa
        if deriv_parameter == 'qS':
            key_dtheta_dkappa = "del theta_src / del qS"
        elif deriv_parameter == 'phiS':
            key_dtheta_dkappa = "del theta_src / del phi_S"
        elif deriv_parameter == 'qK':
            key_dtheta_dkappa = "del theta_src / del qK"
        elif deriv_parameter == 'phiK':
            key_dtheta_dkappa = "del theta_src / del phi_K"
        
        dtheta_dkappa = self._viewing_angle_partials(qS, phiS, qK, phiK)[key_dtheta_dkappa]

        # modify the amplitudes by the derivative of the Ylms
        modified_amps = self.cache['teuk_modes'] * dYlm_dtheta[None, :] * dtheta_dkappa

        return modified_amps