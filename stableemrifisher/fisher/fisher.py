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
import time 
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

try:
    import cupy as cp
except ImportError:
    cp = None

from few.utils.constants import YRSID_SI
from stableemrifisher.fisher.derivatives import derivative, handle_a_flip
from stableemrifisher.utils import inner_product, SNRcalc, generate_PSD, fishinv
from stableemrifisher.noise import noise_PSD_AE 
from stableemrifisher.plot import CovEllipsePlot, StabilityPlot

import logging
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

    Attributes (selection):
        waveform (np.ndarray or cp.ndarray): Cached waveform (channels x N).
        waveform_generator: FEW `GenerateEMRIWaveform` or a `ResponseWrapper`.
        channels (list[str]): Channel names used to build PSDs and products.
        deltas (dict[str, float] | None): Per-parameter finite-difference
            step sizes. Computed if not provided.
        param_names (list[str]): Parameter names corresponding to Fisher order.
        npar (int): Number of parameters in the Fisher matrix.
        SNR2 (float): SNR squared of the current waveform (set after call).
    """
    
    def __init__(self, m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                 Phi_phi0, Phi_theta0, Phi_r0, dt = 10., T = 1.0, add_param_args = None, waveform_kwargs=None, EMRI_waveform_gen = None, window = None, fmin = None, fmax = None, noise_model = noise_PSD_AE, noise_kwargs={"TDI":'TDI1'}, channels=["A","E"],
                 param_names=None, deltas=None, der_order=2, Ndelta=8, delta_range = None, CovEllipse=False, stability_plot=False, save_derivatives=False,
                 live_dangerously = False, plunge_check=True, filename=None, suffix=None, stats_for_nerds=False, use_gpu=False):
        """Initialize a Fisher-matrix computation for an EMRI configuration.

        Args:
            m1 (float): Central (MBH) mass.
            m2 (float): Compact object mass.
            a (float): Dimensionless spin of the MBH.
            p0 (float): Initial semi-latus rectum.
            e0 (float): Initial eccentricity.
            Y0 (float): Initial cosine of the inclination of the CO orbit
                relative to the equatorial plane.
            dist (float): Luminosity distance in Gpc.
            qS (float): Sky polar angle (detector frame), radians.
            phiS (float): Sky azimuthal angle (detector frame), radians.
            qK (float): Spin-axis polar angle, radians.
            phiK (float): Spin-axis azimuthal angle, radians.
            Phi_phi0 (float): Initial azimuthal phase, radians.
            Phi_theta0 (float): Initial polar phase, radians.
            Phi_r0 (float): Initial radial phase, radians.
            dt (float, optional): Time step of the waveform in seconds.
            T (float, optional): Duration of the evolution in years.
            add_param_args (dict[str, float] | None, optional): Extra model
                parameters (e.g., beyond-vacuum GR). Added to waveform and
                trajectory parameter sets.
            waveform_kwargs (dict | None, optional): Additional kwargs passed
                to the waveform generator.
            EMRI_waveform_gen (object, optional): FEW `GenerateEMRIWaveform`
                instance or a `ResponseWrapper` that applies a LISA response.
            window (np.ndarray | None, optional): Optional window to apply in
                inner products.
            fmin (float | None, optional): Minimum frequency for the
                frequency-domain inner product.
            fmax (float | None, optional): Maximum frequency for the
                frequency-domain inner product.
            noise_model (callable, optional): Function S_n(f, **noise_kwargs)
                returning the one-sided noise PSD per channel.
            noise_kwargs (dict, optional): Keyword arguments for `noise_model`.
            channels (list[str], optional): Response channel names.
            param_names (list[str] | None, optional): Ordered parameter names
                used to build derivatives and the Fisher matrix.
            deltas (dict[str, float] | None, optional): Fixed finite-difference
                step sizes. If None, stable values are searched.
            der_order (int, optional): Order of the finite-difference scheme.
            Ndelta (int, optional): Number of trial step sizes per parameter
                when searching for stability.
            delta_range (dict[str, list[float]] | None, optional): Optional
                per-parameter grid of deltas to test.
            CovEllipse (bool, optional): If True, compute and optionally plot
                the covariance matrix after the Fisher is built.
            stability_plot (bool, optional): If True, plot per-parameter
                stability curves during delta search.
            save_derivatives (bool, optional): If True, save the derivative
                stack to the HDF5 file alongside the Fisher matrix.
            live_dangerously (bool, optional): If True, skip the stability
                search and use heuristic step sizes.
            plunge_check (bool, optional): If True, reduce the evolution time
                if the computed trajectory indicates a plunge.
            filename (str | None, optional): Directory to write outputs (text,
                HDF5, plots). If None, nothing is written.
            suffix (str | None, optional): Suffix used to disambiguate files
                under `filename`.
            stats_for_nerds (bool, optional): If True, enable verbose DEBUG
                logging.
            use_gpu (bool, optional): If True, prefer CuPy for array ops where
                supported.

        Raises:
            ValueError: If `param_names` is None or the waveform generator is
                not provided.
        """
        self.waveform = None

        self.use_gpu = use_gpu

        #initializing param_names list
        if param_names == None: # TODO should this just operate over all parameters by default?
            raise ValueError("param_names cannot be empty.")

        else:
            self.param_names = param_names
            self.npar = len(self.param_names)

        if deltas != None and len(deltas) != self.npar:
            logger.critical('Length of deltas array should be equal to length of param_names.\n\
                   Assuming deltas = None.')
            deltas = None
            
        if EMRI_waveform_gen == None:
            raise ValueError("Please set up EMRI waveform model and pass as argument.")
         
        #initializing parameters
        self.dt = dt
        self.T = T

        # Initilising FM details
        self.order = der_order
        self.Ndelta = Ndelta
        self.window = window
        self.fmin = fmin
        self.fmax = fmax
        
        if stats_for_nerds:
            logger.setLevel("DEBUG")

        # =============== Initialise Waveform generator ================
        self.waveform_generator = EMRI_waveform_gen

        # Determine what version of TDI to use or whether to use the LWA 

        self.noise_model = noise_model
        self.noise_kwargs = noise_kwargs
        self.channels = channels
	
        if waveform_kwargs is None:
            waveform_kwargs = {}

        self.waveform_kwargs = waveform_kwargs

        #if self.response in ["TDI1", "TDI2"]:
        
        try: #if ResponseWrapper is provided
            self.traj_module = self.waveform_generator.waveform_gen.waveform_generator.inspiral_generator
            self.traj_module_func = self.waveform_generator.waveform_gen.waveform_generator.inspiral_kwargs['func']
            self.ResponseWrapper = True
        except: #if GenerateEMRIWaveform is provided
            self.traj_module = self.waveform_generator.waveform_generator.inspiral_generator
            self.traj_module_func = self.waveform_generator.waveform_generator.inspiral_kwargs['func']
            self.ResponseWrapper = False
        
        #initializing param dictionary
        self.wave_params = {'m1':m1,
                      'm2':m2,
                      'a':a,
                      'p0':p0,
                      'e0':e0,
                      'Y0':Y0,
                      'dist':dist,
                      'qS':qS,
                      'phiS':phiS,
                      'qK':qK,
                      'phiK':phiK,
                      'Phi_phi0':Phi_phi0,
                      'Phi_theta0':Phi_theta0,
                      'Phi_r0':Phi_r0,
                      }

        self.minmax = {'a':[0.05,0.95],'e0':[0.01,0.7],'Phi_phi0':[0.1,2*np.pi*(0.9)],'Phi_r0':[0.1,2*np.pi*(0.9)],'Phi_theta0':[0.1,2*np.pi*(0.9)],
                              'qS':[0.1,np.pi*(0.9)],'qK':[0.1,np.pi*(0.9)],'phiS':[0.1,2*np.pi*(0.9)],'phiK':[0.1,2*np.pi*(0.9)]}
                                            
        self.traj_params = dict(list(self.wave_params.items())[:6]) 

        #initialise extra args, add them to wave_params/traj_params
        full_EMRI_param = list(self.wave_params.keys())
        if not add_param_args == None:
            for i in range(len(add_param_args.keys())):
                self.wave_params[list(add_param_args.keys())[i]] = list(add_param_args.values())[i]
                self.traj_params[list(add_param_args.keys())[i]] = list(add_param_args.values())[i]
                
        self.wave_params_list = list(self.wave_params.values())
        
        #initializing deltas
        self.deltas = deltas #Use deltas == None as a Flag
        
        if not delta_range == None: 
            self.delta_range = delta_range
        else:
            self.delta_range = {}
        
        #initializing other Flags:
        self.CovEllipse = CovEllipse
        self.stability_plot = stability_plot
        self.save_derivatives = save_derivatives
        self.filename = filename
        self.suffix = suffix
        self.live_dangerously = live_dangerously
        
        if plunge_check:
            # Redefine final time if small body is plunging. More stable FMs.
            final_time = self.check_if_plunging()
            self.T = final_time/YRSID_SI # Years
    
        self.waveform_kwargs.update(dict(dt=self.dt, T=self.T))


    def __call__(self):
        """Run the full pipeline: SNR, stable deltas (optional), Fisher (+cov).

        Workflow:
            1) Build the waveform and compute SNR (stored via `self.SNR2`).
            2) If `self.deltas` is None and `live_dangerously` is False,
               search for stable step sizes; otherwise use heuristics.
            3) Compute the Fisher matrix via inner products of derivatives.
            4) Optionally compute the covariance matrix and generate plots.

        Returns:
            numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray]:
                - Fisher matrix (npar x npar), or
                - (Fisher, Covariance) if `CovEllipse` is True.
        """
        
        rho = self.SNRcalc_SEF()

        self.SNR2 = rho**2

        logger.info(f'Waveform Generated. SNR: {rho}')
        
        if rho <= 20:
            logger.critical('The optimal source SNR is <= 20. The Fisher approximation may not be valid!')
        
        #making parent folder
        if self.filename != None:
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
                
        #1. If deltas not provided, calculating the stable deltas
        # print("Computing stable deltas")
        if self.live_dangerously == False:
            if self.deltas == None:
                start = time.time()
                self.Fisher_Stability() # Attempts to compute stable delta values. 
                end = time.time() - start
                logger.info(f"Time taken to compute stable deltas is {end} seconds")
                
        else:
            logger.debug("You have elected for dangerous living, I like it. ")
            fudge_factor_intrinsic = 3*(self.wave_params["m2"]/self.wave_params["m1"]) * (self.SNR2)**-1
            delta_intrinsic = fudge_factor_intrinsic * np.array([self.wave_params["m1"], self.wave_params["m2"], 1.0, 1.0, 1.0, 1.0])
            danger_delta_dict = dict(zip(self.param_names[0:7],delta_intrinsic))
            delta_dict_final_params = dict(zip(self.param_names[6:14],np.array(8*[1e-6])))
            danger_delta_dict.update(delta_dict_final_params)
            
            self.deltas = danger_delta_dict
            self.save_deltas()

        #2. Given the deltas, we calculate the Fisher Matrix
        start = time.time()
        Fisher = self.FisherCalc()
        end = time.time() - start
        logger.info(f"Time taken to compute FM is {end} seconds")
        
        #3. If requested, calculate the covariance Matrix
        if self.CovEllipse:
            covariance = np.linalg.inv(Fisher)
            # TODO just get the user to pass filename paths in for the plots etc. It's easier to develop and gives the user more control
            if self.filename != None:
                if self.suffix != None:
                    CovEllipsePlot(covariance, self.param_names, self.wave_params,  filename=os.path.join(self.filename, f"covariance_ellipses_{self.suffix}.png"))
                else:
                    CovEllipsePlot(covariance, self.param_names, self.wave_params, filename=os.path.join(self.filename, "covariance_ellipses.png"))                

            else:
                fig, axs = CovEllipsePlot(covariance, self.param_names, self.wave_params)
                plt.show()

            return Fisher, covariance
            
        else:
            return Fisher
        
    def SNRcalc_SEF(self):
        """Generate waveform and PSDs, then compute the optimal SNR.

        The waveform is obtained from `self.waveform_generator` using the
        parameters provided at construction time. If no response wrapper is
        used and a 1D waveform is returned (h+ - i hx), it is replicated across
        the configured channels with equal weighting. Per-channel PSDs are then
        generated and the multi-channel SNR is computed.

        Returns:
            float: The optimal SNR of the current configuration.
        """
    	#generate PSD
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        self.waveform = xp.asarray(self.waveform_generator(*self.wave_params_list, **self.waveform_kwargs))
        
        # If no response is provided and waveform of the form h+ - ihx, create copies equivalent to the number of channels.
        if self.waveform.ndim == 1:
            self.waveform = xp.asarray([self.waveform.copy() for _ in range(len(self.channels))])/len(self.channels) #we assume equal strength in all provided channels.
            
        print("wave ndim: ", self.waveform.ndim)
        #Generate PSDs
        self.PSD_funcs = generate_PSD(waveform=self.waveform, dt=self.dt, noise_PSD=self.noise_model,
                     channels=self.channels,noise_kwargs=self.noise_kwargs,use_gpu=self.use_gpu)
        
        # Compute SNR
        logger.info(f"Computing SNR for parameters: {self.wave_params}") 
        
        return SNRcalc(self.waveform, self.PSD_funcs, dt=self.dt, window=self.window, fmin = self.fmin, fmax = self.fmax, use_gpu=self.use_gpu)
        
    
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
        t_traj = self.traj_module(*traj_vals, Phi_phi0=self.wave_params["Phi_phi0"], 
                                        Phi_theta0=self.wave_params["Phi_theta0"], Phi_r0=self.wave_params["Phi_r0"], 
                                        T = self.T, dt = self.dt)[0] 

        if t_traj[-1] < self.T*YRSID_SI - 1.0: #1.0 is a buffer because self.traj_module can produce trajectories slightly smaller than T*YRSID_SI even if not plunging!
            logger.warning("Body is plunging! Expect instabilities.")
            final_time = t_traj[-1] - 6*60*60 # Remove 6 hours of final inspiral
            logger.warning(f"Removed last 6 hours of inspiral. New evolution time: {final_time/YRSID_SI} years")
        else:
            logger.info("Body is not plunging, Fisher should be stable.")
            final_time = self.T * YRSID_SI
        return final_time

    #defining Fisher_Stability function, generates self.deltas
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
        logger.info('calculating stable deltas...')
        Ndelta = self.Ndelta
        deltas = {}

        #waveform_generator = self.waveform_generator
        #waveform = self.waveform
        #PSD_funcs = self.PSD_funcs
        
        #if ResponseWrapper provided, strip it before calculating stable deltas. 
        #this should improve speed. We switch back to ResponseWrapper before the final Fisher calculation of course.
        if self.ResponseWrapper:
            waveform_generator = self.waveform_generator.waveform_gen #stripped waveform generator
            waveform = xp.asarray(waveform_generator(*self.wave_params_list, **self.waveform_kwargs))
            # stripped waveform of the form h+ - ihx, create copies equivalent to the number of channels.
            waveform = xp.asarray([waveform.copy() for _ in range(len(self.channels))])/len(self.channels) #we assume equal strength in all provided channels.
            PSD_funcs = generate_PSD(waveform=waveform, dt=self.dt,use_gpu=self.use_gpu, channels = self.channels) #produce an N-channel PSD assuming the default noise curve (noise_PSD_AE). Shouldn't matter much.
        else:
            waveform_generator = self.waveform_generator
            waveform = self.waveform
            PSD_funcs = self.PSD_funcs
            
        for i in range(len(self.param_names)):

            try:
                delta_init = self.delta_range[self.param_names[i]]
                
            except KeyError:

                # If a specific parameter equals zero, then consider stepsizes around zero.
                if self.wave_params[self.param_names[i]] == 0.0:
                    delta_init = np.geomspace(1e-4,1e-9,Ndelta)

                # Compute Ndelta number of delta values to compute derivative. Testing stability.
                elif self.param_names[i] == 'm1' or self.param_names[i] == 'm2': 
                    delta_init = np.geomspace(1e-4*self.wave_params[self.param_names[i]],1e-9*self.wave_params[self.param_names[i]],Ndelta)
                elif self.param_names[i] == 'a' or self.param_names[i] == 'p0' or self.param_names[i] == 'e0' or self.param_names[i] == 'Y0':
                    delta_init = np.geomspace(1e-4*self.wave_params[self.param_names[i]],1e-9*self.wave_params[self.param_names[i]],Ndelta)
                else:
                    delta_init = np.geomspace(1e-1*self.wave_params[self.param_names[i]],1e-6*self.wave_params[self.param_names[i]],Ndelta)
 
            Gamma = []
            orderofmag = []

            relerr_flag = False
            for k in range(Ndelta):
                if self.param_names[i] == 'dist':
                    del_k = derivative(waveform_generator, self.wave_params, self.param_names[i], delta_init[k], use_gpu=self.use_gpu, waveform=waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
                    
                    relerr_flag = True
                    deltas['dist'] = 0.0
                    break
                else:
                    # print("For a choice of delta =",delta_init[k])
                    
                    if self.param_names[i] in list(self.minmax.keys()):
                        if self.wave_params[self.param_names[i]] <= self.minmax[self.param_names[i]][0]:
                            del_k = derivative(waveform_generator, self.wave_params, self.param_names[i], delta_init[k], kind="forward", use_gpu=self.use_gpu, waveform=waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
                        elif self.wave_params[self.param_names[i]] > self.minmax[self.param_names[i]][1]:
                            del_k = derivative(waveform_generator, self.wave_params, self.param_names[i], delta_init[k], kind="backward", use_gpu=self.use_gpu, waveform=waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
                        else:
                            del_k = derivative(waveform_generator, self.wave_params, self.param_names[i], delta_init[k], use_gpu=self.use_gpu, waveform=waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
                    else:
                        del_k = derivative(waveform_generator, self.wave_params, self.param_names[i], delta_init[k], use_gpu=self.use_gpu, waveform=waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)

                # If no response is provided and waveform of the form h+ - ihx, create copies equivalent to the number of channels.
                if del_k.ndim == 1:
                    del_k = xp.asarray([del_k.copy() for _ in range(len(self.channels))])/len(self.channels) #we assume equal strength in all provided channels.

                #Calculating the Fisher Elements
                Gammai = inner_product(del_k,del_k, PSD_funcs, self.dt, window=self.window, fmin = self.fmin, fmax = self.fmax, use_gpu=self.use_gpu)
                logger.debug(f"Gamma_ii: {Gammai}")
                if np.isnan(Gammai):
                    Gamma.append(0.0) #handle nan's
                    logger.warning('NaN type encountered during Fisher calculation! Replacing with 0.0.')	
                else:
                    Gamma.append(Gammai)

            if relerr_flag == False:
                if self.use_gpu:
                    Gamma = xp.asnumpy(xp.array(Gamma))
                else:
                    Gamma = xp.array(Gamma)
                
                if (Gamma[1:] == 0.).all(): #handle non-contributing parameters
                    relerr = np.ones(len(Gamma)-1)
                else:
                    relerr = []
                    for m in range(len(Gamma)-1): 
                        if Gamma[m+1] == 0.0: #handle partially null contributors
                            relerr.append(1.0)
                        else:
                            relerr.append(np.abs(Gamma[m+1] - Gamma[m])/Gamma[m+1])   

                logger.debug(relerr)
                
                relerr_min_i, = np.where(np.isclose(relerr, np.min(relerr),rtol=1e-1*np.min(relerr),atol=1e-1*np.min(relerr)))
                if len(relerr_min_i) > 1:
                    relerr_min_i = relerr_min_i[-1]

                logger.debug(relerr_min_i)
                
                if np.min(relerr) >= 0.01:
                    logger.warning(f'minimum relative error is greater than 1% for {self.param_names[i]}. Fisher may be unstable!')

                deltas[self.param_names[i]] = delta_init[relerr_min_i].item()
                
                if self.stability_plot:
                    if self.filename != None:
                        if self.suffix != None:
                            StabilityPlot(delta_init,Gamma,param_name=self.param_names[i],filename=os.path.join(self.filename,f'stability_{self.suffix}_{self.param_names[i]}.png'))
                        else:
                            StabilityPlot(delta_init,Gamma,param_name=self.param_names[i],filename=os.path.join(self.filename,f'stability_{self.param_names[i]}.png'))
                    else:
                        StabilityPlot(delta_init,Gamma,param_name=self.param_name[i])
        logger.debug(f'stable deltas: {deltas}')
        
        self.deltas = deltas
        self.save_deltas()

    def save_deltas(self):
        """Persist the currently selected `self.deltas` to disk (if configured).

        Writes a small text file into `self.filename` containing the string
        representation of the `self.deltas` dictionary. If no output directory
        is configured, this function does nothing.
        """
        if self.filename is not None:
            if self.suffix != None:
                with open(f"{self.filename}/stable_deltas_{self.suffix}.txt", "w", newline="") as file:
                    file.write(str(self.deltas))
            else:
                with open(f"{self.filename}/stable_deltas.txt", "w", newline="") as file:
                    file.write(str(self.deltas))

    #defining FisherCalc function, returns Fisher
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

        logger.info('calculating Fisher matrix...')
 
        Fisher = np.zeros((self.npar,self.npar), dtype=np.float64)
        dtv = []
        for i in range(self.npar):

            if self.param_names[i] in list(self.minmax.keys()):
                if self.wave_params[self.param_names[i]] <= self.minmax[self.param_names[i]][0]:
                    dtv_i = derivative(self.waveform_generator, self.wave_params, self.param_names[i], self.deltas[self.param_names[i]], kind="forward", waveform=self.waveform, order=self.order, use_gpu=self.use_gpu, waveform_kwargs=self.waveform_kwargs)
                elif self.wave_params[self.param_names[i]] > self.minmax[self.param_names[i]][1]:
                    dtv_i = derivative(self.waveform_generator, self.wave_params, self.param_names[i],self.deltas[self.param_names[i]], kind="backward", waveform=self.waveform, order=self.order, use_gpu=self.use_gpu, waveform_kwargs=self.waveform_kwargs)
                else:
                    dtv_i = derivative(self.waveform_generator, self.wave_params, self.param_names[i],self.deltas[self.param_names[i]],use_gpu=self.use_gpu, waveform=self.waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
            else:
                dtv_i = derivative(self.waveform_generator, self.wave_params, self.param_names[i], self.deltas[self.param_names[i]],use_gpu=self.use_gpu, waveform=self.waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
            
            # If no response is provided and waveform of the form h+ - ihx, create copies equivalent to the number of channels.
            if dtv_i.ndim == 1:
                dtv_i = xp.asarray([dtv_i.copy() for _ in range(len(self.channels))])/len(self.channels) #we assume equal strength in all provided channels.

            dtv.append(dtv_i)

        logger.info("Finished derivatives")
        
        if self.save_derivatives:
            dtv_save = xp.asarray(dtv)
            if self.use_gpu:
                dtv_save = xp.asnumpy(dtv_save)
            if not self.filename == None:
                if not self.suffix == None:
                    with h5py.File(f"{self.filename}/Fisher_{self.suffix}.h5", "w") as f: 
                        f.create_dataset("derivatives",data=dtv_save)
                else:
                    with h5py.File(f"{self.filename}/Fisher.h5", "w") as f:
                        f.create_dataset("derivatives",data=dtv_save)

        for i in range(self.npar):
            for j in range(i,self.npar):
                if self.use_gpu:
                    Fisher[i,j] = np.float64(xp.asnumpy(inner_product(dtv[i],dtv[j],self.PSD_funcs, self.dt, window=self.window,  fmin = self.fmin, fmax = self.fmax, use_gpu=self.use_gpu).real))
                else:
                    Fisher[i,j] = np.float64((inner_product(dtv[i],dtv[j],self.PSD_funcs, self.dt, window=self.window,  fmin = self.fmin, fmax = self.fmax, use_gpu=self.use_gpu).real))

                #Exploiting symmetric property of the Fisher Matrix
                Fisher[j,i] = Fisher[i,j]

        # Check for degeneracies
        diag_elements = np.diag(Fisher)
        
        if 0 in diag_elements:
            logger.critical("Nasty. We have a degeneracy. Can't measure a parameter")
            degen_index = np.argwhere(diag_elements == 0)[0][0]
            Fisher[degen_index,degen_index] = 1.0
        
        # Check for positive-definiteness
        if 'm1' in self.param_names:
            index_of_M = np.where(np.array(self.param_names) == 'm1')[0][0]
            Fisher_inv = fishinv(self.wave_params['m1'], Fisher, index_of_M = index_of_M)
        else:
            Fisher_inv = np.linalg.inv(Fisher)

        if (np.linalg.eigvals(Fisher_inv) < 0.0).any():
            logger.critical("Calculated Fisher is not positive semi-definite. Try lowering inspiral error tolerance or increasing the derivative order.")
        else:
            logger.info("Calculated Fisher is *atleast* positive-definite.")
        
        if self.filename == None:
            pass
        else:
            if self.save_derivatives:
                mode = "a" #append
            else:
                mode = "w" #write new
            if self.suffix != None:                    
                with h5py.File(f"{self.filename}/Fisher_{self.suffix}.h5", mode) as f:
                    f.create_dataset("Fisher",data=Fisher)
            else:
                with h5py.File(f"{self.filename}/Fisher.h5", mode) as f:
                    f.create_dataset("Fisher",data=Fisher)
                    
        return Fisher
