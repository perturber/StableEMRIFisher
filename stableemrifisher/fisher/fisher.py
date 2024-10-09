import numpy as np
import os
import time 
import sys

from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import YRSID_SI
from stableemrifisher.fisher.derivatives import derivative, handle_a_flip
from stableemrifisher.utils import inner_product, get_inspiral_overwrite_fun, SNRcalc
from stableemrifisher.noise import noise_PSD_AE, sensitivity_LWA
from stableemrifisher.plot import CovEllipsePlot, StabilityPlot
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger("stableemrifisher")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel("INFO")
logger.info("startup")

try:
    import cupy as cp
    cp.ones(5)
    GPU_AVAILABLE=True

except:
    GPU_AVAILABLE=False
    logger.warning("CuPy failed to import and initialise; disabling GPU usage.")

class StableEMRIFisher:
    
    def __init__(self, M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                 Phi_phi0, Phi_theta0, Phi_r0, dt = 10., T = 1.0, param_args = None, EMRI_waveform_gen = None, window = None, return_derivs = False,
                 param_names=None, deltas=None, der_order=2, Ndelta=8, CovEllipse=False, stability_plot=False, interpolation_factor=10, spline_order=7,
                 live_dangerously = False, filename=None, suffix=None, stats_for_nerds=False, use_gpu=False, waveform_kwargs=None):
        """
            This class computes the Fisher matrix for an Extreme Mass Ratio Inspiral (EMRI) system.

            Args:
                M (float): Mass of the Massive Black Hole (MBH).
                mu (float): Mass of the Compact Object (CO).
                a (float): Spin of the MBH.
                p0 (float): Initial semi-latus rectum of the EMRI.
                e0 (float): Initial eccentricity of the EMRI.
                Y0 (float): Initial cosine of the inclination of the CO orbit with respect to the EMRI equatorial plane.
                dist (float): Distance from the detector in gigaparsecs (Gpc).
                qS, phiS (float): Sky location parameters from the detector.
                qK, phiK (float): Source spin vector orientation with respect to the detector equatorial plane.
                Phi_phi0, Phi_theta0, Phi_r0 (float): Initial phases of the CO orbit.
                dt (float, optional): Time steps in the EMRI signal in seconds. Default is 10.
                T (float, optional): Duration of the EMRI signal in years. Default is 1.

                EMRI_waveform_gen (object, optional): EMRI waveform generator object. Default is None.
                param_names (np.ndarray, optional): Order in which Fisher matrix elements will be arranged. Default is None.
                deltas (np.ndarray, optional): Range of stable deltas for numerical differentiation of each parameter. Default is None.
                der_order (int, optional): Order at which to calculate the numerical derivatives. Default is 2.
                Ndelta (int, optional): Density of the delta range grid for calculation of stable deltas. Default is 8.
                CovEllise (bool, optional): If True, compute the inverse Fisher matrix, i.e., the Covariance Matrix for the given parameters and the covariance triangle plot.

                interpolation_factor (int, optional): factor by which to upsample the inspiral trajectory. This trades stability for expense. Default is 10.
                spline_order (int, optional): order of interpolation spline used to calculate the upsampled trajectory points. valid values are '3', '5', '7'. Default is 7.
                live_dangerously (bool, optional): If True, perform calculations without basic consistency checks. Default is False.
                filename (string, optional): If not None, save the Fisher matrix, stable deltas, and covariance triangle plot in the folder with the same filename.
                suffix (string, optional): Used in case multiple Fishers are to be stored under the same filename.
                stats_for_nerds (bool, optional): print special stats for development purposes. Default is False.
        """
        self.waveform = None

        self.use_gpu = use_gpu
        if use_gpu and not GPU_AVAILABLE:
            raise ValueError("CuPy failed to import; GPU is not available.")
        #Basic Consistency Checks

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

        #initialising extra parameters
        self.param_args = param_args

        # Initilising FM details
        self.order = der_order
        self.Ndelta = Ndelta
        self.window = window
        
        if stats_for_nerds:
            logger.setLevel("DEBUG")

        # =============== Initialise Waveform generator ================
        self.waveform_generator = EMRI_waveform_gen

	# Determine what version of TDI to use or whether to use the LWA 
        try:
            if self.waveform_generator.response_model.tdi == '1st generation':
                self.response = "TDI1"
            elif self.waveform_generator.response_model.tdi == '2nd generation': 
                self.response = "TDI2"
        except:
            self.response = "LWA"

        # filthy method for applying our post-trajectory upsampling to curb the erroneous behaviour in current FEW
        # dont feel bad if you are confused by wtf is going on here, because it is bad practice
        repl_fun = get_inspiral_overwrite_fun(interpolation_factor=interpolation_factor, spline_order=spline_order)
        
        
        if self.response in ["TDI1", "TDI2"]:
            try:
                if hasattr(self.waveform_generator.waveform_gen.waveform_generator.inspiral_generator, "get_inspiral_inner"):
                    pass
                else:
                    self.waveform_generator.waveform_gen.waveform_generator.inspiral_generator.get_inspiral_inner = self.waveform_generator.waveform_gen.waveform_generator.inspiral_generator.get_inspiral.__get__(
		            self.waveform_generator.waveform_gen.waveform_generator.inspiral_generator
		        )
		        
                    self.waveform_generator.waveform_gen.waveform_generator.inspiral_generator.get_inspiral = repl_fun.__get__(self.waveform_generator.waveform_gen.waveform_generator.inspiral_generator)
		
            except:
                if hasattr(self.waveform_generator.wave_gen.waveform_generator.inspiral_generator, "get_inspiral_inner"):
                    pass
                else:
                    self.waveform_generator.wave_gen.waveform_generator.inspiral_generator.get_inspiral_inner = self.waveform_generator.wave_gen.waveform_generator.inspiral_generator.get_inspiral.__get__(
                    self.waveform_generator.wave_gen.waveform_generator.inspiral_generator
		        )

                    self.waveform_generator.wave_gen.waveform_generator.inspiral_generator.get_inspiral = repl_fun.__get__(self.waveform_generator.wave_gen.waveform_generator.inspiral_generator)
		
		
        else:
            if hasattr(self.waveform_generator.waveform_generator.inspiral_generator, "get_inspiral_inner"):
                pass
            else:
                self.waveform_generator.waveform_generator.inspiral_generator.get_inspiral_inner = self.waveform_generator.waveform_generator.inspiral_generator.get_inspiral.__get__(self.waveform_generator.waveform_generator.inspiral_generator)
                self.waveform_generator.waveform_generator.inspiral_generator.get_inspiral = repl_fun.__get__(self.waveform_generator.waveform_generator.inspiral_generator)
	
	
        if waveform_kwargs is None:
            waveform_kwargs = {}

        self.waveform_kwargs = waveform_kwargs

        if self.response in ["TDI1", "TDI2"]:
            self.channels = ["A", "E"]
            self.traj_module = self.waveform_generator.waveform_gen.waveform_generator.inspiral_generator
            self.traj_module_func = self.waveform_generator.waveform_gen.waveform_generator.inspiral_kwargs['func']
        else:
            self.channels = ["I", "II"]
            self.waveform_kwargs["mich"] = True
            self.traj_module = self.waveform_generator.waveform_generator.inspiral_generator
            self.traj_module_func = self.waveform_generator.waveform_generator.inspiral_kwargs['func']

        # Define what EMRI waveform model we are using  
        if 'Schwarz' in self.traj_module_func:
            self.waveform_model_choice = "SchwarzEccFlux"
        elif 'Kerr' in self.traj_module_func:
            self.waveform_model_choice = "KerrEccentricEquatorial"
        elif 'pn5' in self.traj_module_func:
            self.waveform_model_choice = "Pn5AAKWaveform" 

        for i in range(len(self.param_names)):
            if self.waveform_model_choice == "SchwarzEccFlux" and self.param_names[i] in ['a', 'Y0', 'Phi_theta0']: 
                logger.warning(f"{self.param_names[i]} unmeasurable in {self.waveform_model_choice} EMRI model.")
            elif self.waveform_model_choice == "KerrEccentricEquatorial" and self.param_names[i] in ['Y0', 'Phi_theta0']: 
                logger.warning(f"{self.param_names[i]} unmeasurable in {self.waveform_model_choice} EMRI model.")
        
        #initializing param dictionary
        self.wave_params = {'M':M,
                      'mu':mu,
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

        self.wave_params_list = list(self.wave_params.values())

        self.minmax = {'Phi_phi0':[0.1,2*np.pi*(0.9)],'Phi_r0':[0.1,2*np.pi*(0.9)],'Phi_theta0':[0.1,2*np.pi*(0.9)],
                              'qS':[0.1,np.pi*(0.9)],'qK':[0.1,np.pi*(0.9)],'phiS':[0.1,2*np.pi*(0.9)],'phiK':[0.1,2*np.pi*(0.9)]}
                                            
        self.traj_params = dict(list(self.wave_params.items())[:6]) 

        #initialise extra args, add them to wave_params/traj_params
        full_EMRI_param = list(self.wave_params.keys())
        additional_args_flag = not(all(param in full_EMRI_param for param in param_names))
        if additional_args_flag == True and param_args != None:
            i = 0
            for param_label in param_names:
                if param_label not in full_EMRI_param:
                    arg_dict = {param_label:float(self.param_args[i])}
                    # Update both lists
                    self.traj_params.update(arg_dict)
                    self.wave_params.update(arg_dict)
                    i += 1
        elif (additional_args_flag == True and param_args == None) or (param_args != None and additional_args_flag == False):
            raise ValueError("Number of FM parameter labels do not match parameter labels") 
        # elif param_args != None and additional_args_flag == False
        #     raise ValueError("param_args must not be a list if there are additional_args_flag is False") 
        
        #initializing deltas
        self.deltas = deltas #Use deltas == None as a Flag
        
        #initializing other Flags:
        self.CovEllipse = CovEllipse
        self.stability_plot = stability_plot
        self.filename = filename
        self.suffix = suffix
        self.live_dangerously = live_dangerously
        self.return_derivs = return_derivs
        
        # Redefine final time if small body is plunging. More stable FMs.
        final_time = self.check_if_plunging()
        self.T = final_time/YRSID_SI # Years
    
        self.waveform_kwargs.update(dict(dt=self.dt, T=self.T))


    def __call__(self):
    
        
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
            fudge_factor_intrinsic = 3*(self.wave_params["mu"]/self.wave_params["M"]) * (self.SNR2)**-1
            delta_intrinsic = fudge_factor_intrinsic * np.array([self.wave_params["M"], self.wave_params["mu"], 1.0, 1.0, 1.0, 1.0])
            danger_delta_dict = dict(zip(self.param_names[0:7],delta_intrinsic))
            delta_dict_final_params = dict(zip(self.param_names[6:14],np.array(8*[1e-6])))
            danger_delta_dict.update(delta_dict_final_params)
            
            self.deltas = danger_delta_dict
            self.save_deltas()

        #2. Given the deltas, we calculate the Fisher Matrix
        start = time.time()
        if self.return_derivs:
            dtv, Fisher = self.FisherCalc()
        else:
            Fisher = self.FisherCalc()
        end = time.time() - start
        logger.info(f"Time taken to compute FM is {end} seconds")
        
        #3. If requested, calculate the covariance Matrix
        if self.CovEllipse:
            covariance = np.linalg.inv(Fisher)
            # TODO just get the user to pass filename paths in for the plots etc. It's easier to develop and gives the user more control
            if self.filename != None:
                if self.suffix != None:
                    CovEllipsePlot(self.param_names, self.wave_params, covariance, filename=os.path.join(self.filename, f"covariance_ellipses_{self.suffix}.png"))
                else:
                    CovEllipsePlot(self.param_names, self.wave_params, covariance, filename=os.path.join(self.filename, "covariance_ellipses.png"))                
            return Fisher, covariance
        elif self.return_derivs == False:
            return Fisher
        elif self.return_derivs == True:
            return dtv, Fisher
        
    def SNRcalc_SEF(self):
    	#generate PSD
        if self.use_gpu:
            xp = cp
        else:
            xp = np

        self.waveform = xp.asarray(self.waveform_generator(*self.wave_params_list, **self.waveform_kwargs))
        
        # If we use LWA, extract real and imaginary components (channels 1 and 2)
        if self.waveform.ndim == 1:
            self.waveform = xp.asarray([self.waveform.real, self.waveform.imag])
                        
        # Extract fourier frequencies
        self.length = len(self.waveform[0])
        self.freq = xp.fft.rfftfreq(self.length)/self.dt
        self.df = 1/(self.length * self.dt)

        # Compute evolution time of EMRI 
        T = (self.df * YRSID_SI)**-1

        freq_np = xp.asnumpy(self.freq) # Compute frequencies

        # TODO perform all of this setup in the __init__
        # Generate PSDs given LWA/TDI variables
        if self.response == "TDI1" or self.response == "TDI2":
            PSD = 2*[noise_PSD_AE(freq_np[1:], TDI = self.response)]
        else:
            PSD = 2*[sensitivity_LWA(freq_np[1:])]  
        PSD_cp = [xp.asarray(item) for item in PSD] # Convert to cupy array
        
        self.PSD_funcs = PSD_cp[0:len(self.channels)] # Choose which channels to include

        # Compute SNR
        logger.info(f"Computing SNR for parameters: {self.wave_params}") 
        
        return SNRcalc(self.waveform, self.PSD_funcs, dt=self.dt, window=self.window, use_gpu=self.use_gpu)
        
    
    def check_if_plunging(self):
        """
        Checks if the body is plunging based on the computed trajectory.

        Returns:
            float: The adjusted final time of the trajectory.

        Notes:
            This method computes the trajectory of the body using the EMRIInspiral module 
            and checks if the final time of the trajectory is less than a threshold. If 
            the body is plunging, it adjusts the final time by subtracting 6 hours. If 
            not, it keeps the final time unchanged. The adjusted final time is returned.

        Raises:
            None
        """         
        # Compute trajectory 
        
        traj_vals = list(handle_a_flip(self.traj_params).values())
        t_traj, _, _, _, _, _, _ = self.traj_module(*traj_vals, Phi_phi0=self.wave_params["Phi_phi0"], 
                                        Phi_theta0=self.wave_params["Phi_theta0"], Phi_r0=self.wave_params["Phi_r0"], 
                                        T = self.T) 

        if round(t_traj[-1]) < round(self.T*YRSID_SI):
            logger.warning("Body is plunging! Expect instabilities.")
            final_time = t_traj[-1] - 6*60*60 # Remove 12 hours of final inspiral
            logger.warning(f"Removed last 6 hours of inspiral. New evolution time: {final_time/YRSID_SI} years")
        else:
            logger.info("Body is not plunging, Fisher should be stable.")
            final_time = self.T * YRSID_SI
        return final_time

    #defining Fisher_Stability function, generates self.deltas
    def Fisher_Stability(self):
        if not self.use_gpu:
            xp = np
        else:
            xp = cp
        logger.info('calculating stable deltas...')
        Ndelta = self.Ndelta
        deltas = {}

        for i in range(len(self.param_names)):

            # If a specific parameter equals zero, then consider stepsizes around zero.
            if self.wave_params[self.param_names[i]] == 0.0:
                delta_init = np.geomspace(1e-4,1e-9,Ndelta)

            # Compute Ndelta number of delta values to compute derivative. Testing stability.
            elif self.param_names[i] == 'M' or self.param_names[i] == 'mu': 
                delta_init = np.geomspace(1e-4*self.wave_params[self.param_names[i]],1e-9*self.wave_params[self.param_names[i]],Ndelta)
            elif self.param_names[i] == 'a' or self.param_names[i] == 'p0' or self.param_names[i] == 'e0' or self.param_names[i] == 'Y0':
                delta_init = np.geomspace(1e-4*self.wave_params[self.param_names[i]],1e-9*self.wave_params[self.param_names[i]],Ndelta)
            else:
                delta_init = np.geomspace(1e-1*self.wave_params[self.param_names[i]],1e-10*self.wave_params[self.param_names[i]],Ndelta)
 
            Gamma = []
            orderofmag = []

            relerr_flag = False
            for k in range(Ndelta):
                if self.param_names[i] == 'dist':
                    del_k = derivative(self.waveform_generator, self.wave_params, self.param_names[i], delta_init[k], use_gpu=self.use_gpu, waveform=self.waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
                    
                    relerr_flag = True
                    deltas['dist'] = 0.0
                    break
                else:
                    # print("For a choice of delta =",delta_init[k])
                    
                    if self.param_names[i] in list(self.minmax.keys()):
                        if self.wave_params[self.param_names[i]] <= self.minmax[self.param_names[i]][0]:
                            del_k = derivative(self.waveform_generator, self.wave_params, self.param_names[i], delta_init[k], kind="forward", use_gpu=self.use_gpu, waveform=self.waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
                        elif self.wave_params[self.param_names[i]] > self.minmax[self.param_names[i]][1]:
                            del_k = derivative(self.waveform_generator, self.wave_params, self.param_names[i], delta_init[k], kind="backward", use_gpu=self.use_gpu, waveform=self.waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
                        else:
                            del_k = derivative(self.waveform_generator, self.wave_params, self.param_names[i], delta_init[k], use_gpu=self.use_gpu, waveform=self.waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
                    else:
                        del_k = derivative(self.waveform_generator, self.wave_params, self.param_names[i], delta_init[k], use_gpu=self.use_gpu, waveform=self.waveform, order=self.order, waveform_kwargs=self.waveform_kwargs)
                

                    # fig,ax = plt.subplots(1,1,figsize = (12,12))
                    # ax.plot(xp.asnumpy(del_k[0]), label = delta_init[k])
                    # ax.legend()
                    # plt.savefig("deriv_plot.pdf",bbox_inches="tight")
                    # breakpoint()
                #Calculating the Fisher Elements
                Gammai = inner_product(del_k,del_k, self.PSD_funcs, self.dt, window=self.window, use_gpu=self.use_gpu)
                # print("gammai=",Gammai)
                Gamma.append(Gammai)

            
            if relerr_flag == False:
                Gamma = xp.asnumpy(xp.array(Gamma))
                
                if (Gamma[1:] == 0.).any(): #handle non-contributing parameters
                    relerr = np.ones(len(Gamma)-1)
                else:    
                    relerr = np.abs(Gamma[1:] - Gamma[:-1])/Gamma[1:]

                logger.debug(relerr)
                
                relerr_min_i, = np.where(np.isclose(relerr, np.min(relerr),rtol=1e-1*np.min(relerr),atol=1e-1*np.min(relerr)))
                if len(relerr_min_i) > 1:
                    relerr_min_i = relerr_min_i[-1]

                logger.debug(relerr_min_i)
                
                if np.min(relerr) >= 0.01:
                    logger.warning('minimum relative error is greater than 1%. Fisher may be unstable!')

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
        # TODO fix the filename handling...
        if self.filename is not None:
            if self.suffix != None:
                with open(f"{self.filename}/stable_deltas_{self.suffix}.txt", "w", newline="") as file:
                    file.write(str(self.deltas))
            else:
                with open(f"{self.filename}/stable_deltas.txt", "w", newline="") as file:
                    file.write(str(self.deltas))

    #defining FisherCalc function, returns Fisher
    def FisherCalc(self):
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
                    dtv.append(derivative(self.waveform_generator, self.wave_params, self.param_names[i], self.deltas[self.param_names[i]], kind="forward", waveform=self.waveform, order=self.order, use_gpu=self.use_gpu, waveform_kwargs=self.waveform_kwargs))
                elif self.wave_params[self.param_names[i]] > self.minmax[self.param_names[i]][1]:
                    dtv.append(derivative(self.waveform_generator, self.wave_params, self.param_names[i],self.deltas[self.param_names[i]], kind="backward", waveform=self.waveform, order=self.order, use_gpu=self.use_gpu, waveform_kwargs=self.waveform_kwargs))
                else:
                    dtv.append(derivative(self.waveform_generator, self.wave_params, self.param_names[i],self.deltas[self.param_names[i]],use_gpu=self.use_gpu, waveform=self.waveform, order=self.order, waveform_kwargs=self.waveform_kwargs))
            else:
                dtv.append(derivative(self.waveform_generator, self.wave_params, self.param_names[i], self.deltas[self.param_names[i]],use_gpu=self.use_gpu, waveform=self.waveform, order=self.order, waveform_kwargs=self.waveform_kwargs))

        logger.info("Finished derivatives")

        for i in range(self.npar):
            for j in range(i,self.npar):
                Fisher[i,j] = np.float64(xp.asnumpy(inner_product(dtv[i],dtv[j],self.PSD_funcs, self.dt, window=self.window, use_gpu=self.use_gpu).real))

                #Exploiting symmetric property of the Fisher Matrix
                Fisher[j,i] = Fisher[i,j]

        # Check for degeneracies
        diag_elements = np.diag(Fisher)
        
        if 0 in diag_elements:
            logger.critical("Nasty. We have a degeneracy. Can't measure a parameter")
            degen_index = np.argwhere(diag_elements == 0)[0][0]
            Fisher[degen_index,degen_index] = 1.0
        
        # Check for positive-definiteness
        if (np.linalg.eigvals(Fisher) <= 0.0).any():
            logger.critical("Calculated Fisher is not positive-definite. Try lowering inspiral error tolerance or increasing the derivative order.")
        else:
            logger.info("Calculated Fisher is *atleast* positive-definite.")

        
        if self.filename == None:
            pass
        else:
            if self.suffix != None:
                np.save(f'{self.filename}/Fisher_{self.suffix}.npy',Fisher)
            else:
                np.save(f'{self.filename}/Fisher.npy',Fisher)
        
        if self.return_derivs == False:   
            return Fisher
        else:
            return dtv, Fisher
