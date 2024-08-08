import numpy as np
import cupy as cp
import warnings
import os
import scipy
import time 

from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import *
from few.waveform import Pn5AAKWaveform, GenerateEMRIWaveform
from few.utils.utility import get_separatrix, get_p_at_t

import matplotlib.pyplot as plt

import time
import warnings


class StableEMRIFisher:
    
    def __init__(self, M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                 Phi_phi0, Phi_theta0, Phi_r0, dt = 10, T = 1.0, param_args = None, EMRI_waveform_gen = None, window = None,
                 param_names=None, deltas=None, der_order=2, Ndelta=8, CovMat=False, 
                 Live_Dangerously = False, filename='', suffix=None, stats_for_nerds=True):
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
                CovMat (bool, optional): If True, compute the inverse Fisher matrix, i.e., the Covariance Matrix for the given parameters.
                
                Live_Dangerously (bool, optional): If True, perform calculations without basic consistency checks. Default is False.
                filename (string, optional): If not '', save the Fisher matrix, stable deltas, and covariance triangle plot in the folder with the same filename.
                suffix (string, optional): Used in case multiple Fishers are to be stored under the same filename.
                stats_for_nerds (bool, optional): print special stats for development purposes. Default is True.
        """
        
        #Basic Consistency Checks

        #initializing param_names list
        if param_names == None:
            raise ValueError("param_names cannot be empty.")

        else:
            self.param_names = param_names
         
        if deltas != None and len(deltas) != len(self.param_names):
            print('Length of deltas array should be equal to length of param_names.\n\
                   Assuming deltas = None.')
            deltas = None
            
        if EMRI_waveform_gen == None:
            raise ValueError("Please set up EMRI waveform model and pass as argument.")
         
        #initializing parameters
        self.M = M
        self.mu = mu
        self.a = a
        self.p0 = p0
        self.e0 = e0
        self.Y0 = Y0
        self.dist = dist
        self.qS = qS
        self.phiS = phiS
        self.qK = qK
        self.phiK = phiK
        self.Phi_phi0 = Phi_phi0
        self.Phi_theta0 = Phi_theta0
        self.Phi_r0 = Phi_r0
        self.dt = dt
        self.T = T

        #initialising extra parameters
        self.param_args = param_args

        # Handle retrograde orbits
        if self.a < 0:
            self.a *= -1.0
            self.Y0 = -1.0

        # Initilising FM details
        self.order = der_order
        self.Ndelta = Ndelta
        self.window = window
        self.SFN = stats_for_nerds

		# Determine what version of TDI to use or whether to use the LWA 
        try:
            if EMRI_waveform_gen.response_model.tdi == '1st generation':
                self.response = "TDI1"
            elif EMRI_waveform_gen.response_model.tdi == '2nd generation': 
                self.response = "TDI2"
        except:
            self.response = "LWA"

		
        if self.response in ["TDI1", "TDI2"]:
            self.channels = ["A", "E"]
            self.mich = False
            self.traj_module = EMRI_waveform_gen.waveform_gen.waveform_generator.inspiral_kwargs['func']
        else:
            self.channels = ["I", "II"]
            self.mich = True
            self.traj_module = EMRI_waveform_gen.waveform_generator.inspiral_kwargs['func']
			

        # Define what EMRI waveform model we are using  
        if 'Schwarz' in self.traj_module:
            self.waveform_model_choice = "SchwarzEccFlux"
        elif 'Kerr' in self.traj_module:
            self.waveform_model_choice = "KerrEccentricEquatorial"
        elif 'pn5' in self.traj_module:
            self.waveform_model_choice = "Pn5AAKWaveform" 

        
        # =============== Initialise Waveform generator ================
        self.waveform_generator = EMRI_waveform_gen


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
                      
        self.minmax = {'Phi_phi0':[0.1,2*np.pi*(0.99)],'Phi_r0':[0.1,np.pi*(0.99)],'Phi_theta0':[0.1,np.pi*(0.99)],
                              'qS':[0.1,np.pi*(0.99)],'qK':[0.1,np.pi*(0.99)],'phiS':[0.1,2*np.pi(0.99)],'phiK':[0.1,2*np.pi*(0.99)]}
                                            
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
        self.CovMat = CovMat
        self.filename = filename
        self.suffix = suffix
        self.Live_Dangerously = Live_Dangerously
        
        # Redefine final time if small body is plunging. More stable FMs.
        final_time = self.check_if_plunging()
        self.T = final_time/YRSID_SI # Years
    
    def __call__(self):

        # Compute SNR 
        rho = self.SNRcalc()

        self.SNR2 = rho**2

        print('Waveform Generated. SNR: ', rho)
        
        #making parent folder
        if self.filename != '':
            if not os.path.exists(self.filename):
                os.makedirs(self.filename)
                
        #1. If deltas not provided, calculating the stable deltas
        # print("Computing stable deltas")
        if self.Live_Dangerously == False:
            if self.deltas == None:
                start = time.time()
                self.Fisher_Stability() # Attempts to compute stable delta values. 
                end = time.time() - start
                print("Time taken to compute stable deltas is ", end, " seconds")
                    
        else:
            print("You have elected for dangerous living, I like it. ")
            fudge_factor_intrinsic = 3*(self.mu/self.M) * (self.SNR2)**-1
            delta_intrinsic = fudge_factor_intrinsic * np.array([self.M, self.mu, 1.0, 1.0, 1.0, 1.0])
            danger_delta_dict = dict(zip(self.param_names[0:7],delta_intrinsic))
            delta_dict_final_params = dict(zip(self.param_names[6:14],np.array(8*[1e-6])))
            danger_delta_dict.update(delta_dict_final_params)
            
            self.deltas = danger_delta_dict

        #2. Given the deltas, we calculate the Fisher Matrix
        start = time.time()
        Fisher = self.FisherCalc()
        end = time.time() - start
        print("Time taken to compute FM is", end," seconds")
        
        #3. If requested, calculate the covariance Matrix
        if self.CovMat:
            covariance = np.linalg.inv(Fisher)
            return Fisher, covariance
        else:
            return Fisher

    def SNRcalc(self):
        """
        Give the SNR of a given waveform after SEF initialization.

        Returns:
            float: SNR of the source.
        """
        param_vals = list(self.wave_params.values())
	
        print(param_vals)
	
        self.waveform = self.waveform_generator(*param_vals, mich=self.mich, dt=self.dt, T=self.T)
        
        # If we use LWA, extract real and imaginary components (channels 1 and 2)
        if self.response == "LWA":
            self.waveform = cp.asarray([self.waveform.real, self.waveform.imag])
                        
        # Extract fourier frequencies
        self.length = len(self.waveform[0])
        self.freq = cp.fft.rfftfreq(self.length)/self.dt
        self.df = 1/(self.length * self.dt)

        # Compute evolution time of EMRI 
        T = (self.df * YRSID_SI)**-1

        freq_np = cp.asnumpy(self.freq) # Compute frequencies

        # Generate PSDs given LWA/TDI variables
        if self.response == "TDI1" or self.response == "TDI2":
            PSD = 2*[noise_PSD_AE(freq_np[1:], TDI = self.response)]
        else:
            PSD = 2*[sensitivity_LWA(freq_np[1:])]  
        PSD_cp = [cp.asarray(item) for item in PSD] # Convert to cupy array
        
        self.PSD_funcs = PSD_cp[0:len(self.channels)] # Choose which channels to include
        
        return cp.asnumpy(np.sqrt(inner_product(self.waveform,self.waveform, self.df, self.PSD_funcs, self.window)).real)
    
    def check_if_plunging(self):
        """
        Checks if the body is plunging based on the computed trajectory.

        Returns:
            float: The adjusted final time of the trajectory.

        Notes:
            This method computes the trajectory of the body using the EMRIInspiral module 
            and checks if the final time of the trajectory is less than a threshold. If 
            the body is plunging, it adjusts the final time by subtracting 4 hours. If 
            not, it keeps the final time unchanged. The adjusted final time is returned.

        Raises:
            None
        """         
        traj = EMRIInspiral(func=self.traj_module)

        # Compute trajectory 
        if self.a < 0:  # Handle retrograde case
            a_val = self.a * -1.0
            Y0_val = -1.0
        else:
            a_val = self.a
            Y0_val = self.Y0
        
        traj_vals = list(self.traj_params.values())
        t_traj, _, _, _, _, _, _ = traj(*traj_vals, Phi_phi0=self.Phi_phi0, 
                                        Phi_theta0=self.Phi_theta0, Phi_r0=self.Phi_r0, 
                                        T = self.T) 

        if t_traj[-1] < self.T*YRSID_SI:
            warnings.warn("Body is plunging! Expect instabilities.")
            final_time = t_traj[-1] - 4*60*60 # Remove 4 hours of final inspiral
            print("Removed last 4 hours of inspiral. New evolution time: ", final_time/YRSID_SI, " years")
        else:
            print("Body is not plunging, Fisher should be stable.")
            final_time = self.T * YRSID_SI
        return final_time

    #defining Fisher_Stability function, generates self.deltas
    def Fisher_Stability(self):
        print('calculating stable deltas...')
        Ndelta = self.Ndelta
        deltas = {}

        for i in range(len(self.param_names)):
            # If we use Schwarzschild model, warn user about computing FM over a, Y0 and Phi_theta0
            if self.waveform_model_choice == "SchwarzEccFlux" and self.param_names[i] in ['a', 'Y0', 'Phi_theta0']:
                warnings.warn(f"{self.param_names[i]} unmeasurable in {self.waveform_model_choice} EMRI model.")
                
            # If we use KerrEccentric model, warn user about computing FM over Y0 and Phi_theta0
            elif self.waveform_model_choice == "KerrEccentricEquatorial" and self.param_names[i] in ['Y0', 'Phi_theta0']:
                warnings.warn(f"{self.param_names[i]} unmeasurable in {self.waveform_model_choice} EMRI model.")
                
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
                    del_k = self.derivative(i, delta_init[k])
                    relerr_flag = True
                    deltas['dist'] = 0.0
                    break
                else:
                    # print("For a choice of delta =",delta_init[k])
                    
                    if self.param_names[i] in list(self.minmax.keys()):
                        if self.wave_params[self.param_names[i]] <= self.minmax[self.param_names[i]][0]:
                            del_k = self.forward_derivative(i,delta_init[k])
                        elif self.wave_params[self.param_names[i]] > self.minmax[self.param_names[i]][1]:
                            del_k = self.backward_derivative(i,delta_init[k])
                        else:
                            del_k = self.derivative(i,delta_init[k])
                    else:
                        del_k = self.derivative(i,delta_init[k])

                #Calculating the Fisher Elements
                Gammai = inner_product(del_k,del_k, self.df, self.PSD_funcs,self.window)
                Gamma.append(Gammai)

            
            if relerr_flag == False:
                Gamma = cp.asnumpy(cp.array(Gamma))
                
                if (Gamma[1:] == 0.).any(): #handle non-contributing parameters
                    relerr = np.ones(len(Gamma)-1)
                else:    
                    relerr = np.abs(Gamma[1:] - Gamma[:-1])/Gamma[1:]
                if self.SFN:
                    print(relerr)
                
                relerr_min_i, = np.where(np.isclose(relerr, np.min(relerr),rtol=1e-1*np.min(relerr),atol=1e-1*np.min(relerr)))
                if len(relerr_min_i) > 1:
                    relerr_min_i = relerr_min_i[-1]

                if self.SFN:
                    print(relerr_min_i)
                
                if np.min(relerr) >= 0.01:
                    warnings.warn('minimum relative error is greater than 1%. Fisher may be unstable!')

                deltas[self.param_names[i]] = delta_init[relerr_min_i].item()
             
        if self.SFN:
            print('stable deltas: ', deltas)
        self.deltas = deltas
        self.save_deltas()

    def save_deltas(self):
        if self.filename != '':
            if self.suffix != None:
                with open(f"{self.filename}/stable_deltas_{self.suffix}.txt", "w", newline="") as file:
                    file.write(str(self.deltas))
            else:
                with open(f"{self.filename}/stable_deltas.txt", "w", newline="") as file:
                    file.write(str(self.deltas))

    #defining FisherCalc function, returns Fisher
    def FisherCalc(self):
        print('calculating Fisher matrix...')
        
        Fisher_size = len(self.param_names)
 
        Fisher = np.zeros((Fisher_size,Fisher_size), dtype=np.float64)
        dtv = []
        for i in range(len(self.param_names)):
            
            if self.waveform_model_choice == "SchwarzEccFlux" and self.param_names[i] in ['a', 'Y0', 'Phi_theta0']: 
                warnings.warn(f"{self.param_names[i]} unmeasurable in {self.waveform_model_choice} EMRI model.")
            elif self.waveform_model_choice == "KerrEccentricEquatorial" and self.param_names[i] in ['Y0', 'Phi_theta0']: 
                warnings.warn(f"{self.param_names[i]} unmeasurable in {self.waveform_model_choice} EMRI model.")
                    
            if self.param_names[i] in list(self.minmax.keys()):
                if self.wave_params[self.param_names[i]] <= self.minmax[self.param_names[i]][0]:
                    dtv.append(self.forward_derivative(i,self.deltas[self.param_names[i]]))
                elif self.wave_params[self.param_names[i]] > self.minmax[self.param_names[i]][1]:
                    dtv.append(self.backward_derivative(i,self.deltas[self.param_names[i]]))
                else:
                    dtv.append(self.derivative(i,self.deltas[self.param_names[i]]))
            else:
                dtv.append(self.derivative(i, self.deltas[self.param_names[i]]))

        print("Finished derivatives")

        for i in range(Fisher_size):
            for j in range(i,Fisher_size):
                Fisher[i,j] = np.float64(cp.asnumpy(inner_product(dtv[i],dtv[j],self.df, self.PSD_funcs,self.window).real))

                #Exploiting symmetric property of the Fisher Matrix
                Fisher[j,i] = Fisher[i,j]

        # Check for degeneracies
        diag_elements = np.diag(Fisher)
        
        if 0 in diag_elements:
            print("Nasty. We have a degeneracy. Can't measure a parameter")
            degen_index = np.argwhere(diag_elements == 0)[0][0]
            Fisher[degen_index,degen_index] = 1.0
        
        # Check for positive-definiteness
        if (np.linalg.eigvals(Fisher) <= 0.0).any():
            warnings.warn("Calculated Fisher is not positive-definite. Try lowering inspiral error tolerance or increasing the derivative order.")
        else:
            print("Calculated Fisher is *atleast* positive-definite.")

        
        if self.filename == '':
            pass
        else:
            if self.suffix != None:
                np.save(f'{self.filename}/Fisher_{self.suffix}.npy',Fisher)
            else:
                np.save(f'{self.filename}/Fisher.npy',Fisher)
                    
        return Fisher
