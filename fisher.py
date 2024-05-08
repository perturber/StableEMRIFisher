import numpy as np
import cupy as cp
import warnings
import os
import scipy
import time 
gpu_available = True

from few.trajectory.inspiral import EMRIInspiral
from few.utils.constants import *
from few.waveform import Pn5AAKWaveform, GenerateEMRIWaveform
from few.utils.utility import get_separatrix, get_p_at_t

import matplotlib.pyplot as plt

import time
import warnings

# from lisatools.sensitivity import noisepsd_AE,noisepsd_T # Power spectral densities
# from lisatools.sensitivity import noisepsd_AE2 # Power spectral densities
from fastlisaresponse import ResponseWrapper             # Response
    
YRSID_SI = 31558149.763545603

def sensitivity_LWA(f):
    """
    LISA sensitivity function in the long-wavelength approximation.
    
    args:
        f (float): LISA-band frequency of the signal
    
    Returns:
        The output sensitivity strain Sn(f)
    """
    
    #Defining supporting functions
    L = 2.5e9 #m
    fstar = 19.09e-3 #Hz
    
    P_OMS = (1.5e-11**2)*(1+(2e-3/f)**4) #Hz-1
    P_acc = (3e-15**2)*(1+(0.4e-3/f)**2)*(1+(f/8e-3)**4) #Hz-1
    
    #S_c changes depending on signal duration (Equation 14 in 1803.01944)
    #for 1 year
    alpha = 0.171
    beta = 292
    kappa = 1020
    gamma = 1680
    fk = 0.00215
    #log10_Sc = (np.log10(9)-45) -7/3*np.log10(f) -(f*alpha + beta*f*np.sin(kappa*f))*np.log10(np.e) + np.log10(1 + np.tanh(gamma*(fk-f))) #Hz-1 
    
    A=9e-45
    Sc = A*f**(-7/3)*np.exp(-f**alpha+beta*f*np.sin(kappa*f))*(1+np.tanh(gamma*(fk-f)))
    sensitivity_LWA = (10/(3*L**2))*(P_OMS+4*(P_acc)/((2*np.pi*f)**4))*(1 + 6*f**2/(10*fstar**2))+Sc
    return sensitivity_LWA

def noise_PSD_AE(f, TDI = 'TDI1'):
    """
    Inputs: Frequency f [Hz]
    Outputs: Power spectral density of noise process for TDI1 or TDI2.

    TODO: Incorporate the background!! 
    """
    # Define constants
    L = 2.5e9
    c = 299758492
    x = 2*np.pi*(L/c)*f
    
    # Test mass acceleration
    Spm = (3e-15)**2 * (1 + ((4e-4)/f)**2)*(1 + (f/(8e-3))**4) * (1/(2*np.pi*f))**4 * (2 * np.pi * f/ c)**2
    # Optical metrology subsystem noise 
    Sop = (15e-12)**2 * (1 + ((2e-3)/f)**4 )*((2*np.pi*f)/c)**2
    
    S_val = (2 * Spm *(3 + 2*np.cos(x) + np.cos(2*x)) + Sop*(2 + np.cos(x))) 
    
    if TDI == 'TDI1':
        S = 8*(np.sin(x)**2) * S_val
    elif TDI == 'TDI2':
        S = 32*np.sin(x)**2 * np.sin(2*x)**2 * S_val
    return S

#Defining the inner product (this runs on the GPU)
from cupy.fft import rfft, rfftfreq
import cupy as cp 

def tukey_gpu(N, alpha):
    """
    Generate a Tukey window function using GPU acceleration.

    Parameters:
    - N (int): The number of points in the window.
    - alpha (float): Shape parameter of the Tukey window. It determines the fraction of the window inside the tapered regions. 
      When alpha=0, the Tukey window reduces to a rectangular window, and when alpha=1, it reduces to a Hann window.

    Returns:
    - window (cupy.ndarray): The Tukey window function as a 1-dimensional CuPy array of length N.

    Note:
    The Tukey window is defined as a function of the input vector t, where t is a linearly spaced vector from 0 to 1 
    with N points. The function computes the values of the Tukey window function at each point in t using GPU-accelerated 
    operations and returns the resulting window as a CuPy array.
    """

    t = cp.linspace(0, 1, N, dtype=cp.float32)
    window = cp.ones(N, dtype=cp.float32)
    condition1 = (t > (1 - alpha / 2)) & (t <= 1)
    condition2 = (t >= 0) & (t < alpha / 2)
    window[condition1] = 0.5 * (1 + cp.cos(2 * cp.pi / alpha * ((t[condition1] - 1 + alpha / 2) - 1)))
    window[condition2] = 0.5 * (1 + cp.cos(2 * cp.pi / alpha * (t[condition2] - alpha / 2)))
    return window

def inner_product(a, b, df, PSD, window = None):
    """
    Compute the frequency domain inner product of two time-domain arrays.

    This function computes the frequency domain inner product of two time-domain arrays using the GPU for acceleration.
    It operates under the assumption that the signals are evenly spaced and applies a Tukey window to each signal.
    This function is optimized for GPUs.

    Args:
        a (np.ndarray): The first time-domain signal. It should have dimensions (N_channels, N), where N is the length of the signal.
        b (np.ndarray): The second time-domain signal. It should have dimensions (N_channels, N), where N is the length of the signal.
        df (float): The frequency resolution, i.e., the spacing between frequency bins.
        PSD (np.ndarray): The power spectral density (PSD) of the signals. It should be a 1D array of length N_channels.

    Returns:
        float: The frequency-domain inner product of the two signals.

    Note:
        This function requires a GPU for execution.

    """
    
    N = len(a[0])
    N_channels = len(PSD)
    dt = (N * df) ** -1 

    if window is not None:    # If we do not supply a window function 
        a_fft = [dt * rfft(a[k]*window)[1:] for k in range(N_channels)]
        b_conj_fft = [cp.conj(dt * rfft(b[k]*window))[1:] for k in range(N_channels)]
    else:                     # Fall into this case if we supply a window 
        a_fft = [dt * rfft(a[k])[1:] for k in range(N_channels)]
        b_conj_fft = [cp.conj(dt * rfft(b[k]))[1:] for k in range(N_channels)]

    # Compute inner products over given channels
    inn_prods = cp.asarray([4 * df * cp.real(a_fft[k] * b_conj_fft[k] / PSD[k]) for k in range(N_channels)])

    inner_product = cp.sum(inn_prods)
    
    #clearing cupy cache
    cache = cp.fft.config.get_plan_cache()
    cache.clear()
    
    return inner_product

#waveform padding function for numerical derivative consistency
def padding(a, b):
    """
    Make time series 'a' the same length as time series 'b'.
    Both 'a' and 'b' must be cupy array.

    returns padded 'a'
    """
    
    if len(a) < len(b):
        return cp.concatenate((a,cp.zeros(len(b)-len(a), dtype=cp.complex64)))

    elif len(a) > len(b):
        return a[:len(b)]

    else:
        return a

#defining the function for plotting the covariance ellipses
from matplotlib.patches import Ellipse
from matplotlib import transforms

def cov_ellipse(mean, cov, ax, n_std=1.0, edgecolor='blue', facecolor='none', lw=5, **kwargs):
    """
    Plot a covariance ellipse.

    This function plots a covariance ellipse for visualizing the parameter covariance matrix.
    
    Args:
        mean (tuple): Mean of the distribution in the form of (mean_x, mean_y).
        cov (np.ndarray): Covariance matrix of the distribution.
        ax (matplotlib.axes.Axes): Axes object on which to plot the ellipse.
        n_std (float, optional): Number of standard deviations to encompass within the ellipse. Default is 1.0.
        edgecolor (str, optional): Color of the ellipse's edge. Default is 'blue'.
        facecolor (str, optional): Fill color of the ellipse. Default is 'none'.
        lw (float, optional): Linewidth of the ellipse. Default is 5.
        **kwargs: Additional keyword arguments passed to matplotlib.patches.Ellipse.

    Returns:
        matplotlib.patches.Ellipse: The covariance ellipse plotted on the given Axes object.
    """
    
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        lw=lw,
        **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
 
def normal(mean, var, x):
    return np.exp(-(mean-x)**2/var/2)
    
    
class StableEMRIFisher:
    
    def __init__(self, M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
                 Phi_phi0, Phi_theta0, Phi_r0, dt = 10, T = 1.0, EMRI_waveform_gen = None, TDI = "LWA", window = None,
                 param_names=None, deltas=None, der_order=2, Ndelta=8, CovMat=False, CovEllipse=False, 
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
                TDI (str, optional): Type of Time Delay Interferometry (TDI) to be used. Options are 'LWA', 'TDI1', or 'TDI2'. Default is 'LWA'.
                param_names (np.ndarray, optional): Order in which Fisher matrix elements will be arranged. Default is None.
                deltas (np.ndarray, optional): Range of stable deltas for numerical differentiation of each parameter. Default is None.
                der_order (int, optional): Order at which to calculate the numerical derivatives. Default is 2.
                Ndelta (int, optional): Density of the delta range grid for calculation of stable deltas. Default is 8.
                CovMat (bool, optional): If True, compute the inverse Fisher matrix, i.e., the Covariance Matrix for the given parameters.
                CovEllipse (bool, optional): If True, compute the triangle plot ellipses for the Fisher approximation of the covariance matrix.
                
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
        
        if TDI not in ["LWA", "TDI1", "TDI2"]:
            raise ValueError("TDI must be one of 'LWA', 'TDI1', or 'TDI2'")
            
        if deltas != None and len(deltas) != len(self.param_names):
            print('Length of deltas array should be equal to length of param_names.\n\
                   Assuming deltas = None.')
            deltas = None
            
        if EMRI_waveform_gen == None:
            raise ValueError("Please set up EMRI waveform model and pass as argument.")
        
        #initializing FEW
        #defining model parameters

        use_gpu = True
         
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

        # Handle retrograde orbits
        if self.a < 0:
            self.a *= -1.0
            self.Y0 = -1.0

        # Initilising FM details
        self.order = der_order
        self.Ndelta = Ndelta
        self.use_gpu = use_gpu
        self.window = window
        self.SFN = stats_for_nerds

        
        # Initialise trajectory module 
        if TDI in ["TDI1", "TDI2"]: 
            self.traj_module = EMRI_waveform_gen.waveform_gen.waveform_generator.inspiral_kwargs['func']
        else:
            self.traj_module = EMRI_waveform_gen.waveform_generator.inspiral_kwargs['func']

        # Define what EMRI waveform model we are using  
        if 'Schwarz' in self.traj_module:
            self.waveform_model_choice = "SchwarzEccFlux"
        elif 'Kerr' in self.traj_module:
            self.waveform_model_choice = "KerrEccentricEquatorial"
        elif 'pn5' in self.traj_module:
            self.waveform_model_choice = "Pn5AAKWaveform" 

        # Redefine final time if small body is plunging. More stable FMs.
        final_time = self.check_if_plunging()
        self.T = final_time/YRSID_SI # Years
        
        # =============== Initialise Waveform generator ================
        self.waveform_generator = EMRI_waveform_gen

        # Determine what version of TDI to use or whether to use the LWA 
        if TDI in ["TDI1", "TDI2"]:  
            if EMRI_waveform_gen.response_model.tdi == '1st generation':
                self.response = "TDI1"
            elif EMRI_waveform_gen.response_model.tdi == '2nd generation': 
                self.response = "TDI2"
            self.channels = ["A", "E"] # This is hard coded, we could use "A, E, T"
            self.mich = False
        else:
            self.response = 'LWA'
            self.channels = ["I", "II"]
            self.mich = True

        #initializing param dictionary
        self.param = {'M':M,
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
        
        #initializing deltas
        self.deltas = deltas #Use deltas == None as a Flag
        
        #initializing other Flags:
        self.CovMat = CovMat
        self.CovEllipse = CovEllipse
        self.filename = filename
        self.suffix = suffix
        self.Live_Dangerously = Live_Dangerously
    
    def __call__(self):
        # Generate base waveform
        
        # if self.a < 0:  # Handle retrograde case
        #     a_val = self.a * -1.0
        #     Y0_val = -1.0
        # else:
        #     a_val = self.a
        #     Y0_val = self.Y0

        self.waveform = self.waveform_generator(self.M, self.mu, self.a, self.p0, self.e0, self.Y0, 
                                                self.dist, self.qS, self.phiS, self.qK, self.phiK, 
                                                self.Phi_phi0, self.Phi_theta0, self.Phi_r0, 
                                                mich=self.mich, dt=self.dt, T=self.T)
        
        # If we use LWA, extract real and imaginary components (channels 1 and 2)
        if self.response == "LWA":
            self.waveform = [self.waveform.real, self.waveform.imag]
                        
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

        # Compute SNR 
        rho = self.SNRcalc(self.waveform, self.df, self.PSD_funcs, self.window)

        self.SNR2 = rho**2

        print('Waveform Generated. SNR: ', rho)
        
        #making parent folder
        if self.filename != None:
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
        else: 
            covariance = []
            
        if self.CovEllipse:
            covariance = np.linalg.inv(Fisher)
            self.CovEllipsePlot(covariance)
            
        return Fisher, covariance

    def SNRcalc(self, waveform, df, PSD_funcs, window = None):
        """
        Give the SNR of a given waveform.

        Args:
            waveform (ndarray): time series waveform for which SNR is to be calculated.
            df (float): sampling rate of the signal in Hz.
            PSD_funcs: channel wise PSD of the detector in the chosen sensitivity curve approximation.

        Returns:
            float: SNR of the source.
        """
        
        return cp.asnumpy(np.sqrt(inner_product(waveform,waveform, df, PSD_funcs, window)).real)
    
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
        # if self.a < 0:  # Handle retrograde case
        #     a_val = self.a * -1.0
        #     Y0_val = -1.0
        # else:
        #     a_val = self.a
        #     Y0_val = self.Y0
        t_traj, _, _, _, _, _, _ = traj(self.M, self.mu, self.a, self.p0, self.e0, self.Y0,
                                                    Phi_phi0=self.Phi_phi0, Phi_theta0=self.Phi_theta0, Phi_r0=self.Phi_r0, 
                                                    T=self.T) 
        if t_traj[-1] < self.T*YRSID_SI:
            warnings.warn("Body is plunging! Expect instabilities.")
            final_time = t_traj[-1] - 4*60*60 # Remove 4 hours of final inspiral
            print("Removed last 4 hours of inspiral. New evolution time: ", final_time/YRSID_SI, " years")
        else:
            print("Body is not plunging, Fisher should be stable.")
            final_time = self.T * YRSID_SI
        return final_time
       
    def derivative(self, i, delta):

        if self.param_names[i] == 'dist':
            # Compute derivative analytically for the distance
            derivative = (-1/self.dist) * cp.asarray(self.waveform)
            return derivative
        else:

            #modifying the given parameter
            temp = self.param.copy()
            if temp['a'] < 0:           # Handle retrograde case.  
                temp['a'] *= -1.0
                temp['Y0'] = -1.0

            temp[self.param_names[i]] += delta
            # Print details if wanted
            if self.SFN:    
                print("For parameter",self.param_names[i])
                print(self.param_names[i],' = ', temp[self.param_names[i]])

            # f(x + h)
            waveform_plus = cp.asarray(self.waveform_generator(temp['M'],
                        temp['mu'],
                        temp['a'],
                        temp['p0'],
                        temp['e0'],
                        temp['Y0'],
                        temp['dist'],
                        temp['qS'],
                        temp['phiS'],
                        temp['qK'],
                        temp['phiK'],
                        temp['Phi_phi0'],
                        temp['Phi_theta0'],
                        temp['Phi_r0'],
                        mich=self.mich,\
                        T = self.T,\
                        dt = self.dt
                        ))

            if self.response == "LWA":
                waveform_plus = cp.asarray([waveform_plus.real, waveform_plus.imag])

            temp = self.param.copy()

            temp[self.param_names[i]] -= delta
            if self.SFN:
                print(self.param_names[i],' = ', temp[self.param_names[i]])

            # Need to change the values of spin so the code doesn't break. 
            if temp['a'] < 0:           # Handle retrograde case.  
                temp['a'] *= -1.0
                temp['Y0'] = -1.0

            # f(x - h)
            waveform_minus = cp.asarray(self.waveform_generator(temp['M'],
                        temp['mu'],
                        temp['a'],
                        temp['p0'],
                        temp['e0'],
                        temp['Y0'],
                        temp['dist'],
                        temp['qS'],
                        temp['phiS'],
                        temp['qK'],
                        temp['phiK'],
                        temp['Phi_phi0'],
                        temp['Phi_theta0'],
                        temp['Phi_r0'],
                        mich=self.mich,\
                        T = self.T,\
                        dt = self.dt
                        ))

            if self.response == "LWA":
                waveform_minus = cp.asarray([waveform_minus.real, waveform_minus.imag])

            #padding
            waveform_plus = padding(waveform_plus,self.waveform)
            waveform_minus = padding(waveform_minus,self.waveform)
            
            # Actually compute derivative
            if self.order == 2:
                     
                derivative = (waveform_plus - waveform_minus)/(2*delta)
            
                del waveform_plus
                del waveform_minus
            
                return derivative
        
            temp = self.param.copy()
            if temp['a'] < 0:           # Handle retrograde case.  
                temp['a'] *= -1.0
                temp['Y0'] = -1.0

            temp[self.param_names[i]] += 2*delta
            if self.SFN:
                print(self.param_names[i],' = ', temp[self.param_names[i]])
                
            waveform_2plus = cp.asarray(self.waveform_generator(temp['M'],
                            temp['mu'],
                            temp['a'],
                            temp['p0'],
                            temp['e0'],
                            temp['Y0'],
                            temp['dist'],
                            temp['qS'],
                            temp['phiS'],
                            temp['qK'],
                            temp['phiK'],
                            temp['Phi_phi0'],
                            temp['Phi_theta0'],
                            temp['Phi_r0'],
                            mich=self.mich,\
                            T = self.T,\
                            dt = self.dt
                            ))
 
            temp = self.param.copy()

            temp[self.param_names[i]] -= 2*delta
            if self.SFN:
                print(self.param_names[i],' = ', temp[self.param_names[i]])
                
            if temp['a'] < 0:           # Handle retrograde case.  
                temp['a'] *= -1.0
                temp['Y0'] = -1.0
            waveform_2minus = cp.asarray(self.waveform_generator(temp['M'],
                            temp['mu'],
                            temp['a'],
                            temp['p0'],
                            temp['e0'],
                            temp['Y0'],
                            temp['dist'],
                            temp['qS'],
                            temp['phiS'],
                            temp['qK'],
                            temp['phiK'],
                            temp['Phi_phi0'],
                            temp['Phi_theta0'],
                            temp['Phi_r0'],
                            mich=self.mich,\
                            T = self.T,\
                            dt = self.dt
                            ))

            #padding
            waveform_2plus = padding(waveform_2plus,self.waveform)
            waveform_2minus = padding(waveform_2minus,self.waveform)
        
            if self.order == 4:
    
                #4th order finite difference differentiation
                derivative = (1/12*waveform_2minus - 2/3*waveform_minus + 2/3*waveform_plus -1/12*waveform_2plus)/(delta)
                    
                del waveform_plus
                del waveform_minus
                del waveform_2plus
                del waveform_2minus
                
                return derivative
            
            temp = self.param.copy()
            
            if temp['a'] < 0:           # Handle retrograde case.  
                temp['a'] *= -1.0
                temp['Y0'] = -1.0
    
            temp[self.param_names[i]] += 3*delta
            if self.SFN:
                print(self.param_names[i],' = ', temp[self.param_names[i]])
                
            waveform_3plus = cp.asarray(self.waveform_generator(temp['M'],
                            temp['mu'],
                            temp['a'],
                            temp['p0'],
                            temp['e0'],
                            temp['Y0'],
                            temp['dist'],
                            temp['qS'],
                            temp['phiS'],
                            temp['qK'],
                            temp['phiK'],
                            temp['Phi_phi0'],
                            temp['Phi_theta0'],
                            temp['Phi_r0'],
                            mich=self.mich,\
                            T = self.T,\
                            dt = self.dt
                            ))
            
            temp = self.param.copy()
            
            temp[self.param_names[i]] -= 3*delta
            if self.SFN:
                print(self.param_names[i],' = ', temp[self.param_names[i]])
                
            if temp['a'] < 0:           # Handle retrograde case.  
                temp['a'] *= -1.0
                temp['Y0'] = -1.0
                
            waveform_3minus = cp.asarray(self.waveform_generator(temp['M'],
                            temp['mu'],
                            temp['a'],
                            temp['p0'],
                            temp['e0'],
                            temp['Y0'],
                            temp['dist'],
                            temp['qS'],
                            temp['phiS'],
                            temp['qK'],
                            temp['phiK'],
                            temp['Phi_phi0'],
                            temp['Phi_theta0'],
                            temp['Phi_r0'],
                            mich=self.mich,\
                            T = self.T,\
                            dt = self.dt
                            ))

            #padding
            waveform_3plus = padding(waveform_3plus,self.waveform)
            waveform_3minus = padding(waveform_3minus,self.waveform)
            
            if self.order == 6:
                    
            #     #4th order finite difference differentiation
                derivative = (-1/60*waveform_3minus+3/20*waveform_2minus - 3/4*waveform_minus + 3/4*waveform_plus - 3/20*waveform_2plus +1/60*waveform_3plus)/(delta)
                    
                del waveform_plus
                del waveform_minus
                del waveform_2plus
                del waveform_2minus
                del waveform_3plus
                del waveform_3minus
                
                return derivative
            
            temp = self.param.copy()
    
            temp[self.param_names[i]] += 4*delta
            if self.SFN:
                print(self.param_names[i],' = ', temp[self.param_names[i]])
                
            waveform_4plus = cp.asarray(self.waveform_generator(temp['M'],
                            temp['mu'],
                            temp['a'],
                            temp['p0'],
                            temp['e0'],
                            temp['Y0'],
                            temp['dist'],
                            temp['qS'],
                            temp['phiS'],
                            temp['qK'],
                            temp['phiK'],
                            temp['Phi_phi0'],
                            temp['Phi_theta0'],
                            temp['Phi_r0'],
                            mich=self.mich,\
                            T = self.T,\
                            dt = self.dt
                            ))
            
            temp = self.param.copy()
    
            temp[self.param_names[i]] -= 4*delta
            if self.SFN:
                print(self.param_names[i],' = ', temp[self.param_names[i]])
            
            if temp['a'] < 0:           # Handle retrograde case.  
                temp['a'] *= -1.0
                temp['Y0'] = -1.0
    
            waveform_4minus = cp.asarray(self.waveform_generator(temp['M'],
                            temp['mu'],
                            temp['a'],
                            temp['p0'],
                            temp['e0'],
                            temp['Y0'],
                            temp['dist'],
                            temp['qS'],
                            temp['phiS'],
                            temp['qK'],
                            temp['phiK'],
                            temp['Phi_phi0'],
                            temp['Phi_theta0'],
                            temp['Phi_r0'],
                            mich=self.mich,\
                            T = self.T,\
                            dt = self.dt
                            ))
            #padding
            waveform_4plus = padding(waveform_4plus,self.waveform)
            waveform_4minus = padding(waveform_4minus,self.waveform)
    
            # #8th order finite difference differentiation
            derivative = (waveform_4minus/280 - waveform_3minus*4/105 + waveform_2minus/5 - waveform_minus*4/5 \
                        - waveform_4plus/280 + waveform_3plus*4/105 - waveform_2plus/5 + waveform_plus*4/5)/(delta)
    
            del waveform_plus
            del waveform_minus
            del waveform_2plus
            del waveform_2minus
            del waveform_3plus
            del waveform_3minus
            del waveform_4plus
            del waveform_4minus
            
            return derivative
    


    
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
                
            #custom delta_inits for different parameters
            if self.param[self.param_names[i]] == 0.0:
                delta_init = np.geomspace(1e-4,1e-9,Ndelta)

            # Compute Ndelta number of delta values to compute derivative. Testing stability.
            elif self.param_names[i] == 'M' or self.param_names[i] == 'mu': 
                delta_init = np.geomspace(1e-4*self.param[self.param_names[i]],1e-9*self.param[self.param_names[i]],Ndelta)
            elif self.param_names[i] == 'a' or self.param_names[i] == 'p0' or self.param_names[i] == 'e0' or self.param_names[i] == 'Y0':
                delta_init = np.geomspace(1e-4*self.param[self.param_names[i]],1e-9*self.param[self.param_names[i]],Ndelta)
            else:
                delta_init = np.geomspace(1e-1*self.param[self.param_names[i]],1e-10*self.param[self.param_names[i]],Ndelta)

            #sanity check:
            #if self.param_names[i] == 'a' and self.param[self.param_names[i]] >= 1.:
            #    self.param_names[i] = 0.999
            #if self.param_names[i] == 'p0' and self.param[self.param_names[i]] <= 5:
            #    self.param_names[i] = 5.0001
            #if self.param_names[i] == 'e0' and self.param[self.param_names[i]] <= 0.:
            #    self.param_names[i] = 1e-6
            
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
                    del_k = self.derivative(i,delta_init[k])

                #Calculating the Fisher Elements
                Gammai = inner_product(del_k,del_k, self.df, self.PSD_funcs,self.window)
                Gamma.append(Gammai)

            
            if relerr_flag == False:
                Gamma = cp.asnumpy(cp.array(Gamma))
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
        if self.filename != None:
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
        
        if self.filename == None:
            pass
        else:
            if self.suffix != None:
                np.save(f'{self.filename}/Fisher_{self.suffix}.npy',Fisher)
            else:
                np.save(f'{self.filename}/Fisher.npy',Fisher)
        return Fisher

    
    #defining CovEllipsePlot function, produces Matplotlib plot
    #for the generated covariance matrix 
    def CovEllipsePlot(self, covariance):
        print('plotting covariance ellipses...')
        
        fig, axs = plt.subplots(len(self.param_names),len(self.param_names),figsize=(20,20))

        #first param index
        for i in range(len(self.param_names)):
            #second param index
            for j in range(i,len(self.param_names)):

                if i != j:
                    cov = np.array(((covariance[i][i],covariance[i][j]),(covariance[j][i],covariance[j][j])))
                    #print(cov)
                    mean = np.array((self.param[self.param_names[i]],self.param[self.param_names[j]]))

                    cov_ellipse(mean,cov,axs[j,i],lw=2,edgecolor='blue')

                    #custom setting the x-y lim for each plot
                    axs[j,i].set_xlim([self.param[self.param_names[i]]-2.5*np.sqrt(covariance[i][i]), self.param[self.param_names[i]]+2.5*np.sqrt(covariance[i][i])])
                    axs[j,i].set_ylim([self.param[self.param_names[j]]-2.5*np.sqrt(covariance[j][j]), self.param[self.param_names[j]]+2.5*np.sqrt(covariance[j][j])])

                    axs[j,i].set_xlabel(self.param_names[i],labelpad=20,fontsize=16)
                    axs[j,i].set_ylabel(self.param_names[j],labelpad=20,fontsize=16)

                else:
                    mean = self.param[self.param_names[i]]
                    var = covariance[i][i]

                    x = np.linspace(mean-3*np.sqrt(var),mean+3*np.sqrt(var))

                    axs[j,i].plot(x,normal(mean,var,x),c='blue')
                    axs[j,i].set_xlim([self.param[self.param_names[i]]-2.5*np.sqrt(covariance[i][i]), self.param[self.param_names[i]]+2.5*np.sqrt(covariance[i][i])])
                    axs[j,i].set_xlabel(self.param_names[i],labelpad=20,fontsize=16)
                    if i == j and j == 0:
                        axs[j,i].set_ylabel(self.param_names[i],labelpad=20,fontsize=16)

        for ax in fig.get_axes():
            ax.label_outer()

        for i in range(len(self.param_names)):
            for j in range(i+1,len(self.param_names)):
                fig.delaxes(axs[i,j])
                
        if self.filename == '':
            pass
        else:
            if self.suffix != None:
                plt.savefig(f'{self.filename}/CovEllipse_{self.filename}_{suffix}.png',dpi=300,bbox_inches='tight')
            else:
                plt.savefig(f'{self.filename}/CovEllipse.png',dpi=300,bbox_inches='tight')
