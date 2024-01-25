import numpy as np
import cupy as cp
import warnings
import os

gpu_available = True


from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux
from few.utils.utility import get_overlap, get_mismatch, get_separatrix, get_fundamental_frequencies
from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *
from few.utils.baseclasses import SchwarzschildEccentric, Pn5AAK, ParallelModuleBase
from few.waveform import AAKWaveformBase, Pn5AAKWaveform
from few.summation.aakwave import AAKSummation

import matplotlib.pyplot as plt

from few.utils.utility import get_p_at_t

import time
import warnings

    
traj = EMRIInspiral(func="pn5")

#supporting functions defined outside
#defining Sensitivity curve (for 1 year) 
def sensitivity(f):
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
    
    return (10/(3*L**2))*(P_OMS+4*(P_acc)/((2*np.pi*f)**4))*(1 + 6*f**2/(10*fstar**2))+Sc
    
#Defining the inner product (this runs on the GPU)
from cupy.fft import rfft, rfftfreq
import cupy as cp 

def inner_product(a,b,dt=10):
    """
    Frequency Domain inner product of two time-domain arrays.
    !!This works only on GPUs!!
    
    args:
        a, b (np.ndarray): time-domain signals with time step dt
        dt (float): time stepping of the signals
    
    Returns:
        The frequency-domain inner product (float)
    """
    
    #Taking the FFTs of a and b
    n = max((len(a),len(b)))
    f = rfftfreq(n)/dt
    df = 1/(n*dt)
    atilde_real = rfft(a.real*dt, n=n)[1:]
    btilde_real = rfft(b.real*dt, n=n)[1:]

    plus_prod = cp.conj(atilde_real)@(btilde_real/sensitivity(f[1:]))

    atilde_imag = rfft(a.imag*dt, n=n)[1:]
    btilde_imag = rfft(b.imag*dt, n=n)[1:]

    cross_prod = cp.conj(atilde_imag)@(btilde_imag/sensitivity(f[1:]))
    return 4*df*cp.real(plus_prod+cross_prod)

#defining the function for plotting the covariance ellipses
from matplotlib.patches import Ellipse
from matplotlib import transforms

def cov_ellipse(mean, cov, ax, n_std=1.0, edgecolor='blue',facecolor='none',lw=5, **kwargs):

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    #print(pearson)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        lw = lw,
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
    
class Fisher:
    
    def __init__(self, M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,\
                 Phi_phi0, Phi_theta0, Phi_r0, dt = 10, T = 1, param_names=[],\
                 deltas=None, der_order=2, Ndelta=8, err = 5e-12, CovMat=False, CovEllipse=False, filename='', suffix=None):
        """
        This Fisher class takes as input the source parameters of an EMRI as modeled in the 
        5PNAAK waveform model of the FastEMRIWaveforms package, and produces the corresponding
        Fisher matrix from it.
        args:
            M (float): Mass of the MBH
            mu (float): Mass of the Compact Object
            a (float): Spin of the MBH
            p0 (float): Initial semi-latus rectum of the EMRI
            Y0 (float): cosine of the initial inclination of the CO orbit with respect to the EMRI equatorial plane
            dist (float): distance from detector in Gpc
            qS, phiS (float): sky location parameters from the detector
            qK, phiK (float): source spin vector orientation with respect to the detector equatorial plane
            Phi_phi0, Phi_theta0, Phi_r0 (float): Initial phases of the CO orbit
            dt (float): time steps in the EMRI signal in seconds
            T (float): duration of the EMRI signal in years
            
            param_names (np.ndarray of size 14): order in which Fisher matrix elements will be arranged; default is None
            deltas (np.ndarray of size 14): range of stable deltas for numerical differentation of each parameter; default is None
            der_order (int): order at which to calculate the numerical derivatives. 2nd, 4th, 6th, and 8th order derivatives available.; 
                             default is 2
            Ndelta (int): Density of the delta range grid for calculation of stable deltas. Default is 8.
            err (float): Error tolerance for the inspiral trajectory interpolation. Default is 5e-12. 
                         We do not recommend changing this value.
                         
            CovMat (bool): if True, compute the inverse Fisher matrix, i.e. the Covariance Matrix for the given parameters.
            CovEllipse (bool): if True, compute the triangle plot ellipses for the Fisher approximation of the covariance matrix.
            
            filename (string): if not '', save the Fisher matrix, stable deltas, and covariance triangle plot in the folder with
                               the same filename.
            suffix (string): Used in case multiple Fishers are to be stored under the same filename.
        
        """
        
        #Basic Consistency Checks
        if err > 5e-12:
            print('Integrator error tolerance is greater than recommended (err = 5e-12). Fishers may be inaccurate.')
        
        if deltas != None and len(deltas) != 14:
            print('Length of deltas array should be equal to the no. of 5PNAAK parameters.\n\
                   Assuming deltas = None.')
            deltas = None
        
        #initializing param_names list
        if len(param_names) == 0:
            self.param_names = ['M','mu','a','p0','e0','Y0','dist','qS','phiS','qK','phiK','Phi_phi0', 'Phi_theta0','Phi_r0']
            
        elif len(param_names) != 14:
            print('Warning! Lenght of param_names must be equal to the no. of 5PNAAK parameters in FEW.\n\
                   Assuming default order of parameters.')
            
            self.param_names = ['M','mu','a','p0','e0','Y0','dist','qS','phiS','qK','phiK','Phi_phi0', 'Phi_theta0','Phi_r0']
            
            
        elif len(param_names) == 14:
            self.param_names = param_names
        
        #initializing FEW
        #defining model parameters

        use_gpu = True
        
        self.err = err
        
        inspiral_kwargs = {
                "DENSE_STEPPING": 0,
                "max_init_len": int(1e8),
                "err": self.err,
            }
            
        # keyword arguments for summation generator (AAKSummation)
        sum_kwargs = {
            "use_gpu": use_gpu,  # GPU is availabel for this type of summation
            "pad_output": False,
        }

        self.wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

        
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
        self.order = der_order
        self.Ndelta = Ndelta
        
        #initializing param dictionary
        self.param = {'M':M,\
                      'mu':mu,\
                      'a':a,\
                      'p0':p0,\
                      'e0':e0,\
                      'Y0':Y0,\
                      'dist':dist,\
                      'qS':qS,\
                      'phiS':phiS,\
                      'Phi_phi0':Phi_phi0,\
                      'Phi_theta0':Phi_theta0,\
                      'Phi_r0':Phi_r0,\
                      'qK':qK,\
                      'phiK':phiK}
        
        #initializing deltas
        self.deltas = deltas #Use deltas == None as a Flag
        
        #initializing other Flags:
        self.CovMat = CovMat
        self.CovEllipse = CovEllipse
        self.filename = filename
        self.suffix = suffix
        
    def __call__(self):
        
        #generating the waveform
        self.waveform = self.wave_generator(M=self.M, mu=self.mu, a=self.a, p0=self.p0, e0=self.e0, Y0=self.Y0, \
                                  qS = self.qS, phiS = self.phiS, qK = self.qK, phiK = self.phiK, dist = self.dist, \
                                  Phi_phi0=self.Phi_phi0, Phi_theta0=self.Phi_theta0, Phi_r0=self.Phi_r0, mich=True, dt=self.dt, T=self.T)
        rho = np.sqrt(inner_product(self.waveform,self.waveform,dt=self.dt)).real
        print('Waveform Generated. SNR: ', rho)
        
        #making parent folder
        if not os.path.exists(self.filename):
            os.makedirs(self.filename)
        
        #Performing tasks chronologically
        
        #1. If deltas not provided, calculating the stable deltas
        if self.deltas == None:
            self.Fisher_Stability() #generates self.deltas
        
        #2. Given the deltas, we calculate the Fisher Matrix
        Fisher = self.FisherCalc()
        
        #3. If demanded, calculate the covariance Matrix
        if self.CovMat:
            covariance = np.linalg.inv(Fisher)
        else: 
            covariance = []
            
        if self.CovEllipse:
            covariance = np.linalg.inv(Fisher)
            self.CovEllipsePlot(covariance)
            
        return Fisher, covariance
    
    #Now, we define the functions being used by the class
    
    #defining the derivative function
    def derivative(self, i, delta):
        
        #if calculating partial derivatives over extrinsic param set, sparse trajectories can be used.
        if self.param_names[i] == 'dist' or self.param_names[i] == 'qS' or self.param_names[i] == 'phiS' or self.param_names[i] == 'qK' or self.param_names[i] == 'phiK':
            inspiral_kwargs = {
                "DENSE_STEPPING": 0,
                "max_init_len": int(1e8),
                "err":1e-10, #default
            }

            # keyword arguments for summation generator (AAKSummation)
            sum_kwargs = {
                "use_gpu": True,
                "pad_output": False,
            }

            self.wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=True)

        else:
            inspiral_kwargs = {
                "DENSE_STEPPING": 0,
                "max_init_len": int(1e8),
                "err": self.err,
            }

            # keyword arguments for summation generator (AAKSummation)
            sum_kwargs = {
                "use_gpu": True,
                "pad_output": False,
            }

            self.wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=True)

            
        
        #modifying the given parameter
        temp = self.param.copy()

        temp[self.param_names[i]] += delta
        print(self.param_names[i],' = ', temp[self.param_names[i]])

        waveform_plus = self.wave_generator(M=temp['M'],\
                    mu = temp['mu'],\
                    a = temp['a'],\
                    p0 = temp['p0'],\
                    e0 = temp['e0'],\
                    Y0 = temp['Y0'],\
                    qS = temp['qS'],\
                    phiS = temp['phiS'],\
                    qK = temp['qK'],\
                    phiK = temp['phiK'],\
                    dist = temp['dist'],\
                    Phi_phi0= temp['Phi_phi0'],\
                    Phi_theta0=temp['Phi_theta0'],\
                    Phi_r0=temp['Phi_r0'],\
                    mich=True,\
                    T = self.T,\
                    dt = self.dt
                    )

        temp = self.param.copy()

        temp[self.param_names[i]] -= delta

        waveform_minus = self.wave_generator(M=temp['M'],\
                    mu = temp['mu'],\
                    a = temp['a'],\
                    p0 = temp['p0'],\
                    e0 = temp['e0'],\
                    Y0 = temp['Y0'],\
                    qS = temp['qS'],\
                    phiS = temp['phiS'],\
                    qK = temp['qK'],\
                    phiK = temp['phiK'],\
                    dist = temp['dist'],\
                    Phi_phi0= temp['Phi_phi0'],\
                    Phi_theta0=temp['Phi_theta0'],\
                    Phi_r0=temp['Phi_r0'],\
                    mich=True,\
                    T = self.T,\
                    dt = self.dt
                    )
        
        if self.order == 2:
            #padding waveforms with zeros in the end in case of early plunge than the original waveform
            if len(waveform_plus) < len(self.waveform):
                waveform_plus = cp.concatenate((waveform_plus,cp.zeros(len(self.waveform)-len(waveform_plus), dtype=cp.complex64)))

            elif len(waveform_plus) > len(self.waveform):
                waveform_plus = waveform_plus[:len(self.waveform)]
                
            if len(waveform_minus) < len(self.waveform):
                waveform_minus = cp.concatenate((waveform_minus,cp.zeros(len(self.waveform)-len(waveform_minus), dtype=cp.complex64)))

            elif len(waveform_minus) > len(self.waveform):
                waveform_minus = waveform_minus[:len(self.waveform)]

             
            #2nd order finite difference differentiation
            derivative = (waveform_plus - waveform_minus)/(2*delta)
            
            del waveform_plus
            del waveform_minus
            
            return derivative
        
        temp = self.param.copy()

        temp[self.param_names[i]] += 2*delta

        waveform_2plus = self.wave_generator(M=temp['M'],\
                    mu = temp['mu'],\
                    a = temp['a'],\
                    p0 = temp['p0'],\
                    e0 = temp['e0'],\
                    Y0 = temp['Y0'],\
                    qS = temp['qS'],\
                    phiS = temp['phiS'],\
                    qK = temp['qK'],\
                    phiK = temp['phiK'],\
                    dist = temp['dist'],\
                    Phi_phi0= temp['Phi_phi0'],\
                    Phi_theta0=temp['Phi_theta0'],\
                    Phi_r0=temp['Phi_r0'],\
                    mich=True,\
                    T = self.T,\
                    dt = self.dt
                    )
        
        temp = self.param.copy()

        temp[self.param_names[i]] -= 2*delta

        waveform_2minus = self.wave_generator(M=temp['M'],\
                    mu = temp['mu'],\
                    a = temp['a'],\
                    p0 = temp['p0'],\
                    e0 = temp['e0'],\
                    Y0 = temp['Y0'],\
                    qS = temp['qS'],\
                    phiS = temp['phiS'],\
                    qK = temp['qK'],\
                    phiK = temp['phiK'],\
                    dist = temp['dist'],\
                    Phi_phi0= temp['Phi_phi0'],\
                    Phi_theta0=temp['Phi_theta0'],\
                    Phi_r0=temp['Phi_r0'],\
                    mich=True,\
                    T = self.T,\
                    dt = self.dt
                    )

        if self.order == 4:
            #padding waveforms with zeros in the end in case of early plunge than the original waveform
            if len(waveform_plus) < len(self.waveform):
                waveform_plus = cp.concatenate((waveform_plus,cp.zeros(len(self.waveform)-len(waveform_plus), dtype=cp.complex64)))

            elif len(waveform_plus) > len(self.waveform):
                waveform_plus = waveform_plus[:len(self.waveform)]
                
            if len(waveform_minus) < len(self.waveform):
                waveform_minus = cp.concatenate((waveform_minus,cp.zeros(len(self.waveform)-len(waveform_minus), dtype=cp.complex64)))

            elif len(waveform_minus) > len(self.waveform):
                waveform_minus = waveform_minus[:len(self.waveform)]
                
            if len(waveform_2plus) < len(self.waveform):
                waveform_2plus = cp.concatenate((waveform_2plus,cp.zeros(len(self.waveform)-len(waveform_2plus), dtype=cp.complex64)))

            elif len(waveform_2plus) > len(self.waveform):
                waveform_2plus = waveform_2plus[:len(self.waveform)]

            if len(waveform_2minus) < len(self.waveform):
                waveform_2minus = cp.concatenate((waveform_2minus,cp.zeros(len(self.waveform)-len(waveform_2minus), dtype=cp.complex64)))

            elif len(waveform_2minus) > len(self.waveform):
                waveform_2minus = waveform_2minus[:len(self.waveform)]
                
            #4th order finite difference differentiation
            derivative = (1/12*waveform_2minus - 2/3*waveform_minus + 2/3*waveform_plus -1/12*waveform_2plus)/(delta)
                
            del waveform_plus
            del waveform_minus
            del waveform_2plus
            del waveform_2minus
            
            return derivative
        
        temp = self.param.copy()

        temp[self.param_names[i]] += 3*delta

        waveform_3plus = self.wave_generator(M=temp['M'],\
                    mu = temp['mu'],\
                    a = temp['a'],\
                    p0 = temp['p0'],\
                    e0 = temp['e0'],\
                    Y0 = temp['Y0'],\
                    qS = temp['qS'],\
                    phiS = temp['phiS'],\
                    qK = temp['qK'],\
                    phiK = temp['phiK'],\
                    dist = temp['dist'],\
                    Phi_phi0= temp['Phi_phi0'],\
                    Phi_theta0=temp['Phi_theta0'],\
                    Phi_r0=temp['Phi_r0'],\
                    mich=True,\
                    T = self.T,\
                    dt = self.dt
                    )
        
        temp = self.param.copy()

        temp[self.param_names[i]] -= 3*delta

        waveform_3minus = self.wave_generator(M=temp['M'],\
                    mu = temp['mu'],\
                    a = temp['a'],\
                    p0 = temp['p0'],\
                    e0 = temp['e0'],\
                    Y0 = temp['Y0'],\
                    qS = temp['qS'],\
                    phiS = temp['phiS'],\
                    qK = temp['qK'],\
                    phiK = temp['phiK'],\
                    dist = temp['dist'],\
                    Phi_phi0= temp['Phi_phi0'],\
                    Phi_theta0=temp['Phi_theta0'],\
                    Phi_r0=temp['Phi_r0'],\
                    mich=True,\
                    T = self.T,\
                    dt = self.dt
                    )
        
        if self.order == 6:
            #padding waveforms with zeros in the end in case of early plunge than the original waveform
            if len(waveform_plus) < len(self.waveform):
                waveform_plus = cp.concatenate((waveform_plus,cp.zeros(len(self.waveform)-len(waveform_plus), dtype=cp.complex64)))

            elif len(waveform_plus) > len(self.waveform):
                waveform_plus = waveform_plus[:len(self.waveform)]
                
            if len(waveform_minus) < len(self.waveform):
                waveform_minus = cp.concatenate((waveform_minus,cp.zeros(len(self.waveform)-len(waveform_minus), dtype=cp.complex64)))

            elif len(waveform_minus) > len(self.waveform):
                waveform_minus = waveform_minus[:len(self.waveform)]
                
            if len(waveform_2plus) < len(self.waveform):
                waveform_2plus = cp.concatenate((waveform_2plus,cp.zeros(len(self.waveform)-len(waveform_2plus), dtype=cp.complex64)))

            elif len(waveform_2plus) > len(self.waveform):
                waveform_2plus = waveform_2plus[:len(self.waveform)]

            if len(waveform_2minus) < len(self.waveform):
                waveform_2minus = cp.concatenate((waveform_2minus,cp.zeros(len(self.waveform)-len(waveform_2minus), dtype=cp.complex64)))

            elif len(waveform_2minus) > len(self.waveform):
                waveform_2minus = waveform_2minus[:len(self.waveform)]
                
            if len(waveform_3plus) < len(self.waveform):
                waveform_3plus = cp.concatenate((waveform_3plus,cp.zeros(len(self.waveform)-len(waveform_3plus), dtype=cp.complex64)))

            elif len(waveform_3plus) > len(self.waveform):
                waveform_3plus = waveform_3plus[:len(self.waveform)]
                
            if len(waveform_3minus) < len(self.waveform):
                waveform_3minus = cp.concatenate((waveform_3minus,cp.zeros(len(self.waveform)-len(waveform_3minus), dtype=cp.complex64)))

            elif len(waveform_3minus) > len(self.waveform):
                waveform_3minus = waveform_3minus[:len(self.waveform)]

                
            #4th order finite difference differentiation
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

        waveform_4plus = self.wave_generator(M=temp['M'],\
                    mu = temp['mu'],\
                    a = temp['a'],\
                    p0 = temp['p0'],\
                    e0 = temp['e0'],\
                    Y0 = temp['Y0'],\
                    qS = qS,\
                    phiS = phiS,\
                    qK = qK,\
                    phiK = phiK,\
                    dist = dist,\
                    Phi_phi0= Phi_phi0,\
                    Phi_theta0=Phi_theta0,\
                    Phi_r0=Phi_r0,\
                    mich=True,\
                    T = self.T,\
                    dt = self.dt
                    )
        
        temp = self.param.copy()

        temp[self.param_names[i]] -= 4*delta

        waveform_4minus = self.wave_generator(M=temp['M'],\
                    mu = temp['mu'],\
                    a = temp['a'],\
                    p0 = temp['p0'],\
                    e0 = temp['e0'],\
                    Y0 = temp['Y0'],\
                    qS = qS,\
                    phiS = phiS,\
                    qK = qK,\
                    phiK = phiK,\
                    dist = dist,\
                    Phi_phi0= Phi_phi0,\
                    Phi_theta0=Phi_theta0,\
                    Phi_r0=Phi_r0,\
                    mich=True,\
                    T = self.T,\
                    dt = self.dt
                    )


        #padding waveforms with zeros in the end in case of early plunge than the original waveform
        if len(waveform_plus) < len(self.waveform):
            waveform_plus = cp.concatenate((waveform_plus,cp.zeros(len(self.waveform)-len(waveform_plus), dtype=cp.complex64)))

        elif len(waveform_plus) > len(self.waveform):
            waveform_plus = waveform_plus[:len(self.waveform)]

        if len(waveform_2plus) < len(self.waveform):
            waveform_2plus = cp.concatenate((waveform_2plus,cp.zeros(len(self.waveform)-len(waveform_2plus), dtype=cp.complex64)))

        elif len(waveform_2plus) > len(self.waveform):
            waveform_2plus = waveform_2plus[:len(self.waveform)]

        if len(waveform_3plus) < len(self.waveform):
            waveform_3plus = cp.concatenate((waveform_3plus,cp.zeros(len(self.waveform)-len(waveform_3plus), dtype=cp.complex64)))

        elif len(waveform_3plus) > len(self.waveform):
            waveform_3plus = waveform_3plus[:len(self.waveform)]

        if len(waveform_4plus) < len(self.waveform):
            waveform_4plus = cp.concatenate((waveform_4plus,cp.zeros(len(self.waveform)-len(waveform_4plus), dtype=cp.complex64)))

        elif len(waveform_4plus) > len(self.waveform):
            waveform_4plus = waveform_4plus[:len(self.waveform)]

        if len(waveform_minus) < len(self.waveform):
            waveform_minus = cp.concatenate((waveform_minus,cp.zeros(len(self.waveform)-len(waveform_minus), dtype=cp.complex64)))

        elif len(waveform_minus) > len(self.waveform):
            waveform_minus = waveform_minus[:len(self.waveform)]

        if len(waveform_2minus) < len(self.waveform):
            waveform_2minus = cp.concatenate((waveform_2minus,cp.zeros(len(self.waveform)-len(waveform_2minus), dtype=cp.complex64)))

        elif len(waveform_2minus) > len(self.waveform):
            waveform_2minus = waveform_2minus[:len(self.waveform)]

        if len(waveform_3minus) < len(self.waveform):
            waveform_3minus = cp.concatenate((waveform_3minus,cp.zeros(len(self.waveform)-len(waveform_3minus), dtype=cp.complex64)))

        elif len(waveform_3minus) > len(self.waveform):
            waveform_3minus = waveform_3minus[:len(self.waveform)]

        if len(waveform_4minus) < len(self.waveform):
            waveform_4minus = cp.concatenate((waveform_4minus,cp.zeros(len(self.waveform)-len(waveform_4minus), dtype=cp.complex64)))

        elif len(waveform_4minus) > len(self.waveform):
            waveform_4minus = waveform_4minus[:len(self.waveform)]

        #8th order finite difference differentiation
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
            #custom delta_inits for different parameters
            if self.param[self.param_names[i]] == 0.0:
                delta_init = np.geomspace(1e-8,1e-16,Ndelta)

            elif self.param_names[i] == 'M' or self.param_names[i] == 'mu' or self.param_names[i] == 'a' or self.param_names[i] == 'p0' or self.param_names[i] == 'e0' or self.param_names[i] == 'Y0':
                delta_init = np.geomspace(1e-4*self.param[self.param_names[i]],1e-10*self.param[self.param_names[i]],Ndelta)

            else:
                delta_init = np.geomspace(1e-1*self.param[self.param_names[i]],1e-10*self.param[self.param_names[i]],Ndelta)

            #sanity check:
            if self.param_names[i] == 'a' and self.param[self.param_names[i]] >= 1.:
                self.param_names[i] = 0.999
            if self.param_names[i] == 'p0' and self.param[self.param_names[i]] <= 5:
                self.param_names[i] = 5.0001
            if self.param_names[i] == 'e0' and self.param[self.param_names[i]] <= 0.:
                self.param_names[i] = 1e-6
            
            Gamma = []
            orderofmag = []

            for k in range(Ndelta):
                del_k = self.derivative(i, delta_init[k])

                #Calculating the Fisher Elements
                Gammai = inner_product(del_k,del_k,dt=self.dt)
                print(Gammai)
                Gamma.append(Gammai)
                relerr_flag = False
                #if k >= 1:
                #    relerr = np.abs(Gamma[k]-Gamma[k-1])/Gamma[k]
                #    print(relerr)
                #    orderofmag.append(np.floor(np.log10(relerr)))
                #    print(orderofmag)
                #    if k >= 2:
                #        if orderofmag[-1] > orderofmag[-2]:
                #            deltas[self.param_names[i]] = np.float64(delta_init[k-1])
                #            relerr_flag = True
                #            break
            
            if relerr_flag == False:
                Gamma = cp.asnumpy(cp.array(Gamma))
                relerr = np.abs(Gamma[1:] - Gamma[:-1])/Gamma[1:]
                print(relerr)
                relerr_min_i, = np.where(np.isclose(relerr, np.min(relerr),rtol=1e-1*np.min(relerr),atol=1e-1*np.min(relerr)))
                if len(relerr_min_i) > 1:
                    relerr_min_i = relerr_min_i[-1]
                print(relerr_min_i)
                
                if np.min(relerr) >= 0.01:
                    print('minimum relative error is greater than 1%. Fisher may be unstable!')

                deltas[self.param_names[i]] = np.float64(delta_init[relerr_min_i])
            
            #plt.plot(delta_init,Gamma,'ro-')
            #plt.xscale('log')
            #plt.yscale('log')
            #plt.show()
            
        print('stable deltas: ', deltas)
        self.deltas = deltas
        if self.filename != '':
            if self.suffix != None:
                with open(f"{self.filename}/stable_deltas_{self.suffix}.txt", "w", newline="") as file:
                    file.write(str(deltas))
            else:
                with open(f"{self.filename}/stable_deltas.txt", "w", newline="") as file:
                    file.write(str(deltas))
        
    #defining FisherCalc function, returns Fisher
    def FisherCalc(self):
        print('calculating Fisher matrix...')
        Fisher = np.zeros((len(self.param_names),len(self.param_names)),dtype = np.float64)

        dtv = cp.zeros((len(self.param_names),len(self.waveform)),dtype=np.complex64)

        for i in range(len(self.param_names)):
            dtv[i] = self.derivative(i, self.deltas[self.param_names[i]])

        for i in range(len(self.param_names)):
            for j in range(i,len(self.param_names)):
                #compute the diagonal Gamma
                Fisher[i,j] = np.float64(cp.asnumpy(inner_product(dtv[i],dtv[j],dt=self.dt).real))

                #Exploiting symmetric property of the Fisher Matrix
                Fisher[j,i] = Fisher[i,j]
                
        if self.filename == '':
            pass
        else:
            if self.suffix != None:
                np.savetxt(f'{self.filename}/Fisher_{self.suffix}.txt',Fisher)
            else:
                np.savetxt(f'{self.filename}/Fisher.txt',Fisher)
                
        return Fisher

    
    #defining CovEllipsePlot function, produces Matplotlib plot
    #for the generated covariance matrix
    
    def CovEllipsePlot(self, covariance):
        print('plotting covariance ellipses...')
        
        fig, axs = plt.subplots(len(self.param_names),len(self.param_names),figsize=(20,20))

        #first param index
        for i in range(len(self.param_names)):
            #second param index
            for j in range(i+1,len(self.param_names)):

                cov = np.array(((covariance[i][i],covariance[i][j]),(covariance[j][i],covariance[j][j])))
                #print(cov)
                mean = np.array((self.param[self.param_names[i]],self.param[self.param_names[j]]))

                cov_ellipse(mean,cov,axs[j,i],lw=2)

                #custom setting the x-y lim for each plot
                axs[j,i].set_xlim([self.param[self.param_names[i]]-2.5*np.sqrt(covariance[i][i]), self.param[self.param_names[i]]+2.5*np.sqrt(covariance[i][i])])
                axs[j,i].set_ylim([self.param[self.param_names[j]]-2.5*np.sqrt(covariance[j][j]), self.param[self.param_names[j]]+2.5*np.sqrt(covariance[j][j])])

                axs[j,i].set_xlabel(self.param_names[i],labelpad=20)
                axs[j,i].set_ylabel(self.param_names[j],labelpad=20)

        for ax in fig.get_axes():
            ax.label_outer()

        for i in range(len(self.param_names)):
            for j in range(i,len(self.param_names)):
                fig.delaxes(axs[i,j])
                
        if self.filename == '':
            pass
        else:
            if self.suffix != None:
                plt.savefig(f'{self.filename}/CovEllipse_{self.filename}_{suffix}.png',dpi=300,bbox_inches='tight')
            else:
                plt.savefig(f'{self.filename}/CovEllipse.png',dpi=300,bbox_inches='tight')
