import numpy as np

try:
    import cupy as cp
except:
    pass

from scipy.interpolate import make_interp_spline
from stableemrifisher.noise import noise_PSD_AE, sensitivity_LWA
from few.utils.constants import YRSID_SI

def tukey(N, alpha=0.5, use_gpu=False):
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
    if use_gpu:
        xp = cp
    else:
        xp = np

    t = xp.linspace(0., 1., N)
    window = xp.ones(N)
    condition1 = (t > (1 - alpha / 2)) & (t <= 1)
    condition2 = (t >= 0) & (t < alpha / 2)
    window[condition1] = 0.5 * (1 + xp.cos(2 * xp.pi / alpha * ((t[condition1] - 1 + alpha / 2) - 1)))
    window[condition2] = 0.5 * (1 + xp.cos(2 * xp.pi / alpha * (t[condition2] - alpha / 2)))
    return window
    

def generate_PSD(waveform, dt, noise_PSD=noise_PSD_AE, channels = ["A","E"], noise_kwargs={"TDI":"TDI1"}, use_gpu=False):
    """
    generate the power spectral density for a given waveform, noise_PSD function,
    requested number of response channels, and response generation
    
    Args:
        waveform (nd.array): the waveform which will decide some properties of the PSD. It should have dimensions (N_channels, N), where N is the length of the signal.
        dt (float): time step in seconds at which the waveform is samples.
        noise_PSD (func): function to calculate the noise of the instrument at a given frequency and noise configuration (default is noise_PSD_AE)
        channels (list): list of LISA response channels (default is ["A","E"]
        noise_kwargs (dict): additional keyword arguments to be provided to the noise function
        
    returns:
        nd.array: power spectral density of the requested noise model and of the size of the input waveform.
    """
    #generate PSD
    if use_gpu:
        xp = cp
    else:
        xp = np

    # Extract fourier frequencies
    length = len(waveform[0])
    freq = xp.fft.rfftfreq(length)/dt
    df = 1/(length * dt)

    # Compute evolution time of EMRI 
    T = (df * YRSID_SI)**-1

    if use_gpu:
        freq_np = freq.get() # Compute frequencies
    else:
        freq_np = freq

    # Generate PSDs given LWA/TDI variables
    if isinstance(noise_kwargs, list):
        PSD = [noise_PSD(freq_np[1:], **noise_kwargs_temp) for noise_kwargs_temp in noise_kwargs]
        PSD_cp = [xp.asarray(item) for item in PSD] # Convert to cupy array
        return PSD_cp[0:len(channels)]      
    elif (channels == ["A", "E"]) and (noise_kwargs == {}): #TODO: FIX. This is horrendous Ollie 
        PSD_cp = noise_PSD(freq_np[1:])
        return PSD_cp
    else:
        PSD = len(channels) * [noise_PSD(freq_np[1:], **noise_kwargs)]
        PSD_cp = [xp.asarray(item) for item in PSD] # Convert to cupy array
        return PSD_cp[0:len(channels)]      
    

        
    
    #PSD_funcs = PSD_cp[0:len(PSD_cp)] # Choose which channels to include
    


def inner_product(a, b, PSD, dt, window=None, fmin = None, fmax = None, use_gpu=False):
    """
    Compute the frequency domain inner product of two time-domain arrays.

    This function computes the frequency domain inner product of two time-domain arrays using the GPU for acceleration.
    It operates under the assumption that the signals are evenly spaced and applies a Tukey window to each signal.
    This function is optimized for GPUs.

    Args:
        a (np.ndarray): The first time-domain signal. It should have dimensions (N_channels, N), where N is the length of the signal.
        b (np.ndarray): The second time-domain signal. It should have dimensions (N_channels, N), where N is the length of the signal.
        PSD (np.ndarray): The power spectral density (PSD) of the signals. It should be a 1D array of length N_channels.
        dt (float): The sampling interval, i.e., the spacing between time samples.
        window (np.ndarray, optional): a window array to envelope the waveform time series. Default is None (no window).
        fmin (float, optional): minimum frequency for inner_product sum. Default is None.
        fmax (float, optional): maximum frequency for inner_product sum. Default is None.
        use_gpu (bool, optional): whether to use gpu. Default is False.
    Returns:
        float: The frequency-domain inner product of the two signals.

    """
    if use_gpu:
        xp = cp
    else:
        xp = np

    #print("fmin: {}, fmax: {}".format(fmin, fmax))

    #frequency cutoff mask
    if (fmin != None) or (fmax != None):
        
        length = len(a[0])
        freq = xp.fft.rfftfreq(length)/dt

        if use_gpu:
            freq = freq.get() #convert to numpy

        if fmin != None:
            mask_min = (freq > fmin)
        
        if fmax != None:
            mask_max = (freq < fmax)

        if (fmin != None) and (fmax == None):
            freq_mask = mask_min
        elif (fmin == None) and (fmax != None):
            freq_mask = mask_max
        else:
            freq_mask = xp.logical_and(mask_min, mask_max)

    else:
        length = len(a[0])
        freq = xp.fft.rfftfreq(length)/dt

        freq_mask = np.full(len(freq), True, dtype = bool)

    freq_mask = freq_mask[1:] #skip the first element corresponding to f = 0.0

    a = xp.atleast_2d(a)
    b = xp.atleast_2d(b)
    PSD = xp.atleast_2d(xp.asarray(PSD))  # handle passing the same PSD for multiple channels

    N = a.shape[1]

    df = (N * dt) ** -1

    if window is not None:
        window = xp.atleast_2d(xp.asarray(window))
        a_in = a * window
        b_in = b * window
    else:
        a_in, b_in = a, b

    if xp.iscomplexobj(a_in):
        a_fft_plus = (dt * xp.fft.rfft(a_in.real, axis=-1)[:,1:])[:,freq_mask]
        a_fft_cross = (dt * xp.fft.rfft(a_in.imag, axis=-1)[:,1:])[:,freq_mask]

        b_fft_plus = (dt * xp.fft.rfft(b_in.real, axis=-1)[:,1:])[:,freq_mask]
        b_fft_cross = (dt * xp.fft.rfft(b_in.imag, axis=-1)[:,1:])[:,freq_mask]

        inner_prod = 4 * df * ((a_fft_plus.conj() * b_fft_plus + a_fft_cross * b_fft_cross.conj()).real / PSD[:,freq_mask]).sum()
        
    else:
        a_fft = (dt * xp.fft.rfft(a_in, axis=-1)[:,1:])[:,freq_mask]
        b_fft = (dt * xp.fft.rfft(b_in, axis=-1)[:,1:])[:,freq_mask]

        # Compute inner products over given channels
        inner_prod = 4 * df * ((a_fft.conj() * b_fft).real / PSD[:,freq_mask]).sum()
    
    if use_gpu:
        inner_prod = inner_prod.get()

    return inner_prod
    
def SNRcalc(waveform, PSD, dt, window=None, fmin = None, fmax = None, use_gpu=False):
    """
    Give the SNR of a given waveform after SEF initialization.
    Returns:
        float: SNR of the source.
    """
        
    return np.sqrt(inner_product(waveform,waveform, PSD, dt , window=window, fmin = fmin, fmax = fmax, use_gpu=use_gpu))

def padding(a, b, use_gpu=False):
    """
    Make time series 'a' the same length as time series 'b'.
    Both 'a' and 'b' must be cupy array of the same shape.

    returns padded 'a'
    """
    if use_gpu:
        xp = cp
    else:
        xp = np

    a = xp.asarray(a)
    b = xp.asarray(b)

    assert a.ndim == b.ndim
    
    if a.ndim > 1:
        a_temp = []
        
        for i in range(len(a)):
            if len(a[i]) < len(b[i]):
                a_temp.append(xp.concatenate((a,xp.zeros(len(b[i])-len(a[i])))))
        
            elif len(a[i]) > len(b[i]):
                a_temp.append(a[i][:len(b[i])])
                
            else:
                a_temp.append(a[i])
        
        a = xp.array(a_temp)

    else:
        if len(a) < len(b):
            a = xp.concatenate((a,xp.zeros(len(b)-len(a))))
        elif len(a) > len(b):
            a = a[:len(b)]
            
    return a

def get_inspiral_overwrite_fun(interpolation_factor, spline_order=7):
    def func(self, *args, **kwargs):

        traj_output = self.get_inspiral_inner(*args, **kwargs)
        t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj_output
        out = np.vstack((p, e, x, Phi_phi, Phi_theta, Phi_r))
        t_new = np.interp(
            np.arange((len(t) - 1) * interpolation_factor + 1), 
            np.arange(0, interpolation_factor*len(t), interpolation_factor), 
            t
        )
        
        valid_spline_orders = [3, 5, 7]
        
        if spline_order in valid_spline_orders:
            spl = make_interp_spline(t,out,k=spline_order, axis=1)
        else:
            raise ValueError(f'spline_order should be one of {valid_spline_orders}')
            
        upsampled = spl(t_new)

        return (t_new.copy(),) + tuple(upsampled.copy())

    return func    

def fishinv(M, Fisher, index_of_M = 0):
    """ 
    Calculate the Fisher inverse by transforming the index of M to lnM to improve conditionality of the matrix first. 
    ONLY WORKS WITH INPUTS THAT HAVE PARAM M AT INDEX "index_of_M"!
    Helps with stability of Fisher inversion.
    """
    
    #Jacobian for Fisher = partial old/partial new, going from M -> lnM
    
    J = np.eye(len(Fisher))
    J[index_of_M,index_of_M] = M

    Fisher_lnM = J.T @ Fisher @ J

    Fisher_lnM_inv = np.linalg.inv(Fisher_lnM)

    #Jacobian for Covariance = partial new/partial old, going from lnM -> M

    Fisher_inv = J.T @ Fisher_lnM_inv @ J
    
    return Fisher_inv

def viewing_angle_partials(qS, phiS, qK, phiK):
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
        "del theta_src / del qS": d_theta_d_qS,      # key as requested
        "del theta_src / del qK": d_theta_d_qK,
        "del theta_src / del phi_K": d_theta_d_phiK,
    }

import numpy as np

def fplus_fcross_derivs(qS, phiS, qK, phiK,
                       with_respect_to=None):
    """
    Compute psi, (FplusI,FcrossI,FplusII,FcrossII) and their partial derivatives.

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
        Contains psi, FplusI, FcrossI, FplusII, FcrossII, and only the
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
    FplusI, FcrossI = c2p, -s2p
    FplusII, FcrossII = s2p, c2p

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
        'FplusI': FplusI, 'FcrossI': FcrossI,
        'FplusII': FplusII, 'FcrossII': FcrossII,
    }

    if with_respect_to is None:
        with_respect_to = ['qS','phiS','qK','phiK']

    # Loop only over requested derivatives
    if isinstance(with_respect_to, str):
        with_respect_to = [with_respect_to]
        # Ensure it's a list for consistency

    for x in with_respect_to:
        dpsi_x = - (v*du[x] - u*dv[x]) / denom
        # I-arm
        out[f'dFplusI/d{x}']  = -2.0 * s2p * dpsi_x
        out[f'dFcrossI/d{x}']  = -2.0 * c2p * dpsi_x
        # II-arm
        out[f'dFplusII/d{x}'] =  2.0 * c2p * dpsi_x
        out[f'dFcrossII/d{x}'] = -2.0 * s2p * dpsi_x
    return out


def check_if_plunging(traj, T, m1, m2,a,p0,e0,xI0,Phi_phi0, Phi_theta0, Phi_r0, chop_inspiral_time = 0.0):
    """
    This function checks whether the compact object plunges within the given time T. If it does, then we 
    remove a part of the observation window and redefine a new time. 

    """
    ONE_HOUR = 60*60

    t_traj, p_traj, e_traj, xI_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(m1, m2, a, 
                                                                                    p0, e0, xI0,
                                                                                    Phi_phi0=Phi_phi0, 
                                                                                    Phi_theta0=Phi_theta0, 
                                                                                    Phi_r0=Phi_r0, 
                                                                                    T=T)
    if t_traj[-1] < T*YRSID_SI:
        print("Ah, looks like things are plunging, nightmare. redefining time T")
        end_time_seconds = (t_traj[-1]/YRSID_SI)
        T = end_time_seconds - chop_inspiral_time*(ONE_HOUR)/YRSID_SI
        t_traj, p_traj, e_traj, xI_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(m1, m2, a, 
                                                                                    p0, e0, xI0,
                                                                                    Phi_phi0=Phi_phi0, 
                                                                                    Phi_theta0=Phi_theta0, 
                                                                                    Phi_r0=Phi_r0, 
                                                                                    T=T)
    return T
