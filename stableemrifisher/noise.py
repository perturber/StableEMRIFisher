import numpy as np

def sensitivity_LWA(f):
    """
    LISA sensitivity function in the long-wavelength approximation (https://arxiv.org/pdf/1803.01944.pdf).
    
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
