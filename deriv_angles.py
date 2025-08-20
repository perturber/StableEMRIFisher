import numpy as np
from antenna_derivs import fplusI_derivs, fcrossI_derivs, fplusII_derivs, fcrossII_derivs

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

def fplus_fcross(qS, phiS, qK, phiK):
    """
    Compute psi, (FplusI,FcrossI,FplusII,FcrossII).

    Parameters
    ----------
    qS, phiS, qK, phiK : float
        Detector-frame angles.
    hp, hc : float, optional
        Polarizations before rotation. If given, rotated strains are returned.

    Returns
    -------
    np.ndarray
        contains FplusI, FcrossI, FplusII, FcrossII
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

    return FplusI, FcrossI, FplusII, FcrossII

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
        Contains only the requested derivatives.
    """

    out = {}

    if phiS == phiK:
        
        #psi = 0.5*np.pi, antenna patterns are constant, all derivatives are zero.
        if with_respect_to is None:
            with_respect_to = ['qS','phiS','qK','phiK']

        # Loop only over requested derivatives
        if isinstance(with_respect_to, str):
            with_respect_to = [with_respect_to]
            # Ensure it's a list for consistency

        for x in with_respect_to:
            # I-arm
            out[f'dFplusI/d{x}']  = 0.0
            out[f'dFcrossI/d{x}']  = 0.0
            # II-arm
            out[f'dFplusII/d{x}'] =  0.0
            out[f'dFcrossII/d{x}'] = 0.0
        return out 
    
    if with_respect_to is None:
        with_respect_to = ['qS','phiS','qK','phiK']

    # Loop only over requested derivatives
    if isinstance(with_respect_to, str):
        with_respect_to = [with_respect_to]
        # Ensure it's a list for consistency

    for x in with_respect_to:
        # I-arm
        out[f'dFplusI/d{x}']  = fplusI_derivs(qS, phiS, qK, phiK, with_respect_to=x)
        out[f'dFcrossI/d{x}']  = fcrossI_derivs(qS, phiS, qK, phiK, with_respect_to=x)
        # II-arm
        out[f'dFplusII/d{x}'] =  fplusII_derivs(qS, phiS, qK, phiK, with_respect_to=x)
        out[f'dFcrossII/d{x}'] = fcrossII_derivs(qS, phiS, qK, phiK, with_respect_to=x)
    
    return out


