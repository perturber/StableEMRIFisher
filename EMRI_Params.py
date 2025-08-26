###
# This is the main file that should be edited. 
###
try:
    import cupy as cp
    import numpy as np
    xp = cp
    use_gpu = True
except ImportError:
    import numpy as np
    xp = np
    use_gpu = False

# ================== CASE 1 IN TABLE ======================
# EMRI source, eps ~ 1e-5, SNR 50, prograde

# M = 1e6; mu = 10; a = 0.998; p0 = 7.7275; e0 = 0.73; x_I0 = 1.0
# SNR_choice = 50.0;
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
# delta_t = 5.0; T = 2.0

# ==================== PLUNGING EMRI =====================
M = 1e6; mu = 100; a = 0.99; p0 = 14.7275; e0 = 0.4; x_I0 = 1.0
SNR_choice = 30.0
qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
delta_t = 5.0; T = 2.0

# ================== CASE 2 IN TABLE ======================
# Extreme EMRI, mass-ratio 1e-6 
# M = 1e7; mu = 1e1; a = 0.998; p0 = 2.12; e0 = 0.425; x_I0 = 1.0
# SNR_choice = 30;
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
# delta_t = 10.0; T = 2.0;

# ================== CASE 3 IN TABLE ======================
# MEGA IMRI -- Pints. TDI2, SNR = 

# M = 1e7; mu = 100_000; a = 0.95; p0 = 23.4250; e0 = 0.85; x_I0 = 1.0
# SNR_choice = 500.0; 
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
# delta_t = 5.0; T = 2.0;



# ================== CASE 4 IN TABLE ======================
# MEGA IMRI -- Pints. TDI2, SNR = 
# M = 1e5; mu = 1e3; a = 0.95; p0 = 74.38304; e0 = 0.85; x_I0 = 1.0
# SNR_choice = 200; 
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

# delta_t = 2.0; T = 2.0;


# ================== CASE 5 IN TABLE ======================
# EMRI source, eps ~ 1e-4, SNR 30, retrograde

# M = 1e5; mu = 10; a = 0.5; p0 = 26.19; e0 = 0.8; x_I0 = -1.0;
# SNR_choice = 30.0;
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
# delta_t = 2.0; T = 2.0;
# Now check out eccentricity

# M = 1e6; mu = 25; a = 0.998; p0 = 10.628; e0 = 0.1; x_I0 = 1.0
# dist = 1.0; SNR_choice = 50.0; 
# qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
# Phi_phi0 = 1.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

# delta_t = 5.0; T = 2.0;