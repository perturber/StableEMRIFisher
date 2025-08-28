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

# m1 = 1e6; m2 = 10; a = 0.998; p0 = 7.7275; e0 = 0.73; xI0 = 1.0
# SNR_choice = 50.0;
# dist = 2.20360838037185 # Animal -- hardcoded distance for case 1
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
# dt = 5.0; T = 2.0

# ==================== PLUNGING EMRI =====================
m1 = 1e6; m2 = 100; a = 0.99; p0 = 14.7275; e0 = 0.4; xI0 = 1.0
SNR_choice = 30.0
dist = 14.399336686341377 # Animal -- hardcoded distance for case 1
qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
dt = 5.0; T = 0.01
# dt = 5.0; T = 1.796 #really good
# dt = 5.0; T = 1.7963

# ================== CASE 2 IN TABLE ======================
# Extreme EMRI, mass-ratio 1e-6 
# m1 = 1e7; m2 = 1e1; a = 0.998; p0 = 2.12; e0 = 0.425; xI0 = 1.0
# SNR_choice = 30;
# dist = 3.590;
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
# dt = 10.0; T = 2.0;

# ================== CASE 3 IN TABLE ======================
# MEGA IMRI -- Pints. TDI2, SNR = 

# m1 = 1e7; m2 = 100_000; a = 0.95; p0 = 23.4250; e0 = 0.85; xI0 = 1.0
# SNR_choice = 500.0; 
# dist = 3.797946825087277;
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
# dt = 5.0; T = 2.0;



# ================== CASE 4 IN TABLE ======================
# MEGA IMRI -- Pints. TDI2, SNR = 
# m1 = 1e5; m2 = 1e3; a = 0.95; p0 = 74.38304; e0 = 0.85; xI0 = 1.0
# SNR_choice = 200; 
# dist = 3.500;
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

# dt = 2.0; T = 2.0;


# ================== CASE 5 IN TABLE ======================
# EMRI source, eps ~ 1e-4, SNR 30, retrograde

# m1 = 1e5; m2 = 10; a = 0.5; p0 = 26.19; e0 = 0.8; xI0 = -1.0;
# SNR_choice = 30.0;
# dist = 1.081;
# qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0
# dt = 2.0; T = 2.0;
# Now check out eccentricity

# ================== CASE 6 ======================
# m1 = 1e6; m2 = 25; a = 0.9; p0 = 10.78; e0 = 0.3; xI0 = 1.0
# dist = 13.152600545674767; SNR_choice = 20.0; 
# qS = 0.5 ; phiS = 1.2; qK = 0.8; phiK = 0.2; 
# Phi_phi0 = 2.0; Phi_theta0 = 0.0; Phi_r0 = 3.0

# dt = 5.0; T = 2.0;