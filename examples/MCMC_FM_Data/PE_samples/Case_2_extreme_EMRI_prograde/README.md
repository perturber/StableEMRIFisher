# Parameters

M = 1e7; mu = 10; a = 0.998; p0 = 2.12; e0 = 0.425; x\_I0 = 1.0
SNR\_choice = 30.0;
qS = 0.8 ; phiS = 2.2; qK = 1.6; phiK = 1.2;
Phi\_phi0 = 2.0; Phi\_theta0 = 0.0; Phi\_r0 = 3.0

delta\_t = 10.0; T = 2.0

No noise realisation. We use TDI2, with the background, 2nd generation variables in A and E. Not sampling data stream T. 

Equal arm-length configuration. 


Simulations ran for approximately 10 hours. We ran two simulations, injection of waveform with kappa = 0 and recovery 
with kappa = {1e-5, 1e-2}. We observe statistically insignificant biases across the parameters for this particular configuration. The waveforms are well sampled, 
SNR is set to 30, and at realistic luminosity distances around d_L = 3.590. Cosmology chosen is Planck18. 

Wide uniform priors looking like:

Delta\_theta\_intrinsic = [100, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4]  # M, mu, a, p0, e0 Y0
Delta\_theta\_D = dist/np.sqrt(np.sum(SNR\_Kerr\_FEW))

priors_in = {
    # Intrinsic parameters
    0: uniform\_dist(M - n*Delta_theta_intrinsic[0], M + n*Delta_theta_intrinsic[0]), # Primary Mass M
    # 1: uniform_dist(mu - n*Delta_theta_intrinsic[1], mu + n*Delta_theta_intrinsic[1]), # Secondary Mass mu
    1: uniform_dist(mu - 1000, mu + 1000), # Secondary Mass mu for very heavy IMRI
    2: uniform_dist(a - n*Delta_theta_intrinsic[2], 0.999), # Spin parameter a
    3: uniform_dist(p0 - n*Delta_theta_intrinsic[3], p0 + n*Delta_theta_intrinsic[3]), # semi-latus rectum p0
    4: uniform_dist(e0 - n*Delta_theta_intrinsic[4], e0 + n*Delta_theta_intrinsic[4]), # eccentricity e0
    5: uniform_dist(dist - n*Delta_theta_D, dist + n* Delta_theta_D), # distance D
    # Extrinsic parameters -- Angular parameters
    6: uniform_dist(0, np.pi), # Polar angle (sky position)
    7: uniform_dist(0, 2*np.pi), # Azimuthal angle (sky position)
    8: uniform_dist(0, np.pi),  # Polar angle (spin vec)
    9: uniform_dist(0, 2*np.pi), # Azimuthal angle (spin vec)
    # Initial phases
    10: uniform_dist(0, 2*np.pi), # Phi_phi0
    11: uniform_dist(0, 2*np.pi) # Phi_r0
}


Overall, decent sampling and no immediate issues. Happy to terminate for the paper after 10 hours. I have only used the last 30,000 samples to save space. 
