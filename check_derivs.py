# Import relevant EMRI packages
from few.waveform import (
    GenerateEMRIWaveform,
    FastSchwarzschildEccentricFlux,
    FastKerrEccentricEquatorialFlux,
)

from few.trajectory.ode import KerrEccEqFlux

from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.geodesic import get_separatrix

# from stableemrifisher.fisher.derivatives import StableEMRIDerivative
from stableemrifisher.fisher.stablederivative import StableEMRIDerivative

from EMRI_Params import (
    m1,
    m2,
    a,
    p0,
    e0,
    xI0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    T,
    dt,
)

YRSID_SI = 31558149.763545603
ONE_HOUR = 60 * 60
# Waveform params
wave_params = {
    "m1": m1,
    "m2": m2,
    "a": a,
    "p0": p0,
    "e0": e0,
    "xI0": xI0,
    "dist": dist,
    "qS": qS,
    "phiS": phiS,
    "qK": qK,
    "phiK": phiK,
    "Phi_phi0": Phi_phi0,
    "Phi_theta0": Phi_theta0,
    "Phi_r0": Phi_r0,
}

####=======================True Responsed waveform==========================
# waveform class setup
waveform_class = FastKerrEccentricEquatorialFlux
waveform_class_kwargs = dict(
    inspiral_kwargs=dict(
        err=1e-11,
    ),
    mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
)

# waveform generator setup
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(return_list=False, frame="detector")


EMRI_deriv = StableEMRIDerivative(
    waveform_class,
    waveform_generator = waveform_generator,
    waveform_generator_kwargs = waveform_generator_kwargs
)

kwargs = {"T":T, "dt":dt}
breakpoint()
compute_deriv = EMRI_deriv(parameters = wave_params, param_to_vary="m1", delta=1e-1, order=4, kind="central", **kwargs)

