"""Waveform models and source parameters the tests run on.

Every test that needs a FEW waveform runs once per entry in `WAVEFORM_CASES`.
To test a new model, add an entry: a model that the installed FEW does not
provide is skipped, with the FEW version it needs. The parameters a test
differentiates come from `stableemrifisher.fisher.defaults` for the model's
family, so e.g. an inclined model is tested on xI0 without further changes.
"""

from dataclasses import dataclass, field

import numpy as np

# short and cheap, but with SNR > 20 so the Fisher matrix is meaningful
T_OBS = 0.01  # years
DT = 10.0  # seconds

# a well-separated, non-plunging Kerr eccentric source, away from all stencil bounds
KERR_ECC_EQ_PARAMS = dict(
    m1=1e6,
    m2=10.0,
    a=0.9,
    p0=10.0,
    e0=0.3,
    xI0=1.0,
    dist=0.1,
    qS=np.pi / 3,
    phiS=np.pi / 4,
    qK=np.pi / 6,
    phiK=np.pi / 8,
    Phi_phi0=1.0,
    Phi_theta0=0.0,
    Phi_r0=2.0,
)

# Finite-difference step, relative to the parameter value, at which the stable
# and direct derivatives are compared. Within each range the two derivatives
# agree to better than 1e-12 in the noise-weighted norm (FEW 2.0.0, Kerr).
RELATIVE_STEPS = dict(
    m1=1e-6,
    m2=1e-5,
    a=1e-4,
    p0=1e-6,
    e0=1e-5,
    xI0=1e-5,
    qS=1e-4,
    phiS=1e-4,
    qK=1e-4,
    phiK=1e-4,
    Phi_phi0=1e-3,
    Phi_theta0=1e-3,
    Phi_r0=1e-3,
    dist=1e-4,
)

# Parameters the derivative and Fisher tests differentiate: intrinsic ones,
# whose derivatives need finite-difference steps. Every derivative takes
# several waveforms, so the tests do not run over all parameters.
TEST_PARAMS = ["m1", "p0", "e0"]


@dataclass(frozen=True)
class WaveformCase:
    """A FEW waveform model with the source the tests evaluate it at.

    Attributes:
        waveform_class_name: name of the class in `few.waveform`.
        min_few_version: first FEW release that provides the class.
        wave_params: source parameters, in generator order.
        waveform_class_kwargs: kwargs for the waveform class.
    """

    waveform_class_name: str
    min_few_version: str
    wave_params: dict
    waveform_class_kwargs: dict = field(
        default_factory=lambda: dict(inspiral_kwargs=dict(err=1e-11))
    )

    @property
    def id(self):
        return self.waveform_class_name

    def deltas(self, param_names):
        """Step size for each parameter, from `RELATIVE_STEPS`."""
        return {
            p: RELATIVE_STEPS[p] * (self.wave_params[p] or 1.0) for p in param_names
        }


WAVEFORM_CASES = [
    WaveformCase(
        waveform_class_name="FastKerrEccentricEquatorialFlux",
        min_few_version="2.0.0",
        wave_params=KERR_ECC_EQ_PARAMS,
    ),
]
