"""Direct finite-difference derivative of the detector waveform."""

from .base import DerivativeBase


class DirectDerivative(DerivativeBase):
    """Finite differences of the full detector waveform.

    Every derivative is computed by re-generating the whole waveform (with the
    response, if any) at the stencil offsets, so it works with any waveform
    model. The distance derivative is analytic, since h is proportional to
    1/dist.
    """

    supports_response = True
    analytic_params = frozenset({"dist"})

    def compute(self, param_to_vary, delta, order, kind):
        if param_to_vary == "dist":
            return (-1 / self.parameters["dist"]) * self.waveform
        return self._finite_difference(param_to_vary, delta, order, kind)
