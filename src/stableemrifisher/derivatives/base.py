"""Common interface for the waveform derivatives used by StableEMRIFisher."""

import logging
from abc import ABC, abstractmethod

import numpy as np
from few.waveform import GenerateEMRIWaveform

from stableemrifisher.utils import padding
from .finite_difference import check_stencil, offsets, apply_stencil

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger("stableemrifisher")


class DerivativeBase(ABC):
    """Base class for derivatives of the detector waveform.

    A derivative object builds and owns the waveform model it differentiates:
    a FEW waveform generator, wrapped in a LISA response if one is given.
    StableEMRIFisher uses it both to generate the waveform and to compute
    its derivatives, so every derivative class must:

    - return derivatives of the detector waveform, i.e. d(R[h])/d(param) with
      the response included, in the same (channels, samples) layout as the
      waveform from `generate`;
    - implement `compute`, and list in `analytic_params` the parameters it
      differentiates without a finite-difference step.

    Typical use:
        deriv = DirectDerivative(waveform_class=..., waveform_generator_kwargs=...)
        deriv.set_reference(parameters, **waveform_kwargs)
        dh_dm1 = deriv("m1", delta=1e-2, order=4, kind="central")

    Attributes:
        supports_response (bool): whether the class can be used with a
            ResponseWrapper.
        analytic_params (frozenset): parameters whose derivative does not
            depend on the step size.
        base_generator: the FEW waveform generator, without the response.
        waveform_generator: the generator of the detector waveform (the
            response-wrapped generator if a response is used).
        has_response (bool): whether a response wrapper is used.
        parameters (dict): reference parameters set by `set_reference`.
        waveform: reference waveform, shape (channels, samples).
    """

    supports_response = False
    analytic_params = frozenset()

    def __init__(
        self,
        waveform_class,
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=None,
        ResponseWrapper=None,
        ResponseWrapper_kwargs=None,
        use_gpu=False,
    ):
        """
        Args:
            waveform_class: uninitialized FEW waveform class.
            waveform_generator: uninitialized FEW waveform generator class,
                defaults to `few.waveform.GenerateEMRIWaveform`.
            waveform_generator_kwargs (dict): kwargs for the waveform generator
                (including those of the waveform class).
            ResponseWrapper: uninitialized response wrapper class, e.g.
                `fastlisaresponse.ResponseWrapper`, or None for no response.
            ResponseWrapper_kwargs (dict): kwargs for the response wrapper.
            use_gpu (bool): use CuPy arrays.
        """
        if ResponseWrapper is not None and not self.supports_response:
            raise ValueError(
                f"{type(self).__name__} does not support a response wrapper."
            )
        if use_gpu and cp is None:
            raise ImportError("use_gpu=True requires CuPy.")

        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np

        if waveform_generator_kwargs is None:
            waveform_generator_kwargs = {}
        if ResponseWrapper_kwargs is None:
            ResponseWrapper_kwargs = {}

        self.base_generator = waveform_generator(
            waveform_class=waveform_class, **waveform_generator_kwargs
        )
        self.has_response = ResponseWrapper is not None
        if self.has_response:
            self.waveform_generator = ResponseWrapper(
                self.base_generator, **ResponseWrapper_kwargs
            )
        else:
            self.waveform_generator = self.base_generator

        self.parameters = None
        self.waveform = None
        self.waveform_kwargs = {}

    @classmethod
    def supports_waveform(cls, waveform_class):
        """Whether this derivative can differentiate `waveform_class`."""
        return True

    def needs_step_size(self, param):
        """False if the derivative with respect to `param` is analytic."""
        return param not in self.analytic_params

    @property
    def npad(self):
        """Number of zeros padded at the end of the last derivative."""
        return 0

    def generate(self, *parameter_values, **waveform_kwargs):
        """Detector waveform for the given parameters, shape (channels, samples)."""
        return self.as_channels(
            self.waveform_generator(*parameter_values, **waveform_kwargs)
        )

    def as_channels(self, h):
        """Bring a waveform (or derivative) to the (channels, samples) layout.

        Without a response, FEW returns h+ - i hx as a single complex array,
        which is split into the two real channels [h+, hx].
        """
        h = self.xp.asarray(h)
        if h.ndim == 1:
            h = self.xp.asarray([h.real, -h.imag])
        return h

    def set_reference(self, parameters, waveform=None, **waveform_kwargs):
        """Set the point at which derivatives are computed.

        Args:
            parameters (dict): waveform parameters, in the order the waveform
                generator expects them.
            waveform: the waveform at `parameters`, if already computed.
                Generated otherwise.
            **waveform_kwargs: kwargs for the waveform generator (T, dt, ...).
        """
        self.parameters = dict(parameters)
        self.waveform_kwargs = dict(waveform_kwargs)
        if waveform is None:
            waveform = self.generate(*self.parameters.values(), **self.waveform_kwargs)
        self.waveform = self.as_channels(waveform)

    def __call__(self, param_to_vary, delta, order=4, kind="central"):
        """Derivative of the detector waveform with respect to `param_to_vary`.

        Args:
            param_to_vary (str): name of the parameter.
            delta (float): finite-difference step size (ignored for parameters
                in `analytic_params`).
            order (int): order of the finite-difference stencil.
            kind (str): "central", "forward" or "backward".

        Returns:
            The derivative, shape (channels, samples).
        """
        if self.parameters is None:
            raise RuntimeError("Call set_reference() before computing derivatives.")
        if param_to_vary not in self.parameters:
            raise ValueError(f"Parameter {param_to_vary!r} not in the parameters.")
        check_stencil(order, kind)
        return self.as_channels(self.compute(param_to_vary, delta, order, kind))

    @abstractmethod
    def compute(self, param_to_vary, delta, order, kind):
        """Derivative at the reference point. Called by `__call__`."""

    def _finite_difference(self, param_to_vary, delta, order, kind):
        """Finite-difference derivative of the full detector waveform.

        Every perturbed waveform is padded or truncated to the length of the
        reference waveform before the stencil is applied.
        """
        steps = []
        for offset in offsets(delta, order, kind):
            if offset == 0.0:
                steps.append(self.waveform)
                continue
            parameters = dict(self.parameters)
            parameters[param_to_vary] += float(offset)
            logger.debug("%s = %s", param_to_vary, parameters[param_to_vary])
            h = self.generate(*parameters.values(), **self.waveform_kwargs)
            steps.append(padding(h, self.waveform, use_gpu=self.use_gpu))
        return apply_stencil(self.xp.asarray(steps), delta, order, kind, xp=self.xp)
