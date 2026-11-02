"""Direct finite-difference derivative of the detector waveform."""

import contextlib
import logging

from few import get_logger as get_few_logger
from few.waveform import GenerateEMRIWaveform

from .base import DerivativeBase

logger = logging.getLogger("stableemrifisher")


class _LargeModeListFilter(logging.Filter):
    """Drops FEW's advice to give a long mode list at initialization."""

    def filter(self, record):
        return "Mode selection is large" not in record.getMessage()


@contextlib.contextmanager
def _without_large_mode_list_warning():
    """Silence FEW's warning for call-time mode lists of more than 50 modes.

    The fixed modes are known only once the reference waveform is generated,
    long after FEW's mode selector is built, so they can only be given at call
    time, and the warning cannot help the user. The extra checks FEW does on a
    call-time list are negligible next to generating the waveform.
    """
    few_logger = get_few_logger()
    log_filter = _LargeModeListFilter()
    few_logger.addFilter(log_filter)
    try:
        yield
    finally:
        few_logger.removeFilter(log_filter)


class DirectDerivative(DerivativeBase):
    """Finite differences of the full detector waveform.

    Every derivative is computed by re-generating the whole waveform (with the
    response, if any) at the stencil offsets, so it works with any waveform
    model. The distance derivative is analytic, since h is proportional to
    1/dist.

    With `fix_modes` (the default), the modes FEW selects for the reference
    waveform are passed to FEW (as `mode_selection`) for every perturbed
    waveform. FEW otherwise selects the modes again for each one, and a mode
    gained or lost between stencil points spoils the finite difference. The
    reference waveform itself is unchanged.
    """

    supports_response = True
    analytic_params = frozenset({"dist"})

    def __init__(
        self,
        waveform_class,
        waveform_generator=GenerateEMRIWaveform,
        waveform_generator_kwargs=None,
        ResponseWrapper=None,
        ResponseWrapper_kwargs=None,
        use_gpu=False,
        fix_modes=True,
    ):
        """
        Args:
            waveform_class, waveform_generator, waveform_generator_kwargs,
                ResponseWrapper, ResponseWrapper_kwargs, use_gpu: as in
                `DerivativeBase`.
            fix_modes (bool): use the modes of the reference waveform for all
                perturbed waveforms. Default True. Has no effect for models
                without mode selection (e.g. the AAK) or with a predictive
                mode selector (FEW chooses the modes itself), nor when
                `mode_selection` is given in the waveform kwargs.
        """
        super().__init__(
            waveform_class,
            waveform_generator=waveform_generator,
            waveform_generator_kwargs=waveform_generator_kwargs,
            ResponseWrapper=ResponseWrapper,
            ResponseWrapper_kwargs=ResponseWrapper_kwargs,
            use_gpu=use_gpu,
        )
        self.fix_modes = fix_modes
        self.modes = None
        self._last_generated = None

    def generate(self, *parameter_values, **waveform_kwargs):
        if self.modes is not None and waveform_kwargs.get("mode_selection") is self.modes:
            # the fixed modes of the reference waveform, not a list from the user
            with _without_large_mode_list_warning():
                h = super().generate(*parameter_values, **waveform_kwargs)
        else:
            h = super().generate(*parameter_values, **waveform_kwargs)
        # FEW keeps the modes of its latest waveform; remember which one that is
        self._last_generated = h
        return h

    def set_reference(self, parameters, waveform=None, **waveform_kwargs):
        """Set the reference point, and with `fix_modes` fix its modes.

        The arguments are those of `DerivativeBase.set_reference`.
        """
        self.modes = None
        if waveform is not None and waveform is not self._last_generated:
            # the modes FEW kept belong to another waveform: regenerate it
            waveform = None
        super().set_reference(parameters, waveform, **waveform_kwargs)

        if self.fix_modes and "mode_selection" not in waveform_kwargs:
            self.modes = self._reference_modes()
            if self.modes is not None:
                self.waveform_kwargs["mode_selection"] = self.modes

    def _reference_modes(self):
        """The (l, m, n) modes of FEW's latest waveform, or None if not available."""
        few_waveform = getattr(self.base_generator, "waveform_generator", None)
        if few_waveform is None or not hasattr(few_waveform, "ls"):
            return None  # no mode selection, e.g. the AAK
        selector = getattr(few_waveform, "mode_selector", None)
        if selector is not None and selector.is_predictive:
            logger.warning(
                "fix_modes has no effect with a predictive mode selector, "
                "which FEW uses to choose the modes of every waveform."
            )
            return None
        return [
            (int(l), int(m), int(n))
            for l, m, n in zip(few_waveform.ls, few_waveform.ms, few_waveform.ns)
        ]

    def compute(self, param_to_vary, delta, order, kind):
        if param_to_vary == "dist":
            return (-1 / self.parameters["dist"]) * self.waveform
        return self._finite_difference(param_to_vary, delta, order, kind)
