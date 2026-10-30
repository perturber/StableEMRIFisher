from .base import DerivativeBase
from .direct import DirectDerivative
from .stable import StableDerivative

# names accepted by StableEMRIFisher(deriv_type=...)
DERIVATIVES = {
    "direct": DirectDerivative,
    "stable": StableDerivative,
}


def get_derivative_class(deriv_type):
    """Return the derivative class for a name in DERIVATIVES or a DerivativeBase subclass."""
    if isinstance(deriv_type, str):
        try:
            return DERIVATIVES[deriv_type]
        except KeyError:
            raise ValueError(
                f"deriv_type must be one of {list(DERIVATIVES)} or a "
                f"DerivativeBase subclass, got {deriv_type!r}."
            ) from None
    if isinstance(deriv_type, type) and issubclass(deriv_type, DerivativeBase):
        return deriv_type
    raise ValueError(
        f"deriv_type must be one of {list(DERIVATIVES)} or a "
        f"DerivativeBase subclass, got {deriv_type!r}."
    )
