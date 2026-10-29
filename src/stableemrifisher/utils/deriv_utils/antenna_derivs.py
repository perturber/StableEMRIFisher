import numpy as np

sin = np.sin
cos = np.cos
atan2 = np.arctan2


def fplusI_derivs(qS, phiS, qK, phiK, with_respect_to):
    """
    Compute the partial derivatives of FplusI with respect to viewing angles.

    Parameters
    ----------
    qS, phiS, qK, phiK : float
        Detector-frame angles.
    with_respect_to : str
        Allowed values: 'qS','phiS','qK','phiK'.

    Returns
    -------
        the requested derivatives of FplusI.
    """

    if with_respect_to == "qS":
        return (
            2
            * (-sin(qK) * sin(qS) * cos(phiK - phiS) - cos(qK) * cos(qS))
            * sin(qK)
            * sin(phiK - phiS)
            * sin(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK)) ** 2
                + sin(qK) ** 2 * sin(phiK - phiS) ** 2
            )
        )
    elif with_respect_to == "phiS":
        return (
            2
            * (sin(qK) * cos(qS) - sin(qS) * cos(qK) * cos(phiK - phiS))
            * sin(qK)
            * sin(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    elif with_respect_to == "qK":
        return (
            2
            * sin(qS)
            * sin(phiK - phiS)
            * sin(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    elif with_respect_to == "phiK":
        return (
            2
            * (-sin(qK) * cos(qS) + sin(qS) * cos(qK) * cos(phiK - phiS))
            * sin(qK)
            * sin(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    else:
        raise ValueError(
            "Invalid parameter for differentiation. Allowed values: 'qS', 'phiS', 'qK', 'phiK'."
        )


def fcrossI_derivs(qS, phiS, qK, phiK, with_respect_to):
    """
    Compute the partial derivatives of FcrossI with respect to viewing angles.

    Parameters
    ----------
    qS, phiS, qK, phiK : float
        Detector-frame angles.
    with_respect_to : str
        Allowed values: 'qS','phiS','qK','phiK'.

    Returns
    -------
        the requested derivatives of FcrossI.
    """

    if with_respect_to == "qS":
        return (
            -2
            * (-sin(qK) * sin(qS) * cos(phiK - phiS) - cos(qK) * cos(qS))
            * sin(qK)
            * sin(phiK - phiS)
            * cos(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK)) ** 2
                + sin(qK) ** 2 * sin(phiK - phiS) ** 2
            )
        )
    elif with_respect_to == "phiS":
        return (
            2
            * (-sin(qK) * cos(qS) + sin(qS) * cos(qK) * cos(phiK - phiS))
            * sin(qK)
            * cos(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    elif with_respect_to == "qK":
        return (
            -2
            * sin(qS)
            * sin(phiK - phiS)
            * cos(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    elif with_respect_to == "phiK":
        return (
            2
            * (sin(qK) * cos(qS) - sin(qS) * cos(qK) * cos(phiK - phiS))
            * sin(qK)
            * cos(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    else:
        raise ValueError(
            "Invalid parameter for differentiation. Allowed values: 'qS', 'phiS', 'qK', 'phiK'."
        )


def fplusII_derivs(qS, phiS, qK, phiK, with_respect_to):
    """
    Compute the partial derivatives of FplusII with respect to viewing angles.

    Parameters
    ----------
    qS, phiS, qK, phiK : float
        Detector-frame angles.
    with_respect_to : str
        Allowed values: 'qS','phiS','qK','phiK'.

    Returns
    -------
        the requested derivatives of FplusII.
    """

    if with_respect_to == "qS":
        return (
            2
            * (-sin(qK) * sin(qS) * cos(phiK - phiS) - cos(qK) * cos(qS))
            * sin(qK)
            * sin(phiK - phiS)
            * cos(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK)) ** 2
                + sin(qK) ** 2 * sin(phiK - phiS) ** 2
            )
        )
    elif with_respect_to == "phiS":
        return (
            2
            * (sin(qK) * cos(qS) - sin(qS) * cos(qK) * cos(phiK - phiS))
            * sin(qK)
            * cos(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    elif with_respect_to == "qK":
        return (
            2
            * sin(qS)
            * sin(phiK - phiS)
            * cos(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    elif with_respect_to == "phiK":
        return (
            2
            * (-sin(qK) * cos(qS) + sin(qS) * cos(qK) * cos(phiK - phiS))
            * sin(qK)
            * cos(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    else:
        raise ValueError(
            "Invalid parameter for differentiation. Allowed values: 'qS', 'phiS', 'qK', 'phiK'."
        )


def fcrossII_derivs(qS, phiS, qK, phiK, with_respect_to):
    """
    Compute the partial derivatives of FcrossII with respect to viewing angles.

    Parameters
    ----------
    qS, phiS, qK, phiK : float
        Detector-frame angles.
    with_respect_to : str
        Allowed values: 'qS','phiS','qK','phiK'.

    Returns
    -------
        the requested derivatives of FcrossII.
    """

    if with_respect_to == "qS":
        return (
            2
            * (-sin(qK) * sin(qS) * cos(phiK - phiS) - cos(qK) * cos(qS))
            * sin(qK)
            * sin(phiK - phiS)
            * sin(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK)) ** 2
                + sin(qK) ** 2 * sin(phiK - phiS) ** 2
            )
        )
    elif with_respect_to == "phiS":
        return (
            2
            * (sin(qK) * cos(qS) - sin(qS) * cos(qK) * cos(phiK - phiS))
            * sin(qK)
            * sin(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    elif with_respect_to == "qK":
        return (
            2
            * sin(qS)
            * sin(phiK - phiS)
            * sin(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    elif with_respect_to == "phiK":
        return (
            2
            * (-sin(qK) * cos(qS) + sin(qS) * cos(qK) * cos(phiK - phiS))
            * sin(qK)
            * sin(
                2
                * atan2(
                    sin(qK) * cos(qS) * cos(phiK - phiS) - sin(qS) * cos(qK),
                    -sin(qK) * sin(phiK - phiS),
                )
            )
            / (
                (sin(phiK - phiS) ** 2 + cos(qS) ** 2 * cos(phiK - phiS) ** 2)
                * sin(qK) ** 2
                - 2 * sin(qK) * sin(qS) * cos(qK) * cos(qS) * cos(phiK - phiS)
                + sin(qS) ** 2 * cos(qK) ** 2
            )
        )
    else:
        raise ValueError(
            "Invalid parameter for differentiation. Allowed values: 'qS', 'phiS', 'qK', 'phiK'."
        )
