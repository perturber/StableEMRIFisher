import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import transforms
from matplotlib.ticker import MaxNLocator
import numpy as np


def _with_defaults(kwargs, defaults):
    """`kwargs` plus each default whose name (or a matplotlib alias of it) is not set.

    `defaults` maps a tuple of equivalent names, e.g. ("linewidth", "lw"), to a value.
    """
    kwargs = dict(kwargs)
    for names, value in defaults.items():
        if not any(name in kwargs for name in names):
            kwargs[names[0]] = value
    return kwargs


def _set_limits(set_lim, current, lo, hi, widen):
    """Set axis limits (lo, hi); if `widen`, only ever extend the `current` ones.

    Overlaid calls on the same panels then keep every curve in view.
    """
    if widen:
        lo, hi = min(lo, current[0]), max(hi, current[1])
    set_lim(lo, hi)


def cov_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
    """
    Plot a covariance ellipse.

    This function plots a covariance ellipse for visualizing the parameter covariance matrix.

    Args:
        mean (tuple): Mean of the distribution in the form of (mean_x, mean_y).
        cov (np.ndarray): Covariance matrix of the distribution.
        ax (matplotlib.axes.Axes): Axes object on which to plot the ellipse.
        n_std (float, optional): Number of standard deviations to encompass within the ellipse. Default is 1.0.
        **kwargs: Additional keyword arguments passed to matplotlib.patches.Ellipse.

    Returns:
        matplotlib.patches.Ellipse: The covariance ellipse plotted on the given Axes object.
    """

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def normal(mean, var, x):
    """Unnormalized Gaussian exp(-(x - mean)^2 / (2 var)), peaking at 1."""
    return np.exp(-((mean - x) ** 2) / var / 2)


def _format_number(x, sig_figs):
    """LaTeX string of x to `sig_figs` significant figures, e.g. 13.8 or 3.6\\times 10^{-3}.

    Numbers of 1000 or more, or below 0.01, are written in scientific notation.
    """
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.{sig_figs - 1}e}".split("e")
    exponent = int(exponent)
    if -2 <= exponent <= 2:
        decimals = max(0, sig_figs - 1 - exponent)
        return f"{float(mantissa) * 10.0**exponent:.{decimals}f}"
    return rf"{mantissa}\times 10^{{{exponent}}}"


def format_sigma(value, sigma):
    """LaTeX string "value \\pm sigma": value to 3 and sigma to 2 significant figures.

    Numbers of 1000 or more, or below 0.01, are written in scientific notation,
    e.g. 1.00\\times 10^{6} \\pm 3.6\\times 10^{3}.
    """
    if not (np.isfinite(value) and np.isfinite(sigma) and sigma > 0):
        return _format_number(value, 3)
    return rf"{_format_number(value, 3)} \pm {_format_number(sigma, 2)}"


def plot_covariance_ellipses(
    covariance,
    param_names=None,
    wave_params=None,
    fig=None,
    axs=None,
    filename=None,
    ellipse_kwargs=None,
    line_kwargs=None,
    show_titles=True,
):
    """
    Corner plot of a covariance matrix: 1D Gaussians on the diagonal and
    covariance ellipses for every pair of parameters below it.

    Args:
        covariance (np.ndarray): covariance matrix, shape (N_params, N_params).
        param_names (list of str, optional): axis labels, and keys into
            `wave_params`.
        wave_params (dict, optional): parameter values used as the centres.
            Default: all centred on 0.
        fig, axs (optional): existing figure and N_params x N_params axes to
            draw on, e.g. to overlay several confidence levels: pass the
            (fig, axs) returned by a first call. A new figure is sized to the
            number of parameters, with little space between the panels.
        filename (str, optional): save the figure there and close it.
            Otherwise (fig, axs) is returned.
        ellipse_kwargs (dict, optional): kwargs for the ellipses
            (matplotlib.patches.Ellipse), plus "n_std", the number of standard
            deviations they enclose (default 2). Default: unfilled, edge
            color "C0", line width 2.
        line_kwargs (dict, optional): kwargs for the 1D Gaussian lines.
            Default: color "C0", line width 2.
        show_titles (bool, optional): title each 1D panel with the value and
            its 1-sigma uncertainty (see `format_sigma`). Default True.

    Returns:
        (fig, axs) if `filename` is None.
    """
    ellipse_kwargs = _with_defaults(
        ellipse_kwargs or {},
        {
            ("facecolor", "fc"): "none",
            ("edgecolor", "ec"): "C0",
            ("linewidth", "lw"): 2.0,
        },
    )
    line_kwargs = _with_defaults(
        line_kwargs or {}, {("color", "c"): "C0", ("linewidth", "lw"): 2.0}
    )

    n_params = len(covariance)
    if fig is None:
        size = max(3.0, 2.2 * n_params)
        fig, axs = plt.subplots(n_params, n_params, figsize=(size, size), squeeze=False)
        # margins in inches: the top panel's title, and the tick labels and axis labels
        top = 1.0 - (0.45 if show_titles else 0.1) / size
        fig.subplots_adjust(
            left=1.0 / size, bottom=1.0 / size, top=top, right=0.98, wspace=0.06, hspace=0.06
        )
    # a single parameter gives a single Axes; index it as a 1x1 grid
    axs = np.atleast_2d(axs)

    if "n_std" in ellipse_kwargs:
        n_std = ellipse_kwargs["n_std"]
    else:
        n_std = 2.0
        ellipse_kwargs["n_std"] = n_std
    # half-width of each panel, in standard deviations: a margin around the ellipses
    half_width = 1.25 * n_std

    # first param index
    for i in range(len(covariance)):
        # second param index
        for j in range(i, len(covariance)):

            if i != j:
                cov = np.array(
                    (
                        (covariance[i][i], covariance[i][j]),
                        (covariance[j][i], covariance[j][j]),
                    )
                )
                # print(cov)

                if wave_params is not None:
                    mean = np.array(
                        (wave_params[param_names[i]], wave_params[param_names[j]])
                    )
                else:
                    mean = np.zeros(2)

                had_data = axs[j, i].has_data()
                cov_ellipse(mean, cov, axs[j, i], **ellipse_kwargs)

                _set_limits(
                    axs[j, i].set_xlim,
                    axs[j, i].get_xlim(),
                    mean[0] - half_width * np.sqrt(covariance[i][i]),
                    mean[0] + half_width * np.sqrt(covariance[i][i]),
                    had_data,
                )
                _set_limits(
                    axs[j, i].set_ylim,
                    axs[j, i].get_ylim(),
                    mean[1] - half_width * np.sqrt(covariance[j][j]),
                    mean[1] + half_width * np.sqrt(covariance[j][j]),
                    had_data,
                )

                if param_names is not None:
                    axs[j, i].set_xlabel(param_names[i], fontsize=14)
                    axs[j, i].set_ylabel(param_names[j], fontsize=14)

            else:

                if wave_params is not None:
                    mean = wave_params[param_names[i]]
                else:
                    mean = 0.0

                var = covariance[i][i]

                x = np.linspace(
                    mean - half_width * np.sqrt(var), mean + half_width * np.sqrt(var), 200
                )

                had_data = axs[j, i].has_data()
                axs[j, i].plot(x, normal(mean, var, x), **line_kwargs)

                _set_limits(axs[j, i].set_xlim, axs[j, i].get_xlim(), x[0], x[-1], had_data)

                # the 1D Gaussians peak at 1: their y axis carries no information
                axs[j, i].set_ylim(0.0, 1.1)
                axs[j, i].set_yticks([])

                if param_names is not None:
                    axs[j, i].set_xlabel(param_names[i], fontsize=14)

                if show_titles:
                    label = f"{param_names[i]} = " if param_names is not None else ""
                    axs[j, i].set_title(
                        label + f"${format_sigma(mean, np.sqrt(var))}$", fontsize=11
                    )

    # a few widely spaced ticks, inside the panels, without an offset (e.g. "+1.38e1")
    for row, col in zip(*np.tril_indices(n_params)):
        ax = axs[row, col]
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        if row != col:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        try:
            ax.ticklabel_format(useOffset=False)
        except AttributeError:  # an axis without a ScalarFormatter, e.g. a log axis
            pass
        ax.tick_params(direction="in", length=5, width=1, labelsize=10, top=True, right=True)
        ax.label_outer()
    # tilt the tick labels of the outer panels so that they do not overlap
    for ax in axs[-1, :]:
        ax.tick_params(axis="x", labelrotation=45)
    for ax in axs[1:, 0]:
        ax.tick_params(axis="y", labelrotation=45)

    for i in range(len(covariance)):
        for j in range(i + 1, len(covariance)):
            try:
                fig.delaxes(axs[i, j])
            except KeyError:
                continue

    if filename is None:
        return fig, axs
    else:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


def plot_stability(deltas, Gammas, stable_index=None, param_name=None, filename=None):
    """
    Plot a diagonal Fisher element against the trial step size (step-size search).

    Args:
        deltas (array): trial step sizes.
        Gammas (array): diagonal Fisher element at each step size.
        stable_index (int, optional): index of the chosen step, highlighted.
        param_name (str, optional): parameter name, for the title.
        filename (str, optional): save the figure there and close it.
            Otherwise it is shown.
    """

    plt.figure(figsize=(12, 5))
    plt.loglog(deltas, Gammas, "ro-")

    if stable_index != None:
        plt.scatter(
            deltas[stable_index],
            Gammas[stable_index],
            marker="s",
            s=20,
            color="b",
            label="chosen step size",
            zorder=5,
        )

    if param_name != None:
        plt.xlabel(r"$\Delta\theta_i$", fontsize=12)

    plt.ylabel(
        r"$\left.\langle \frac{\partial h}{\partial \theta_i}\right|\frac{\partial h}{\partial \theta_i}\rangle$",
        fontsize=14,
    )
    plt.title(r"$\theta_i = $" + f"${param_name}$", fontsize=12)
    plt.grid(True)
    plt.legend()

    if filename != None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
