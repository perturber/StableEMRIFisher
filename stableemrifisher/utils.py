import numpy as np
try:
    import cupy as xp
    GPU_AVAILABLE = True
except ImportError or ModuleNotFoundError:
    xp = np
    GPU_AVAILABLE = False

def tukey(N, alpha=0.5, use_gpu=False):
    """
    Generate a Tukey window function using GPU acceleration.

    Parameters:
    - N (int): The number of points in the window.
    - alpha (float): Shape parameter of the Tukey window. It determines the fraction of the window inside the tapered regions. 
      When alpha=0, the Tukey window reduces to a rectangular window, and when alpha=1, it reduces to a Hann window.

    Returns:
    - window (cupy.ndarray): The Tukey window function as a 1-dimensional CuPy array of length N.

    Note:
    The Tukey window is defined as a function of the input vector t, where t is a linearly spaced vector from 0 to 1 
    with N points. The function computes the values of the Tukey window function at each point in t using GPU-accelerated 
    operations and returns the resulting window as a CuPy array.
    """
    if not use_gpu:
        xp = np
    else:
        assert GPU_AVAILABLE

    t = xp.linspace(0., 1., N)
    window = xp.ones(N)
    condition1 = (t > (1 - alpha / 2)) & (t <= 1)
    condition2 = (t >= 0) & (t < alpha / 2)
    window[condition1] = 0.5 * (1 + xp.cos(2 * xp.pi / alpha * ((t[condition1] - 1 + alpha / 2) - 1)))
    window[condition2] = 0.5 * (1 + xp.cos(2 * xp.pi / alpha * (t[condition2] - alpha / 2)))
    return window

def inner_product(a, b, PSD, dt, window = None, use_gpu=False):
    """
    Compute the frequency domain inner product of two time-domain arrays.

    This function computes the frequency domain inner product of two time-domain arrays using the GPU for acceleration.
    It operates under the assumption that the signals are evenly spaced and applies a Tukey window to each signal.
    This function is optimized for GPUs.

    Args:
        a (np.ndarray): The first time-domain signal. It should have dimensions (N_channels, N), where N is the length of the signal.
        b (np.ndarray): The second time-domain signal. It should have dimensions (N_channels, N), where N is the length of the signal.
        df (float): The frequency resolution, i.e., the spacing between frequency bins.
        PSD (np.ndarray): The power spectral density (PSD) of the signals. It should be a 1D array of length N_channels.

    Returns:
        float: The frequency-domain inner product of the two signals.

    Note:
        This function requires a GPU for execution.

    """
    if not use_gpu:
        xp = np
    else:
        assert GPU_AVAILABLE

    a = xp.asarray(a)
    b = xp.asarray(b)
    PSD = xp.asarray(PSD)

    N = a.shape[1]

    df = (N * dt) ** -1

    if window is not None:
        window = xp.atleast_2d(xp.asarray(window))
        a_in = a * window
        b_in = b * window
    else:
        a_in, b_in = a, b

    a_fft = dt * xp.fft.rfft(a_in, axis=1)[:,1:]
    b_fft = dt * xp.fft.rfft(b_in, axis=1)[:,1:]

    PSD = xp.atleast_2d(PSD)

    # Compute inner products over given channels
    inner_product = 4 * df * ((a_fft.conj() * b_fft).real / PSD).sum()

    #clearing cupy cache  TODO: do we need this any more?
    cache = xp.fft.config.get_plan_cache()
    cache.clear()
    
    return inner_product

def padding(a, b, use_gpu=False):
    """
    Make time series 'a' the same length as time series 'b'.
    Both 'a' and 'b' must be cupy array.

    returns padded 'a'
    """
    if not use_gpu:
        xp = np
    else:
        assert GPU_AVAILABLE
    a = xp.asarray(a)
    b = xp.asarray(b)

    if len(a) < len(b):
        return xp.concatenate((a,xp.zeros(len(b)-len(a))))

    elif len(a) > len(b):
        return a[:len(b)]

    else:
        return a
