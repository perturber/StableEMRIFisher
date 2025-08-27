import numpy as np
import os
#from stableemrifisher.noise import noise_PSD_AE
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from lisatools.sensitivity import *
from lisatools.utils.constants import lisaLT

def write_psd_file(model='scirdv1', 
                    channels='AET', 
                    tdi2=True,
                    include_foreground=False,
                    filename="example_psd.npy",
                    **kwargs
                   ):
    """
    Write a PSD file for a given model.

    Parameters
    ----------
    model : str
        The noise model to use. Default is 'scirdv1'.
    channels : str
        The channels to include in the PSD. Default is 'AET'. if None, the sensitivity curve without projections is computed.
    tdi2 : bool 
        Whether to use Second generation TDI. Default is True.
    include_foreground : bool
        Whether to include the foreground noise. Default is False. This is just an extra check, the actual
        argument is in the kwargs.
    filename : str
        The name of the file to save the PSD to. Default is 'example_psd.npy'.
    **kwargs : dict
        Additional keyword arguments to pass to the PSD generation function.
    """
    
    assert channels in [None,  'A', 'AE', 'AET'], "channels must be None, 'A', 'AE', or 'AET'"
    if include_foreground:
        assert 'stochastic_params' in kwargs.keys(), "`stochastic_params = List(Tobs) [s]` must be provided if include_foreground is True"

    freqs = np.linspace(0, 1, 100001)[1:]
    
    if channels is None:
        sens_fns = [LISASens]

        default_kwargs = dict(
        return_type='PSD',
        average=False
    )
        
    elif 'A' in channels:
        sens_fns = [A1TDISens]
        if 'E' in channels:
            sens_fns.append(E1TDISens)
        if 'T' in channels:
            sens_fns.append(T1TDISens)
        
        default_kwargs = dict(
            return_type='PSD',
        )

    updated_kwargs = default_kwargs | kwargs

    Sn = [get_sensitivity(freqs, sens_fn=sens_fn, model=model, **updated_kwargs) for sens_fn in sens_fns]

    if tdi2:
        x = 2.0 * np.pi * lisaLT * freqs
        tdi_factor = 4 * np.sin(2*x)**2
        Sn = [sens*tdi_factor for sens in Sn]

    Sn = np.array(Sn)
    np.save(filename,np.vstack((freqs, Sn)).T)


def load_psd_from_file(psd_file, xp=np):
    """
    Load the PSD from a file and return an interpolant.

    Parameters
    ----------
    psd_file : str
        The name of the file to load the PSD from.
    xp : module
        The module to use for array operations. Default is np.
    
    Returns
    -------
    psd_clipped : function
        A function that takes a frequency and returns the PSD at that frequency.
    """

    psd_in = np.load(psd_file).T
    freqs, values = psd_in[0], np.atleast_2d(psd_in[1:])

    #convert to cupy if needed
    freqs = xp.asarray(freqs)
    values = xp.asarray(values)

    backend = 'cpu' if xp is np else 'gpu'
    print(f"Using {backend} backend for PSD interpolation")
    #min_psd = values[:,0]#np.min(values, axis=1)

    min_psd = np.min(values[:, freqs < 1e-2], axis=1) # compatible with both tdi 1 and tdi 2
    max_psd = np.max(values, axis=1)
    print("PSD range", min_psd, max_psd)
    psd_interp = CubicSplineInterpolant(freqs, values, force_backend=backend)

    def psd_clipped(f, **kwargs):
        f = xp.clip(f, 0.00001, 1.0)

        out = xp.array(
            [
                xp.clip(xp.atleast_2d(psd_interp(f))[i], min_psd[i], max_psd[i]) for i in range(len(values))
            ]
        )
        return xp.squeeze(out) # remove the extra dimension if there is only one channel
    return psd_clipped    

def load_psd(
            logger,
            model='scirdv1', 
            channels='AET', 
            tdi2=True, 
            include_foreground=False,
            filename="example_psd.npy",
            xp=np,
            **kwargs
            ):
    """
    Load the PSD from a file and returns an interpolant. If the file does not exist, it will be created.

    Parameters
    ----------
    model : str
        The noise model to use. Default is 'scirdv1'.
    channels : str
        The channels to include in the PSD. Default is 'AET'.
    tdi2 : bool 
        Whether to use Second generation TDI. Default is True.
    include_foreground : bool
        Whether to include the foreground noise. Default is False. This is just an extra check, the actual
        argument is in the kwargs.
    filename : str
        The name of the file to save the PSD to. Default is 'example_psd.npy'.
    xp : module
        The module to use for array operations. Default is np.
    **kwargs : dict
        Additional keyword arguments to pass to the PSD generation function.

    Returns
    -------
    psd_clipped : function
        A function that takes a frequency and returns the PSD at that frequency
    """
    if filename is None or filename == "None":
        tdi_gen = 'tdi2' if tdi2 else 'tdi1'
        foreground = 'wd' if include_foreground else 'no_wd'    
        filename = f"noise_psd_{model}_{channels}_{tdi_gen}_{foreground}.npy"
    if not os.path.exists(filename):
        logger.warning(f"PSD file {filename} does not exist. Creating it now.")
        write_psd_file(model=model, channels=channels, tdi2=tdi2, include_foreground=include_foreground, filename=filename, **kwargs)
    
    logger.info(f"Loading PSD from {filename}")
    return load_psd_from_file(filename, xp=xp)

def get_psd_kwargs(kwargs):
    """
    Return a dictionary of default settings for PSD generation. Use the input dictionary to override the defaults.
    """
    default_settings = {
        "model": "scirdv1",
        "channels": "A",
    }
    return default_settings | kwargs

def compute_snr2(freqs, tdi_channels, psd_fn, dt, xp=np):
    """
    """
    df = freqs[2] - freqs[1]
    
    prefactor = 4 * df
    snr2 = prefactor * xp.sum(
        xp.abs(tdi_channels)**2 / psd_fn(freqs),
        axis=(0,1)
    )

    return snr2




if __name__ == "__main__":
    filename = 'test/psd.npy'
    channels = 'A'
    write_psd_file(filename=filename, channels=channels)
    fn = load_psd(filename)

    f = np.linspace(0.0001, 1, 1000)
