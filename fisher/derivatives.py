import numpy as np
from stableemrifisher.utils import padding
import logging
import sys
logger = logging.getLogger("stableemrifisher")

try:
    import cupy as cp
except:
    logger.info("CuPy not found")
    pass


# store the stencils of different orders here
# stencils all count from - to +
# forward stencils start with the zero-delta (i.e. the waveform)
# backward stencils end with the zero-delta
 #TODO just combine the forward and backward stencils...
stencils = {
    "central": {
        2: np.asarray([-1/2, 1/2]),
        4: np.asarray([1/12, -2/3, 2/3, -1/12]),
        6: np.asarray([-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]),
        8: np.asarray([1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280])
    },
    "forward": {
        2: np.asarray([-3/2, 2, -1/2]),
        4: np.asarray([-25/12, 4, -3, 4/3, -1/4]),
        6: np.asarray([-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6]),
        8: np.asarray([-761/280, 8, -14, 56/3, -35/2, 56/5, -14/3, 8/7, -1/8]) #Fornberg 1988: https://doi.org/10.1090%2FS0025-5718-1988-0935077-0
    },
    "backward": {
        2: np.asarray([1/2, -2, 3/2]),
        4: np.asarray([1/4, -4/3, 3, -4, 25/12]),
        6: np.asarray([1/6, -6/5, 15/4, -20/3, 15/2, -6, 49/20]),
        8: np.asarray([1/8, -8/7, 14/3, -56/5, 35/2, -56/3, 14, -8, 761/280]) #Fornberg 1988: https://doi.org/10.1090%2FS0025-5718-1988-0935077-0
    }
}

def handle_a_flip(params):
    if params['a'] < 0:
        params['a'] *= -1.
        params['Y0'] = -1.
    return params

def derivative(waveform_generator, parameters, param_to_vary, delta, order=4, kind="central", use_gpu=False, waveform=None, waveform_kwargs=None):
    
    if kind not in ["central", "forward", "backward"]:
        raise ValueError('"kind" must be one of ("central", "forward", "backward") ')
    if use_gpu:
        xp = cp
    else:
        xp = np

    if waveform_kwargs is None:
        waveform_kwargs = {}

    order = int(order)

    if waveform is None:
        parameters = handle_a_flip(parameters)
        waveform = xp.asarray(waveform_generator(*list(parameters.values()), **waveform_kwargs))
        if waveform.ndim == 1:
            waveform = xp.asarray([waveform.real, waveform.imag])

    if param_to_vary == "dist":
        # Compute derivative analytically for the distance
        derivative = (-1/parameters["dist"]) * waveform
        return derivative
    else:
        # modifying the given parameter
        temp = parameters.copy()
        delta_waveforms = []

        # handle backwards, central and forwards derivatives differently
        if kind == "central":
            # backwards deltas
            for _ in range(order // 2):
                temp[param_to_vary] -= delta
                temp = handle_a_flip(temp)
                
                logger.debug(f"For parameter {param_to_vary}")
                logger.debug(f"{param_to_vary} = {temp[param_to_vary]}")
                
                temp_vals = list(temp.values())
                
                waveform_delta = xp.asarray(waveform_generator(*temp_vals, **waveform_kwargs))

                if waveform_delta.ndim == 1:
                    waveform_delta = xp.asarray([waveform_delta.real, waveform_delta.imag])

                delta_waveforms.append(waveform_delta)

            # flip this for the stencil ordering later
            delta_waveforms = delta_waveforms[::-1]

            # forwards deltas
            temp = parameters.copy()

            for _ in range(order // 2):
                temp[param_to_vary] += delta
                temp = handle_a_flip(temp)

                logger.debug(f"For parameter {param_to_vary}")
                logger.debug(f"{param_to_vary} = {temp[param_to_vary]}")
                
                temp_vals = list(temp.values())
                
                waveform_delta = xp.asarray(waveform_generator(*temp_vals, **waveform_kwargs))

                if waveform_delta.ndim == 1:
                    waveform_delta = xp.asarray([waveform_delta.real, waveform_delta.imag])

                waveform_delta = padding(waveform_delta, waveform, use_gpu=use_gpu)

                delta_waveforms.append(waveform_delta)

        elif kind == "forward":
            # forwards deltas
            temp = parameters.copy()
            delta_waveforms = [waveform, ]

            for _ in range(order):
                temp[param_to_vary] += delta
                temp = handle_a_flip(temp)

                logger.debug(f"For parameter {param_to_vary}")
                logger.debug(f"{param_to_vary} = {temp[param_to_vary]}")
                
                temp_vals = list(temp.values())
                
                waveform_delta = xp.asarray(waveform_generator(*temp_vals, **waveform_kwargs))

                if waveform_delta.ndim == 1:
                    waveform_delta = xp.asarray([waveform_delta.real, waveform_delta.imag])

                waveform_delta = padding(waveform_delta, waveform, use_gpu=use_gpu)

                delta_waveforms.append(waveform_delta)

        elif kind == "backward":
            # backwards deltas
            temp = parameters.copy()
            delta_waveforms = []

            for _ in range(order):
                temp[param_to_vary] -= delta
                temp = handle_a_flip(temp)
    
                logger.debug(f"For parameter {param_to_vary}")
                logger.debug(f"{param_to_vary} = {temp[param_to_vary]}")
                
                temp_vals = list(temp.values())
                
                waveform_delta = xp.asarray(waveform_generator(*temp_vals, **waveform_kwargs))

                if waveform_delta.ndim == 1:
                    waveform_delta = xp.asarray([waveform_delta.real, waveform_delta.imag])

                waveform_delta = padding(waveform_delta, waveform, use_gpu=use_gpu)

                delta_waveforms.append(waveform_delta)
            
            # flip for stencil order
            delta_waveforms = delta_waveforms[::-1]
            delta_waveforms.append(waveform)

        try:
            derivative = (xp.asarray(stencils[kind][order])[:,None,None] * xp.asarray(delta_waveforms)).sum(0) / delta
        except KeyError:
            raise ValueError(f"Order '{order}' of derivative '{kind}' not supported")
        
        del delta_waveforms

        return derivative

