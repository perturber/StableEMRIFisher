import numpy as np
from utils import padding
import warnings

try:
    import cupy as xp
except ImportError or ModuleNotFoundError:
    xp = np
    GPU_AVAILABLE=False

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
        4: np.asarray([-25/12, 4, -3, 4/3, -1/4])
    },
    "backward": {
        2: np.asarray([1/2, -2, 3/2]),
        4: np.asarray([1/4, -4/3, 3, -4, 25/12])
    }
}

def handle_a_flip(params):
    if params['a'] < 0:
        params['a'] *= -1.
        params['Y0'] = -1.
    return params

def derivative(waveform_generator, parameters, i, delta, order=6, kind="central", SFN=False, use_gpu=False, waveform=None, waveform_kwargs=None):
    if kind not in ["central", "forward", "backward"]:
        raise ValueError('"kind" must be one of ("central", "forward", "backward") ')
    if not use_gpu:
        xp = np
    else:
        assert GPU_AVAILABLE

    if waveform_kwargs is None:
        waveform_kwargs = {}

    order = int(order)

    parameter_names = list(parameters.keys())

    if waveform is None:
        waveform = xp.asarray(waveform_generator(*list(parameters), **waveform_kwargs))
        if waveform.ndim == 1:
            waveform = xp.asarray([waveform.real, waveform.imag])

    if parameter_names[i] == "dist":
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
                temp[parameter_names[i]] -= delta
                temp = handle_a_flip(temp)

                # Print details if wanted
                if SFN:    
                    print("For parameter",parameter_names[i])
                    print(parameter_names[i],' = ', temp[parameter_names[i]])
                
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
                temp[parameter_names[i]] += delta
                temp = handle_a_flip(temp)

                # Print details if wanted
                if SFN:    
                    print("For parameter",parameter_names[i])
                    print(parameter_names[i],' = ', temp[parameter_names[i]])
                
                temp_vals = list(temp.values())
                
                waveform_delta = xp.asarray(waveform_generator(*temp_vals, **waveform_kwargs))

                if waveform_delta.ndim == 1:
                    waveform_delta = xp.asarray([waveform_delta.real, waveform_delta.imag])

                waveform_delta = padding(waveform_delta, waveform)

                delta_waveforms.append(waveform_delta)

        elif kind == "forward":
            # forwards deltas
            temp = parameters.copy()
            delta_waveforms = [waveform, ]

            if order > 4:
                warnings.warn('forward derivatives only available to 4th order accuracy. Setting der_order = 4')
                order = 4

            for _ in range(order):
                temp[parameter_names[i]] += delta
                temp = handle_a_flip(temp)

                # Print details if wanted
                if SFN:    
                    print("For parameter",parameter_names[i])
                    print(parameter_names[i],' = ', temp[parameter_names[i]])
                
                temp_vals = list(temp.values())
                
                waveform_delta = xp.asarray(waveform_generator(*temp_vals, **waveform_kwargs))

                if waveform_delta.ndim == 1:
                    waveform_delta = xp.asarray([waveform_delta.real, waveform_delta.imag])

                waveform_delta = padding(waveform_delta, waveform)

                delta_waveforms.append(waveform_delta)

        elif kind == "backward":
            # backwards deltas
            temp = parameters.copy()
            delta_waveforms = []

            if order > 4:
                warnings.warn('backward derivatives only available to 4th order accuracy. Setting der_order = 4')
                order = 4

            for _ in range(order):
                temp[parameter_names[i]] -= delta
                temp = handle_a_flip(temp)

                # Print details if wanted
                if SFN:    
                    print("For parameter",parameter_names[i])
                    print(parameter_names[i],' = ', temp[parameter_names[i]])
                
                temp_vals = list(temp.values())
                
                waveform_delta = xp.asarray(waveform_generator(*temp_vals, **waveform_kwargs))

                if waveform_delta.ndim == 1:
                    waveform_delta = xp.asarray([waveform_delta.real, waveform_delta.imag])

                waveform_delta = padding(waveform_delta, waveform)

                delta_waveforms.append(waveform_delta)
            
            # flip for stencil order
            delta_waveforms = delta_waveforms[::-1]
            delta_waveforms.append(waveform)

        try:
            derivative = xp.asarray(stencils[kind][order]) * xp.asarray(delta_waveforms) / delta
        except KeyError:
            raise ValueError(f"Order '{order}' of derivative '{kind}' not supported")
        
        del delta_waveforms

        return derivative

