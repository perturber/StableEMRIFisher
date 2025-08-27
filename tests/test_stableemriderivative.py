"""
Pytest test suite for StableEMRIDerivative class.

This test validates the numerical derivatives computed by StableEMRIDerivative
against manual finite difference calculations using noise-weighted inner products.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

# Import the classes to test
from stableemrifisher.fisher.new_derivs import StableEMRIDerivative
from stableemrifisher.utils import inner_product, generate_PSD, padding

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Test configuration - use short waveforms for speed
TEST_CONFIG = {
    'T': 0.01,  # Very short observation time (0.01 years)
    'dt': 10.0,  # Time step in seconds
    'delta': 1e-8,  # Finite difference step size
    'order': 4,  # Finite difference order
    'kind': "central",  # Central differences
    'overlap_threshold': 0.99,  # Minimum overlap for test to pass
}

# Standard EMRI test parameters
STANDARD_PARAMS = {
    'm1': 1e6,
    'm2': 10.0,
    'a': 0.0,  # Non-spinning for simplicity
    'p0': 9.5,
    'e0': 0.4,
    'xI0': 1.0,
    'dist': 1.0,
    'qS': np.pi/3,
    'phiS': np.pi/4,
    'qK': np.pi/6,
    'phiK': np.pi/8,
    'Phi_phi0': np.pi/4,
    'Phi_theta0': 0.0,
    'Phi_r0': 0.0,
}

# Parameters to test derivatives for
PARAMS_TO_TEST = ['p0', 'e0', 'm2', 'phiK', 'qS', 'dist']


class TestStableEMRIDerivative:
    """Test suite for StableEMRIDerivative class."""
    
    @pytest.fixture
    def waveform_derivative(self):
        """Create StableEMRIDerivative instance for testing."""
        from few.waveform import FastSchwarzschildEccentricFlux
        
        return StableEMRIDerivative(
            waveform_class=FastSchwarzschildEccentricFlux,
            mode_selector_kwargs=dict(mode_selection_threshold=1e-3),
            inspiral_kwargs=dict(err=1e-11, max_iter=10000),
            force_backend='cpu'  # Force CPU for consistent testing
        )
    
    @pytest.fixture
    def base_waveform_generator(self):
        """Create base waveform generator for manual derivative calculation."""
        from few.waveform import FastSchwarzschildEccentricFlux
        
        return FastSchwarzschildEccentricFlux(
            inspiral_kwargs=dict(err=1e-11, max_iter=10000)
        )
    
    def generate_waveform(self, waveform_gen, parameters: Dict[str, float]) -> np.ndarray:
        """Generate a waveform with given parameters."""
        wave = waveform_gen(
            parameters['m1'],
            parameters['m2'],
            parameters['a'],
            parameters['p0'],
            parameters['e0'],
            parameters['xI0'],
            parameters['dist'],
            parameters['qS'],
            parameters['phiS'],
            parameters['qK'],
            parameters['phiK'],
            Phi_phi0=parameters['Phi_phi0'],
            Phi_theta0=parameters['Phi_theta0'],
            Phi_r0=parameters['Phi_r0'],
            dt=TEST_CONFIG['dt'],
            T=TEST_CONFIG['T'],
        )
        return wave
    
    def compute_manual_derivative(
        self, 
        waveform_gen, 
        parameters: Dict[str, float], 
        param_to_vary: str,
        Npad: int = 0
    ) -> np.ndarray:
        """
        Compute derivative manually using finite differences.
        
        This replicates the manual calculation shown in the notebook.
        """
        delta = TEST_CONFIG['delta']
        order = TEST_CONFIG['order']
        kind = TEST_CONFIG['kind']
        
        # Generate coefficient and offset arrays for finite differences
        if kind == "central":
            if order == 2:
                coeffs = np.array([-1, 1]) / 2
                offsets = np.array([-1, 1])
            elif order == 4:
                coeffs = np.array([1, -8, 8, -1]) / 12
                offsets = np.array([-2, -1, 1, 2])
            else:
                raise ValueError(f"Order {order} not implemented for central differences")
        else:
            raise ValueError(f"Kind '{kind}' not implemented")
        
        # Generate waveforms at offset points
        wavdelts = []
        
        for offset in offsets:
            parameters_offset = parameters.copy()
            param_value = parameters_offset[param_to_vary]
            parameters_offset[param_to_vary] = param_value + offset * delta * abs(param_value)
            
            wave = self.generate_waveform(waveform_gen, parameters_offset)
            
            # Apply padding if needed
            if Npad > 0:
                wave = np.concatenate((wave[:-Npad], np.zeros(Npad)), dtype=wave.dtype)
            
            wavdelts.append(wave)
        
        # Compute finite difference
        derivative = np.zeros_like(wavdelts[0])
        for coeff, wave in zip(coeffs, wavdelts):
            derivative += coeff * wave
        
        # Scale by step size
        param_value = parameters[param_to_vary]
        derivative /= (delta * abs(param_value))
        
        return derivative
    
    def compute_overlap(
        self, 
        derivative1: np.ndarray, 
        derivative2: np.ndarray, 
        wave_ref: np.ndarray
    ) -> float:
        """
        Compute overlap between two derivatives using noise-weighted inner product.
        """
        # Generate PSD from reference waveform
        wave_list = [wave_ref.real, -wave_ref.imag]
        PSD = generate_PSD(wave_list, dt=TEST_CONFIG['dt'])
        
        # Convert derivatives to real/imaginary lists
        der1_list = [derivative1.real, -derivative1.imag]
        der2_list = [derivative2.real, -derivative2.imag]
        
        # Compute inner products
        inner11 = inner_product(der1_list, der1_list, PSD=PSD, dt=TEST_CONFIG['dt'])
        inner22 = inner_product(der2_list, der2_list, PSD=PSD, dt=TEST_CONFIG['dt'])
        inner12 = inner_product(der1_list, der2_list, PSD=PSD, dt=TEST_CONFIG['dt'])
        
        # Compute overlap
        overlap = inner12 / np.sqrt(inner11 * inner22)
        return overlap.real
    
    @pytest.mark.parametrize("param_to_vary", PARAMS_TO_TEST)
    def test_derivative_overlap(
        self, 
        waveform_derivative, 
        base_waveform_generator, 
        param_to_vary
    ):
        """
        Test that StableEMRIDerivative matches manual finite difference calculation.
        
        Uses overlap/inner product as the comparison metric.
        """
        # Generate derivative using StableEMRIDerivative
        stable_derivative = waveform_derivative(
            T=TEST_CONFIG['T'],
            dt=TEST_CONFIG['dt'],
            parameters=STANDARD_PARAMS,
            param_to_vary=param_to_vary,
            delta=TEST_CONFIG['delta'],
            order=TEST_CONFIG['order'],
            kind=TEST_CONFIG['kind'],
        )
        
        # Get padding info from cache
        Npad = waveform_derivative.cache.get('Npad', 0)
        if Npad == 0:
            Npad = 1
        
        # Generate manual derivative
        manual_derivative = self.compute_manual_derivative(
            base_waveform_generator,
            STANDARD_PARAMS,
            param_to_vary,
            Npad=Npad
        )
        
        # Generate reference waveform for PSD
        wave_ref = self.generate_waveform(base_waveform_generator, STANDARD_PARAMS)
        if Npad > 0:
            wave_ref = np.concatenate((wave_ref[:-Npad], np.zeros(Npad)), dtype=wave_ref.dtype)
        
        # Compute overlap
        overlap = self.compute_overlap(stable_derivative, manual_derivative, wave_ref)
        
        # Check that overlap is very high (indicating agreement)
        assert overlap > TEST_CONFIG['overlap_threshold'], (
            f"Derivative overlap {overlap:.6f} below threshold "
            f"{TEST_CONFIG['overlap_threshold']:.6f} for parameter {param_to_vary}"
        )
        
        print(f"✓ Parameter {param_to_vary}: overlap = {overlap:.6f}")
    
    def test_derivative_basic_properties(self, waveform_derivative):
        """Test basic properties of computed derivatives."""
        param_to_vary = 'p0'
        
        derivative = waveform_derivative(
            T=TEST_CONFIG['T'],
            dt=TEST_CONFIG['dt'],
            parameters=STANDARD_PARAMS,
            param_to_vary=param_to_vary,
            delta=TEST_CONFIG['delta'],
            order=TEST_CONFIG['order'],
            kind=TEST_CONFIG['kind'],
        )
        
        # Check that derivative is not all zeros
        assert not np.allclose(derivative, 0), "Derivative should not be all zeros"
        
        # Check that derivative is finite
        assert np.all(np.isfinite(derivative)), "Derivative should be finite everywhere"
        
        # Check that derivative has correct type
        assert isinstance(derivative, np.ndarray), "Derivative should be numpy array"
        
        print(f"✓ Derivative has correct basic properties")
    
    def test_derivative_scaling(self, waveform_derivative, base_waveform_generator):
        """Test that derivative scales correctly with step size."""
        param_to_vary = 'p0'
        
        # Test with different step sizes
        deltas = [1e-9, 1e-8, 1e-7]
        derivatives = []
        
        for delta in deltas:
            derivative = waveform_derivative(
                T=TEST_CONFIG['T'],
                dt=TEST_CONFIG['dt'],
                parameters=STANDARD_PARAMS,
                param_to_vary=param_to_vary,
                delta=delta,
                order=TEST_CONFIG['order'],
                kind=TEST_CONFIG['kind'],
            )
            derivatives.append(derivative)
        
        # Generate reference waveform for overlap calculation
        wave_ref = self.generate_waveform(base_waveform_generator, STANDARD_PARAMS)
        Npad = waveform_derivative.cache.get('Npad', 1)
        if Npad > 0:
            wave_ref = np.concatenate((wave_ref[:-Npad], np.zeros(Npad)), dtype=wave_ref.dtype)
        
        # Check that derivatives are consistent across step sizes
        overlap_01 = self.compute_overlap(derivatives[0], derivatives[1], wave_ref)
        overlap_12 = self.compute_overlap(derivatives[1], derivatives[2], wave_ref)
        
        # Should be very high overlap for nearby step sizes
        assert overlap_01 > 0.95, f"Overlap between step sizes too low: {overlap_01:.6f}"
        assert overlap_12 > 0.95, f"Overlap between step sizes too low: {overlap_12:.6f}"
        
        print(f"✓ Derivative scaling test passed")
    
    def test_different_parameters_give_different_derivatives(
        self, 
        waveform_derivative, 
        base_waveform_generator
    ):
        """Test that derivatives w.r.t. different parameters are actually different."""
        params_to_test = ['p0', 'e0', 'm2']
        derivatives = {}
        
        # Compute derivatives for different parameters
        for param in params_to_test:
            derivative = waveform_derivative(
                T=TEST_CONFIG['T'],
                dt=TEST_CONFIG['dt'],
                parameters=STANDARD_PARAMS,
                param_to_vary=param,
                delta=TEST_CONFIG['delta'],
                order=TEST_CONFIG['order'],
                kind=TEST_CONFIG['kind'],
            )
            derivatives[param] = derivative
        
        # Generate reference waveform
        wave_ref = self.generate_waveform(base_waveform_generator, STANDARD_PARAMS)
        Npad = waveform_derivative.cache.get('Npad', 1)
        if Npad > 0:
            wave_ref = np.concatenate((wave_ref[:-Npad], np.zeros(Npad)), dtype=wave_ref.dtype)
        
        # Check that derivatives are different from each other
        overlap_p0_e0 = self.compute_overlap(derivatives['p0'], derivatives['e0'], wave_ref)
        overlap_p0_m2 = self.compute_overlap(derivatives['p0'], derivatives['m2'], wave_ref)
        overlap_e0_m2 = self.compute_overlap(derivatives['e0'], derivatives['m2'], wave_ref)
        
        # Overlaps should be low (derivatives should be different)
        assert overlap_p0_e0 < 0.8, f"p0 and e0 derivatives too similar: {overlap_p0_e0:.6f}"
        assert overlap_p0_m2 < 0.8, f"p0 and m2 derivatives too similar: {overlap_p0_m2:.6f}"
        assert overlap_e0_m2 < 0.8, f"e0 and m2 derivatives too similar: {overlap_e0_m2:.6f}"
        
        print(f"✓ Different parameters give different derivatives")
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_cpu_gpu_consistency(self, base_waveform_generator):
        """Test that CPU and GPU give consistent results."""
        from few.waveform import FastSchwarzschildEccentricFlux
        
        # Create CPU and GPU derivative calculators
        cpu_calc = StableEMRIDerivative(
            waveform_class=FastSchwarzschildEccentricFlux,
            mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
            inspiral_kwargs=dict(err=1e-11, max_iter=10000),
            force_backend='cpu'
        )
        
        gpu_calc = StableEMRIDerivative(
            waveform_class=FastSchwarzschildEccentricFlux,
            mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
            inspiral_kwargs=dict(err=1e-11, max_iter=10000),
            force_backend='gpu'
        )
        
        param_to_vary = 'p0'
        
        # Compute derivatives on CPU and GPU
        cpu_derivative = cpu_calc(
            T=TEST_CONFIG['T'],
            dt=TEST_CONFIG['dt'],
            parameters=STANDARD_PARAMS,
            param_to_vary=param_to_vary,
            delta=TEST_CONFIG['delta'],
            order=TEST_CONFIG['order'],
            kind=TEST_CONFIG['kind'],
        )
        
        gpu_derivative = gpu_calc(
            T=TEST_CONFIG['T'],
            dt=TEST_CONFIG['dt'],
            parameters=STANDARD_PARAMS,
            param_to_vary=param_to_vary,
            delta=TEST_CONFIG['delta'],
            order=TEST_CONFIG['order'],
            kind=TEST_CONFIG['kind'],
        )
        
        # Convert GPU result to CPU for comparison
        if hasattr(gpu_derivative, 'get'):
            gpu_derivative = gpu_derivative.get()
        
        # Generate reference waveform
        wave_ref = self.generate_waveform(base_waveform_generator, STANDARD_PARAMS)
        Npad = cpu_calc.cache.get('Npad', 1)
        if Npad > 0:
            wave_ref = np.concatenate((wave_ref[:-Npad], np.zeros(Npad)), dtype=wave_ref.dtype)
        
        # Compute overlap
        overlap = self.compute_overlap(cpu_derivative, gpu_derivative, wave_ref)
        
        assert overlap > 0.999, f"CPU/GPU overlap too low: {overlap:.6f}"
        
        print(f"✓ CPU/GPU consistency test passed: overlap = {overlap:.6f}")


def test_import_dependencies():
    """Test that all required dependencies can be imported."""
    try:
        from stableemriderivative import StableEMRIDerivative
        from stableemrifisher.utils import inner_product, generate_PSD, padding
        from few.waveform import FastSchwarzschildEccentricFlux
        print("✓ All dependencies imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import required dependency: {e}")


if __name__ == "__main__":
    # Run tests directly for development
    pytest.main([__file__, "-v"])
