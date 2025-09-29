"""
Unit tests for the LeastSquaresModel calibration model.

Tests the least squares implementation including X matrix construction,
parameter fitting, prediction, and JAX optimisations.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.least_squares import LeastSquaresModel
from data import CalibrationData, CalibratorData


class TestLeastSquaresModel:
    """Test suite for LeastSquaresModel."""

    @pytest.fixture
    def sample_calibration_data(self):
        """Create sample calibration data for testing."""
        n_time = 5
        n_freq_psd = 100
        n_freq_vna = 50

        # Create frequency arrays
        psd_freq = jnp.linspace(0, 200e6, n_freq_psd)
        vna_freq = jnp.linspace(50e6, 200e6, n_freq_vna)

        # Create calibrators with different characteristics
        calibrators = {}

        # Hot calibrator - high temperature
        calibrators['hot'] = CalibratorData(
            name='hot',
            psd_source=jnp.ones((n_time, n_freq_psd)) * 1.5 +
                      jax.random.normal(jax.random.PRNGKey(0), (n_time, n_freq_psd)) * 0.01,
            psd_load=jnp.ones((n_time, n_freq_psd)) * 1.2,
            psd_ns=jnp.ones((n_time, n_freq_psd)) * 1.8,
            s11_freq=vna_freq,
            s11_complex=jnp.ones(n_freq_vna, dtype=complex) * (0.3 + 0.1j),
            timestamps=jnp.ones((n_time, 2)),
            temperature=jnp.array(350.0)
        )

        # Cold calibrator - low temperature
        calibrators['cold'] = CalibratorData(
            name='cold',
            psd_source=jnp.ones((n_time, n_freq_psd)) * 0.8,
            psd_load=jnp.ones((n_time, n_freq_psd)) * 0.9,
            psd_ns=jnp.ones((n_time, n_freq_psd)) * 1.1,
            s11_freq=vna_freq,
            s11_complex=jnp.ones(n_freq_vna, dtype=complex) * (0.2 - 0.05j),
            timestamps=jnp.ones((n_time, 2)),
            temperature=jnp.array(100.0)
        )

        # Antenna calibrator - intermediate temperature
        calibrators['ant'] = CalibratorData(
            name='ant',
            psd_source=jnp.ones((n_time, n_freq_psd)) * 1.1,
            psd_load=jnp.ones((n_time, n_freq_psd)) * 1.0,
            psd_ns=jnp.ones((n_time, n_freq_psd)) * 1.3,
            s11_freq=vna_freq,
            s11_complex=jnp.ones(n_freq_vna, dtype=complex) * (0.4 + 0.2j),
            timestamps=jnp.ones((n_time, 2)),
            temperature=jnp.array(250.0)
        )

        return CalibrationData(
            calibrators=calibrators,
            psd_frequencies=psd_freq,
            vna_frequencies=vna_freq,
            metadata={'test': True}
        )

    @pytest.fixture
    def minimal_calibration_data(self):
        """Create minimal calibration data with just hot and cold."""
        n_time = 3
        n_freq_psd = 50
        n_freq_vna = 25

        psd_freq = jnp.linspace(50e6, 150e6, n_freq_psd)
        vna_freq = jnp.linspace(50e6, 150e6, n_freq_vna)

        calibrators = {}

        # Hot calibrator
        calibrators['hot'] = CalibratorData(
            name='hot',
            psd_source=jnp.ones((n_time, n_freq_psd)) * 2.0,
            psd_load=jnp.ones((n_time, n_freq_psd)) * 1.5,
            psd_ns=jnp.ones((n_time, n_freq_psd)) * 2.5,
            s11_freq=vna_freq,
            s11_complex=jnp.ones(n_freq_vna, dtype=complex) * 0.3,
            timestamps=jnp.ones((n_time, 2)),
            temperature=jnp.array(400.0)
        )

        # Cold calibrator
        calibrators['cold'] = CalibratorData(
            name='cold',
            psd_source=jnp.ones((n_time, n_freq_psd)) * 0.5,
            psd_load=jnp.ones((n_time, n_freq_psd)) * 0.6,
            psd_ns=jnp.ones((n_time, n_freq_psd)) * 0.8,
            s11_freq=vna_freq,
            s11_complex=jnp.ones(n_freq_vna, dtype=complex) * 0.1,
            timestamps=jnp.ones((n_time, 2)),
            temperature=jnp.array(77.0)
        )

        return CalibrationData(
            calibrators=calibrators,
            psd_frequencies=psd_freq,
            vna_frequencies=vna_freq,
            metadata={'minimal': True}
        )

    def test_model_initialisation(self):
        """Test model initialisation with default config."""
        model = LeastSquaresModel()
        assert not model.fitted
        assert model.regularisation == 0.0
        assert not model.use_gamma_weighting

    def test_model_initialisation_with_config(self):
        """Test model initialisation with custom config."""
        config = {
            'regularisation': 0.001,
            'use_gamma_weighting': True
        }
        model = LeastSquaresModel(config)
        assert model.regularisation == 0.001
        assert model.use_gamma_weighting

    def test_fit_with_minimal_data(self, minimal_calibration_data):
        """Test fitting with minimal calibration data."""
        model = LeastSquaresModel()
        model.fit(minimal_calibration_data)

        assert model.fitted
        assert model._data is not None
        assert model._parameters is not None

        # Check parameter shapes
        params = model.get_parameters()
        n_freq = len(minimal_calibration_data.psd_frequencies)
        for key in ['u', 'c', 's', 'NS', 'L']:
            assert key in params
            assert params[key].shape == (n_freq,)

    def test_fit_with_full_data(self, sample_calibration_data):
        """Test fitting with full calibration data."""
        model = LeastSquaresModel()
        model.fit(sample_calibration_data)

        assert model.fitted
        params = model.get_parameters()

        # Check all parameters are present and have correct shape
        n_freq = len(sample_calibration_data.psd_frequencies)
        for key in ['u', 'c', 's', 'NS', 'L']:
            assert key in params
            assert params[key].shape == (n_freq,)
            # Check parameters are finite
            assert jnp.all(jnp.isfinite(params[key]))

    def test_fit_missing_required_calibrator(self, sample_calibration_data):
        """Test that fitting fails without required calibrators."""
        # Remove 'cold' calibrator
        del sample_calibration_data.calibrators['cold']

        model = LeastSquaresModel()
        with pytest.raises(ValueError, match="Required calibrator 'cold'"):
            model.fit(sample_calibration_data)

    def test_predict_before_fitting(self, sample_calibration_data):
        """Test that prediction fails before fitting."""
        model = LeastSquaresModel()
        freq = sample_calibration_data.psd_frequencies

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.predict(freq, 'hot')

    def test_predict_after_fitting(self, sample_calibration_data):
        """Test prediction after fitting."""
        model = LeastSquaresModel()
        model.fit(sample_calibration_data)

        # Predict for hot calibrator
        freq = sample_calibration_data.psd_frequencies
        T_pred = model.predict(freq, 'hot')

        assert T_pred.shape == freq.shape
        assert jnp.all(jnp.isfinite(T_pred))
        # Temperature should be reasonable
        assert jnp.all(T_pred > 0)
        assert jnp.all(T_pred < 1000)

    def test_predict_invalid_calibrator(self, sample_calibration_data):
        """Test prediction with invalid calibrator name."""
        model = LeastSquaresModel()
        model.fit(sample_calibration_data)

        freq = sample_calibration_data.psd_frequencies
        with pytest.raises(KeyError, match="Calibrator 'invalid'"):
            model.predict(freq, 'invalid')

    def test_predict_with_interpolation(self, sample_calibration_data):
        """Test prediction at different frequency points."""
        model = LeastSquaresModel()
        model.fit(sample_calibration_data)

        # Predict at different frequencies
        new_freq = jnp.linspace(20e6, 180e6, 80)
        T_pred = model.predict(new_freq, 'hot')

        assert T_pred.shape == new_freq.shape
        assert jnp.all(jnp.isfinite(T_pred))

    def test_get_parameters_before_fitting(self):
        """Test getting parameters before fitting."""
        model = LeastSquaresModel()
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.get_parameters()

    def test_regularisation(self, minimal_calibration_data):
        """Test fitting with regularisation."""
        # Fit without regularisation
        model1 = LeastSquaresModel({'regularisation': 0.0})
        model1.fit(minimal_calibration_data)
        params1 = model1.get_parameters()

        # Fit with regularisation
        model2 = LeastSquaresModel({'regularisation': 0.1})
        model2.fit(minimal_calibration_data)
        params2 = model2.get_parameters()

        # Parameters should be different with regularisation
        for key in params1:
            assert not jnp.allclose(params1[key], params2[key])

    def test_gamma_weighting(self, minimal_calibration_data):
        """Test fitting with gamma weighting."""
        # Fit without gamma weighting
        model1 = LeastSquaresModel({'use_gamma_weighting': False})
        model1.fit(minimal_calibration_data)
        params1 = model1.get_parameters()

        # Fit with gamma weighting
        model2 = LeastSquaresModel({'use_gamma_weighting': True})
        model2.fit(minimal_calibration_data)
        params2 = model2.get_parameters()

        # Parameters should be different with gamma weighting
        for key in params1:
            # Gamma weighting should produce different results
            assert not jnp.allclose(params1[key], params2[key], rtol=1e-2)

    def test_get_result(self, sample_calibration_data):
        """Test getting complete calibration result."""
        model = LeastSquaresModel()
        model.fit(sample_calibration_data)

        result = model.get_result()

        assert result.model_name == 'LeastSquaresModel'
        assert result.noise_parameters is not None
        assert result.predicted_temperatures is not None
        assert result.residuals is not None

        # Check predictions exist for all calibrators
        for cal_name in sample_calibration_data.calibrator_names:
            assert cal_name in result.predicted_temperatures
            assert cal_name in result.residuals

    def test_result_caching(self, sample_calibration_data):
        """Test that results are cached."""
        model = LeastSquaresModel()
        model.fit(sample_calibration_data)

        result1 = model.get_result()
        result2 = model.get_result()

        # Should be the same object (cached)
        assert result1 is result2

    def test_jit_compilation(self, minimal_calibration_data):
        """Test that JIT compilation works correctly."""
        model = LeastSquaresModel()

        # First fit (includes compilation)
        model.fit(minimal_calibration_data)
        params1 = model.get_parameters()

        # Second fit with same data structure should be faster (already compiled)
        model2 = LeastSquaresModel()
        model2.fit(minimal_calibration_data)
        params2 = model2.get_parameters()

        # Results should be identical
        for key in params1:
            assert jnp.allclose(params1[key], params2[key])

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        n_time = 3
        n_freq = 20

        # Create data with very small values
        calibrators = {}
        for name, temp in [('hot', 1e6), ('cold', 1e-6)]:
            calibrators[name] = CalibratorData(
                name=name,
                psd_source=jnp.ones((n_time, n_freq)) * 1e-10,
                psd_load=jnp.ones((n_time, n_freq)) * 1e-10,
                psd_ns=jnp.ones((n_time, n_freq)) * 2e-10,
                s11_freq=jnp.linspace(50e6, 150e6, n_freq),
                s11_complex=jnp.ones(n_freq, dtype=complex) * 0.01,
                timestamps=jnp.ones((n_time, 2)),
                temperature=jnp.array(temp)
            )

        data = CalibrationData(
            calibrators=calibrators,
            psd_frequencies=jnp.linspace(50e6, 150e6, n_freq),
            vna_frequencies=jnp.linspace(50e6, 150e6, n_freq)
        )

        # Should handle extreme values with regularisation
        model = LeastSquaresModel({'regularisation': 1e-6})
        model.fit(data)
        params = model.get_parameters()

        # Check all parameters are finite
        for key, values in params.items():
            assert jnp.all(jnp.isfinite(values)), f"Non-finite values in {key}"

    def test_frequency_dependent_temperature(self):
        """Test with frequency-dependent calibrator temperatures."""
        n_time = 3
        n_freq = 30

        # Create frequency-dependent temperature
        temp_array = 300.0 + jnp.sin(jnp.linspace(0, 2*jnp.pi, n_freq)) * 50

        calibrators = {
            'hot': CalibratorData(
                name='hot',
                psd_source=jnp.ones((n_time, n_freq)) * 1.5,
                psd_load=jnp.ones((n_time, n_freq)) * 1.2,
                psd_ns=jnp.ones((n_time, n_freq)) * 1.8,
                s11_freq=jnp.linspace(50e6, 150e6, n_freq),
                s11_complex=jnp.ones(n_freq, dtype=complex) * 0.3,
                timestamps=jnp.ones((n_time, 2)),
                temperature=temp_array  # Frequency-dependent
            ),
            'cold': CalibratorData(
                name='cold',
                psd_source=jnp.ones((n_time, n_freq)) * 0.5,
                psd_load=jnp.ones((n_time, n_freq)) * 0.6,
                psd_ns=jnp.ones((n_time, n_freq)) * 0.8,
                s11_freq=jnp.linspace(50e6, 150e6, n_freq),
                s11_complex=jnp.ones(n_freq, dtype=complex) * 0.1,
                timestamps=jnp.ones((n_time, 2)),
                temperature=jnp.array(100.0)  # Scalar
            )
        }

        data = CalibrationData(
            calibrators=calibrators,
            psd_frequencies=jnp.linspace(50e6, 150e6, n_freq),
            vna_frequencies=jnp.linspace(50e6, 150e6, n_freq)
        )

        model = LeastSquaresModel()
        model.fit(data)

        # Should handle mixed scalar and array temperatures
        assert model.fitted
        params = model.get_parameters()
        assert all(jnp.all(jnp.isfinite(params[key])) for key in params)