"""
Unit tests for the BaseModel abstract class.

Tests interface compliance, validation methods, and proper inheritance.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.base import BaseModel
from data import CalibrationData, CalibratorData, CalibrationResult


class ConcreteModel(BaseModel):
    """Concrete implementation for testing."""

    def fit(self, data: CalibrationData) -> None:
        """Minimal implementation for testing."""
        self._data = data
        self.fitted = True
        # Create dummy parameters
        n_freq = data.psd_frequencies.shape[0]
        self._parameters = {
            'u': jnp.ones(n_freq) * 0.1,
            'c': jnp.ones(n_freq) * 0.2,
            's': jnp.ones(n_freq) * 0.3,
            'NS': jnp.ones(n_freq) * 300.0,
            'L': jnp.ones(n_freq) * 0.95
        }

    def predict(self, frequencies: jax.Array, calibrator: str) -> jax.Array:
        """Return dummy predictions for testing."""
        self._validate_calibrator(calibrator)
        self._validate_frequencies(frequencies)
        # Return temperature based on calibrator
        temp_map = {'hot': 350.0, 'cold': 100.0, 'ant': 200.0}
        base_temp = temp_map.get(calibrator, 250.0)
        return jnp.full(frequencies.shape[0], base_temp)

    def get_parameters(self) -> Dict[str, jax.Array]:
        """Return fitted parameters."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting parameters")
        return self._parameters


@pytest.fixture
def sample_calibration_data():
    """Create sample calibration data for testing."""
    n_time = 10
    n_freq_psd = 100
    n_freq_vna = 50

    # Create calibrator data for hot, cold, and ant
    calibrators = {}
    for name in ['hot', 'cold', 'ant']:
        calibrators[name] = CalibratorData(
            name=name,
            psd_source=jnp.ones((n_time, n_freq_psd)),
            psd_load=jnp.ones((n_time, n_freq_psd)) * 0.9,
            psd_ns=jnp.ones((n_time, n_freq_psd)) * 1.1,
            s11_freq=jnp.linspace(50e6, 200e6, n_freq_vna),
            s11_complex=jnp.ones(n_freq_vna, dtype=complex) * 0.5,
            timestamps=jnp.ones((n_time, 2)),
            temperature=jnp.array({'hot': 350.0, 'cold': 100.0, 'ant': 200.0}[name])
        )

    return CalibrationData(
        calibrators=calibrators,
        psd_frequencies=jnp.linspace(0, 200e6, n_freq_psd),
        vna_frequencies=jnp.linspace(50e6, 200e6, n_freq_vna),
        metadata={'test': True}
    )


class TestBaseModel:
    """Test suite for BaseModel abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()

    def test_concrete_model_initialisation(self):
        """Test concrete model initialisation."""
        model = ConcreteModel()
        assert not model.fitted
        assert model.config == {}
        assert model._data is None
        assert model._parameters is None
        assert model._result is None

    def test_concrete_model_with_config(self):
        """Test model initialisation with config."""
        config = {'regularisation': 0.01, 'solver': 'lstsq'}
        model = ConcreteModel(config)
        assert model.config == config

    def test_fit_method(self, sample_calibration_data):
        """Test model fitting."""
        model = ConcreteModel()
        model.fit(sample_calibration_data)
        assert model.fitted
        assert model._data is not None
        assert model._parameters is not None

    def test_get_parameters_before_fit(self):
        """Test that get_parameters raises error before fitting."""
        model = ConcreteModel()
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.get_parameters()

    def test_get_parameters_after_fit(self, sample_calibration_data):
        """Test parameter extraction after fitting."""
        model = ConcreteModel()
        model.fit(sample_calibration_data)
        params = model.get_parameters()

        # Check all required parameters are present
        required = {'u', 'c', 's', 'NS', 'L'}
        assert set(params.keys()) == required

        # Check shapes
        n_freq = sample_calibration_data.psd_frequencies.shape[0]
        for param in params.values():
            assert param.shape == (n_freq,)

    def test_predict_before_fit(self):
        """Test that predict raises error before fitting."""
        model = ConcreteModel()
        frequencies = jnp.linspace(50e6, 200e6, 100)
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.predict(frequencies, 'hot')

    def test_predict_after_fit(self, sample_calibration_data):
        """Test prediction after fitting."""
        model = ConcreteModel()
        model.fit(sample_calibration_data)

        frequencies = jnp.linspace(50e6, 200e6, 50)
        pred = model.predict(frequencies, 'hot')

        assert pred.shape == frequencies.shape
        assert jnp.all(pred == 350.0)  # Based on ConcreteModel implementation

    def test_validate_calibrator(self, sample_calibration_data):
        """Test calibrator validation."""
        model = ConcreteModel()
        model.fit(sample_calibration_data)

        # Valid calibrator should not raise
        model._validate_calibrator('hot')

        # Invalid calibrator should raise KeyError
        with pytest.raises(KeyError, match="Calibrator 'invalid' not found"):
            model._validate_calibrator('invalid')

    def test_validate_calibrator_before_fit(self):
        """Test calibrator validation before fitting."""
        model = ConcreteModel()
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model._validate_calibrator('hot')

    def test_validate_frequencies(self):
        """Test frequency validation."""
        model = ConcreteModel()

        # Valid 1D array
        model._validate_frequencies(jnp.linspace(0, 200e6, 100))

        # Invalid: 2D array
        with pytest.raises(ValueError, match="Frequencies must be 1D array"):
            model._validate_frequencies(jnp.ones((10, 10)))

        # Invalid: negative frequencies
        with pytest.raises(ValueError, match="Frequencies must be positive"):
            model._validate_frequencies(jnp.array([-1.0, 0.0, 1.0]))

    def test_get_result_before_fit(self):
        """Test that get_result raises error before fitting."""
        model = ConcreteModel()
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.get_result()

    def test_get_result_after_fit(self, sample_calibration_data):
        """Test complete result generation."""
        model = ConcreteModel()
        model.fit(sample_calibration_data)
        result = model.get_result()

        # Check result type
        assert isinstance(result, CalibrationResult)

        # Check noise parameters
        assert set(result.noise_parameters.keys()) == {'u', 'c', 's', 'NS', 'L'}

        # Check predictions for all calibrators
        assert set(result.predicted_temperatures.keys()) == {'hot', 'cold', 'ant'}

        # Check residuals computed
        assert set(result.residuals.keys()) == {'hot', 'cold', 'ant'}

        # Check metadata
        assert result.model_name == 'ConcreteModel'
        assert 'config' in result.metadata
        assert 'n_calibrators' in result.metadata
        assert result.metadata['n_calibrators'] == 3

    def test_result_caching(self, sample_calibration_data):
        """Test that results are cached."""
        model = ConcreteModel()
        model.fit(sample_calibration_data)

        result1 = model.get_result()
        result2 = model.get_result()

        # Should be the same object (cached)
        assert result1 is result2

    def test_model_type_property(self):
        """Test model type property."""
        model = ConcreteModel()
        assert model.model_type == 'ConcreteModel'

    def test_repr(self, sample_calibration_data):
        """Test string representation."""
        model = ConcreteModel({'test': True})

        # Before fitting
        repr_str = repr(model)
        assert 'ConcreteModel' in repr_str
        assert 'not fitted' in repr_str
        assert 'test' in repr_str

        # After fitting
        model.fit(sample_calibration_data)
        repr_str = repr(model)
        assert 'fitted' in repr_str

    def test_abstract_methods_required(self):
        """Test that all abstract methods must be implemented."""
        # Missing fit method
        class IncompleteModel1(BaseModel):
            def predict(self, frequencies, calibrator):
                pass
            def get_parameters(self):
                pass

        with pytest.raises(TypeError):
            IncompleteModel1()

        # Missing predict method
        class IncompleteModel2(BaseModel):
            def fit(self, data):
                pass
            def get_parameters(self):
                pass

        with pytest.raises(TypeError):
            IncompleteModel2()

        # Missing get_parameters method
        class IncompleteModel3(BaseModel):
            def fit(self, data):
                pass
            def predict(self, frequencies, calibrator):
                pass

        with pytest.raises(TypeError):
            IncompleteModel3()