"""
Unit tests for the NeuralCorrectedLSQModel calibration model.

Tests the neural-corrected least squares implementation, verifying that:
- Least squares parameters match pure LSQ (analytical solution preserved)
- Neural network corrections are small on synthetic data (A ≈ 0)
- Two-stage fitting process works correctly
- Predictions combine linear and neural components
"""

import pytest
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import HDF5DataLoader, CalibrationData
from src.models.neural_corrected_lsq import NeuralCorrectedLSQModel
from src.models.least_squares import LeastSquaresModel


class TestNeuralCorrectedLSQModel:
    """Test suite for NeuralCorrectedLSQModel using test dataset."""

    @pytest.fixture
    def test_data_path(self):
        """Path to test data HDF5 file."""
        return Path('data/test_observation.hdf5')

    @pytest.fixture
    def loaded_data(self, test_data_path):
        """Load and prepare test data."""
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        loader = HDF5DataLoader()
        data = loader.load_observation(str(test_data_path))

        # Apply frequency mask (50-130 MHz)
        mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
        masked_data = loader.apply_frequency_mask(data, mask)

        return masked_data

    @pytest.fixture
    def filtered_data(self, loaded_data):
        """Filter to specific calibrators (like example script)."""
        calibrators_to_use = ['hot', 'cold', 'c10open', 'c10short', 'r100', 'ant']

        filtered_calibrators = {
            name: cal for name, cal in loaded_data.calibrators.items()
            if name in calibrators_to_use
        }

        return CalibrationData(
            calibrators=filtered_calibrators,
            psd_frequencies=loaded_data.psd_frequencies,
            vna_frequencies=loaded_data.vna_frequencies,
            lna_s11=loaded_data.lna_s11,
            metadata=loaded_data.metadata
        )

    def test_two_stage_fitting(self, filtered_data):
        """Test that two-stage fitting process completes successfully."""
        config = {
            'regularisation': 0.0,
            'n_iterations': 100,  # Reduced for faster testing
            'learning_rate': 1e-3,
            'correction_regularization': 0.01
        }
        model = NeuralCorrectedLSQModel(config)

        # Fit should complete without errors
        start_time = time.time()
        model.fit(filtered_data)
        fit_time = time.time() - start_time

        assert model.fitted, "Model should be fitted after calling fit()"
        print(f"\nTwo-stage fitting completed in {fit_time:.3f} seconds")

        # Get parameters
        params = model.get_parameters()

        # Verify all parameters are present and finite
        for key in ['u', 'c', 's', 'NS', 'L']:
            assert key in params, f"Parameter '{key}' not found"
            assert jnp.all(jnp.isfinite(params[key])), f"Parameter '{key}' contains non-finite values"

    def test_lsq_parameters_match_pure_lsq(self, filtered_data):
        """Test that LSQ parameters match pure least squares (analytical solution preserved)."""
        # Fit pure least squares model
        lsq_model = LeastSquaresModel({'regularisation': 0.0})
        lsq_model.fit(filtered_data)
        lsq_params = lsq_model.get_parameters()

        # Fit neural-corrected model
        neural_model = NeuralCorrectedLSQModel({
            'regularisation': 0.0,
            'n_iterations': 100
        })
        neural_model.fit(filtered_data)
        neural_params = neural_model.get_parameters()

        print("\nParameter Comparison (Pure LSQ vs Neural-Corrected LSQ):")
        print("-" * 60)

        # Compare parameters
        for key in ['u', 'c', 's', 'NS', 'L']:
            lsq_vals = lsq_params[key]
            neural_vals = neural_params[key]

            # Should be identical (or very close due to numerical precision)
            max_diff = float(jnp.max(jnp.abs(lsq_vals - neural_vals)))
            rel_diff = max_diff / (float(jnp.mean(jnp.abs(lsq_vals))) + 1e-10)

            print(f"{key:3s}: max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}")

            # Allow small numerical differences
            assert rel_diff < 1e-6, (
                f"LSQ parameter '{key}' differs between models (rel_diff={rel_diff:.6e})"
            )

    def test_corrections_small_on_synthetic_data(self, filtered_data):
        """Test that neural network corrections are small on synthetic data.

        For perfect synthetic data, corrections should be near zero since the
        physical model is sufficient.
        """
        config = {
            'regularisation': 0.0,
            'n_iterations': 500,
            'learning_rate': 1e-3,
            'correction_regularization': 0.01
        }
        model = NeuralCorrectedLSQModel(config)
        model.fit(filtered_data)

        # Get correction statistics
        correction_stats = model.get_correction_magnitude()

        print("\nNeural Network Correction Statistics:")
        print("-" * 50)
        print(f"Mean:  {correction_stats['mean']:7.4f} K")
        print(f"Std:   {correction_stats['std']:7.4f} K")
        print(f"RMS:   {correction_stats['rms']:7.4f} K")
        print(f"Range: [{correction_stats['min']:7.4f}, {correction_stats['max']:7.4f}] K")

        # For synthetic data, corrections should be small (< 1K RMS)
        assert correction_stats['rms'] < 1.0, (
            f"Corrections too large on synthetic data: RMS={correction_stats['rms']:.4f}K"
        )

    def test_predictions_combine_linear_and_neural(self, filtered_data):
        """Test that predictions correctly combine linear and neural components."""
        # Fit both models
        lsq_model = LeastSquaresModel({'regularisation': 0.0})
        lsq_model.fit(filtered_data)

        neural_model = NeuralCorrectedLSQModel({
            'regularisation': 0.0,
            'n_iterations': 500
        })
        neural_model.fit(filtered_data)

        # Get predictions for hot calibrator
        frequencies = filtered_data.psd_frequencies
        lsq_pred = lsq_model.predict(frequencies, 'hot')
        neural_pred = neural_model.predict(frequencies, 'hot')

        # Neural prediction should differ from pure LSQ
        # (unless corrections are exactly zero, which is unlikely)
        diff = neural_pred - lsq_pred
        mean_diff = float(jnp.mean(jnp.abs(diff)))

        print(f"\nPrediction difference (Neural vs Pure LSQ):")
        print(f"  Mean absolute difference: {mean_diff:.4f} K")

        # Difference should be small on synthetic data but non-zero
        # (unless regularization forced corrections to exactly zero)
        assert mean_diff < 10.0, "Neural corrections too large"

    def test_result_interface_compatibility(self, filtered_data):
        """Test that get_result() works and follows BaseModel interface."""
        model = NeuralCorrectedLSQModel({'n_iterations': 100})
        model.fit(filtered_data)

        # Get result
        result = model.get_result()

        # Check result structure
        assert hasattr(result, 'predicted_temperatures'), "Missing predicted_temperatures"
        assert hasattr(result, 'residuals'), "Missing residuals"
        assert hasattr(result, 'noise_parameters'), "Missing noise_parameters"
        assert hasattr(result, 'model_name'), "Missing model_name"
        assert hasattr(result, 'metadata'), "Missing metadata"

        # Check predictions for all calibrators
        for cal_name in filtered_data.calibrator_names:
            assert cal_name in result.predicted_temperatures, f"Missing predictions for {cal_name}"
            if cal_name != 'ant':  # Antenna excluded from fitting
                assert cal_name in result.residuals, f"Missing residuals for {cal_name}"

    def test_model_configurations(self, filtered_data):
        """Test different model configurations."""
        configs = [
            {'name': 'Standard', 'config': {'n_iterations': 200}},
            {'name': 'High Regularization', 'config': {'n_iterations': 200, 'correction_regularization': 0.1}},
            {'name': 'Deeper Network', 'config': {'n_iterations': 200, 'hidden_layers': [64, 64, 32]}},
        ]

        print("\nConfiguration Comparison:")
        print("-" * 80)

        for cfg in configs:
            try:
                model = NeuralCorrectedLSQModel(cfg['config'])
                start = time.time()
                model.fit(filtered_data)
                fit_time = time.time() - start

                params = model.get_parameters()
                u_mean = float(jnp.mean(params['u']))
                ns_mean = float(jnp.mean(params['NS']))

                correction_stats = model.get_correction_magnitude()
                correction_rms = correction_stats['rms']

                result = model.get_result()
                if 'hot' in result.residuals:
                    hot_rmse = float(jnp.sqrt(jnp.mean(result.residuals['hot']**2)))
                else:
                    hot_rmse = np.nan

                print(f"{cfg['name']:20s}: time={fit_time:.3f}s, "
                      f"u_mean={u_mean:7.2f}K, NS_mean={ns_mean:7.2f}K, "
                      f"corr_rms={correction_rms:.4f}K, hot_rmse={hot_rmse:.4f}K")

                # All configurations should produce reasonable results
                assert abs(u_mean) < 10000
                assert ns_mean > 0 and ns_mean < 50000
                assert correction_rms < 10.0  # Corrections should be reasonable

            except Exception as e:
                print(f"{cfg['name']:20s}: Failed - {str(e)[:40]}")
                raise

    def test_antenna_prediction_with_corrections(self, filtered_data):
        """Test antenna prediction includes neural corrections."""
        if 'ant' not in filtered_data.calibrators:
            pytest.skip("Antenna calibrator not present in test data")

        model = NeuralCorrectedLSQModel({'n_iterations': 500})
        model.fit(filtered_data)

        result = model.get_result()

        assert 'ant' in result.predicted_temperatures, "Antenna predictions should be present"
        T_ant = result.predicted_temperatures['ant']

        mean_T = float(jnp.mean(T_ant))
        std_T = float(jnp.std(T_ant))

        print(f"\nAntenna Temperature (with Neural Correction):")
        print(f"  Mean: {mean_T:.1f} K")
        print(f"  Std:  {std_T:.1f} K")

        # Check antenna temperature is reasonable
        assert mean_T > 1000, "Antenna temperature too low"
        assert mean_T < 10000, "Antenna temperature too high"

    def test_calibration_rmse_requirements(self, filtered_data):
        """Test that calibration achieves required RMSE levels.

        Should match or improve upon pure LSQ performance.
        """
        model = NeuralCorrectedLSQModel({
            'regularisation': 0.0,
            'n_iterations': 1000,
            'learning_rate': 1e-3,
            'correction_regularization': 0.01
        })

        model.fit(filtered_data)
        result = model.get_result()

        print("\nRMSE Results for Neural-Corrected LSQ:")
        print("-" * 50)

        calibration_sources = ['hot', 'cold', 'c10open', 'c10short', 'r100']

        for cal_name in calibration_sources:
            if cal_name in result.residuals:
                residuals = result.residuals[cal_name]
                rmse = float(jnp.sqrt(jnp.mean(residuals**2)))
                max_abs_residual = float(jnp.max(jnp.abs(residuals)))

                print(f"{cal_name:10s}: RMSE = {rmse:8.4f} K, max|res| = {max_abs_residual:8.4f} K")

                # Should still achieve good calibration on synthetic data
                # Allow slightly higher threshold since NN might not perfectly zero out
                assert rmse < 1.0, (
                    f"Calibration source '{cal_name}' RMSE {rmse:.6f}K exceeds 1.0K"
                )

        # Check antenna (if present)
        if 'ant' in result.residuals:
            ant_residuals = result.residuals['ant']
            ant_rmse = float(jnp.sqrt(jnp.mean(ant_residuals**2)))
            print(f"{'ant':10s}: RMSE = {ant_rmse:8.4f} K")
            assert ant_rmse < 20.0, f"Antenna RMSE {ant_rmse:.2f}K exceeds 20K"

    def test_config_retrieval(self, filtered_data):
        """Test that configuration can be retrieved."""
        config = {
            'regularisation': 0.0,
            'hidden_layers': [32, 32],
            'learning_rate': 1e-3,
            'n_iterations': 100,
            'correction_regularization': 0.01
        }

        model = NeuralCorrectedLSQModel(config)
        retrieved_config = model.get_config()

        # Check all config parameters are present
        assert retrieved_config['regularisation'] == config['regularisation']
        assert retrieved_config['hidden_layers'] == config['hidden_layers']
        assert retrieved_config['learning_rate'] == config['learning_rate']
        assert retrieved_config['n_iterations'] == config['n_iterations']
        assert retrieved_config['correction_regularization'] == config['correction_regularization']

    def test_prediction_shapes(self, filtered_data):
        """Test that predictions have correct shapes."""
        model = NeuralCorrectedLSQModel({'n_iterations': 100})
        model.fit(filtered_data)

        frequencies = filtered_data.psd_frequencies

        for cal_name in filtered_data.calibrator_names:
            prediction = model.predict(frequencies, cal_name)

            # Check shape matches frequencies
            assert prediction.shape == frequencies.shape, (
                f"Prediction shape {prediction.shape} doesn't match frequencies {frequencies.shape}"
            )

            # Check all values are finite
            assert jnp.all(jnp.isfinite(prediction)), (
                f"Prediction for '{cal_name}' contains non-finite values"
            )

    def test_model_not_fitted_error(self):
        """Test that calling predict before fit raises error."""
        model = NeuralCorrectedLSQModel()

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.predict(jnp.array([1e6, 2e6]), 'hot')

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.get_parameters()

        with pytest.raises(RuntimeError, match="Model must be fitted"):
            model.get_correction_magnitude()