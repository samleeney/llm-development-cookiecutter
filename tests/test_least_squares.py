"""
Unit tests for the LeastSquaresModel calibration model.

Tests the least squares implementation using the same structure as
examples/least_squares_calibration.py, verifying RMSE requirements:
- Calibration sources (hot, cold, c10open, c10short, r100): RMSE < 0.001K
- Antenna: RMSE < 15K
"""

import pytest
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import HDF5DataLoader
from src.models.least_squares import LeastSquaresModel


class TestLeastSquaresModel:
    """Test suite for LeastSquaresModel using test dataset."""

    @pytest.fixture
    def test_data_path(self):
        """Path to test data HDF5 file."""
        return Path('data/test_observation.hdf5')

    @pytest.fixture
    def loaded_data(self, test_data_path):
        """Load and prepare test data."""
        # Check if test data exists
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        # Load data (same as example)
        loader = HDF5DataLoader()
        data = loader.load_observation(str(test_data_path))

        # Apply frequency mask (50-130 MHz)
        mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
        masked_data = loader.apply_frequency_mask(data, mask)

        return masked_data

    def test_calibration_with_rmse_requirements(self, loaded_data):
        """Test calibration achieves required RMSE levels.

        Uses only 5 specific calibrators for fitting:
        - Calibration sources (hot, cold, c10open, c10short, r100): RMSE < 0.01K
        - Antenna: RMSE < 15K
        """
        # Filter to only use specific calibrators (like run_limited_calibration.py)
        calibrators_to_use = ['hot', 'cold', 'c10open', 'c10short', 'r100', 'ant']

        # Filter the data to only include these calibrators
        from src.data import CalibrationData
        filtered_calibrators = {
            name: cal for name, cal in loaded_data.calibrators.items()
            if name in calibrators_to_use
        }

        filtered_data = CalibrationData(
            calibrators=filtered_calibrators,
            psd_frequencies=loaded_data.psd_frequencies,
            vna_frequencies=loaded_data.vna_frequencies,
            lna_s11=loaded_data.lna_s11,
            metadata=loaded_data.metadata
        )

        print(f"\nUsing {len(filtered_calibrators)} specific calibrators for fitting")

        # Create and configure model (same as example)
        config = {
            'regularisation': 0.0,
            'use_gamma_weighting': False
        }
        model = LeastSquaresModel(config)

        # Fit the model to filtered calibrators
        start_time = time.time()
        model.fit(filtered_data)
        fit_time = time.time() - start_time

        assert model.fitted, "Model should be fitted after calling fit()"
        print(f"Fitting completed in {fit_time:.3f} seconds")

        # Get fitted parameters
        params = model.get_parameters()

        # Verify all parameters are present and finite
        for key in ['u', 'c', 's', 'NS', 'L']:
            assert key in params, f"Parameter '{key}' not found"
            assert jnp.all(jnp.isfinite(params[key])), f"Parameter '{key}' contains non-finite values"

        # Get result with predictions and residuals
        result = model.get_result()

        # Calculate and verify RMSE for specific calibrators we care about
        print("\nRMSE Results for Key Calibrators:")
        print("-" * 50)

        # Test specific calibrators mentioned in requirements
        calibrators_to_test = ['hot', 'cold', 'c10open', 'c10short', 'r100', 'ant']
        calibration_sources = ['hot', 'cold', 'c10open', 'c10short', 'r100']

        for cal_name in calibrators_to_test:
            if cal_name in result.residuals:
                residuals = result.residuals[cal_name]
                rmse = float(jnp.sqrt(jnp.mean(residuals**2)))
                max_abs_residual = float(jnp.max(jnp.abs(residuals)))

                print(f"{cal_name:10s}: RMSE = {rmse:8.4f} K, max|res| = {max_abs_residual:8.4f} K")

                # Apply RMSE requirements (based on earlier successful run)
                if cal_name in calibration_sources:
                    # Calibration sources should have RMSE < 0.001K for synthetic data
                    assert rmse < 0.001, (
                        f"Calibration source '{cal_name}' RMSE {rmse:.6f}K exceeds 0.001K requirement"
                    )
                elif cal_name == 'ant':
                    # Antenna should have RMSE < 15K
                    assert rmse < 15.0, (
                        f"Antenna RMSE {rmse:.2f}K exceeds 15K requirement"
                    )

    def test_noise_parameters_reasonable(self, loaded_data):
        """Test that fitted noise parameters are physically reasonable."""
        # Create and fit model
        config = {'regularisation': 0.0, 'use_gamma_weighting': False}
        model = LeastSquaresModel(config)
        model.fit(loaded_data)

        # Get parameters
        params = model.get_parameters()

        # Check uncorrelated noise is positive and reasonable
        assert jnp.all(params['u'] > 0), "Uncorrelated noise should be positive"
        assert jnp.all(params['u'] < 10000), "Uncorrelated noise unreasonably high"

        # Check correlated noise components are reasonable
        for key in ['c', 's']:
            assert jnp.all(jnp.abs(params[key]) < 10000), f"Parameter '{key}' unreasonably high"

        # Check noise source temperature is positive and reasonable
        assert jnp.all(params['NS'] > 0), "Noise source temperature should be positive"
        assert jnp.all(params['NS'] < 50000), "Noise source temperature unreasonably high"

        # Check load temperature is reasonable
        assert jnp.all(params['L'] > 0), "Load temperature should be positive"
        assert jnp.all(params['L'] < 10000), "Load temperature unreasonably high"

        print("\nNoise Parameter Ranges:")
        print("-" * 50)
        for key in ['u', 'c', 's', 'NS', 'L']:
            values = params[key]
            print(f"{key:3s}: [{float(jnp.min(values)):7.2f}, {float(jnp.max(values)):7.2f}] K")

    def test_antenna_temperature_prediction(self, loaded_data):
        """Test antenna temperature prediction is reasonable."""
        # Only test if antenna is present
        if 'ant' not in loaded_data.calibrators:
            pytest.skip("Antenna calibrator not present in test data")

        # Fit model
        config = {'regularisation': 0.0, 'use_gamma_weighting': False}
        model = LeastSquaresModel(config)
        model.fit(loaded_data)

        # Get predictions
        result = model.get_result()

        assert 'ant' in result.predicted_temperatures, "Antenna predictions should be present"
        T_ant = result.predicted_temperatures['ant']

        # Check antenna temperature is reasonable (around 5000K for test data)
        mean_T = float(jnp.mean(T_ant))
        std_T = float(jnp.std(T_ant))

        print(f"\nAntenna Temperature: mean={mean_T:.1f}K, std={std_T:.1f}K")

        assert mean_T > 1000, "Antenna temperature too low"
        assert mean_T < 10000, "Antenna temperature too high"
        assert std_T < 1000, "Antenna temperature variation too high"

    def test_model_configurations(self, loaded_data):
        """Test different model configurations (like in example)."""
        configs = [
            {'name': 'Standard', 'config': {}},
            {'name': 'With Regularisation', 'config': {'regularisation': 0.001}},
            {'name': 'Gamma Weighted', 'config': {'use_gamma_weighting': True}},
        ]

        print("\nConfiguration Comparison:")
        print("-" * 60)

        for cfg in configs:
            try:
                model = LeastSquaresModel(cfg['config'])
                start = time.time()
                model.fit(loaded_data)
                fit_time = time.time() - start

                params = model.get_parameters()
                u_mean = float(jnp.mean(params['u']))
                ns_mean = float(jnp.mean(params['NS']))

                # Get RMSE for hot calibrator as reference
                result = model.get_result()
                if 'hot' in result.residuals:
                    hot_rmse = float(jnp.sqrt(jnp.mean(result.residuals['hot']**2)))
                else:
                    hot_rmse = np.nan

                print(f"{cfg['name']:20s}: time={fit_time:.3f}s, "
                      f"u_mean={u_mean:7.2f}K, NS_mean={ns_mean:7.2f}K, "
                      f"hot_rmse={hot_rmse:.4f}K")

                # All configurations should produce reasonable results
                assert abs(u_mean) < 10000  # Allow negative values for some configurations
                assert ns_mean > 0 and ns_mean < 50000
                if not np.isnan(hot_rmse):
                    assert hot_rmse < 20.0  # Should be reasonable for calibration sources

            except Exception as e:
                print(f"{cfg['name']:20s}: Failed - {str(e)[:40]}")
                # Re-raise to fail the test
                raise

    def test_perfect_calibration_on_synthetic_data(self, loaded_data):
        """Test that calibration achieves near-perfect fit on synthetic data.

        The test_observation.hdf5 contains synthetic data generated from known
        parameters, so we should achieve near-perfect calibration.
        """
        # Fit model
        config = {'regularisation': 0.0, 'use_gamma_weighting': False}
        model = LeastSquaresModel(config)
        model.fit(loaded_data)

        # Get result
        result = model.get_result()

        # For synthetic data, all calibration sources should have extremely low RMSE
        calibration_sources = ['hot', 'cold', 'c10open', 'c10short', 'r100']

        print("\nSynthetic Data Calibration Quality:")
        print("-" * 50)

        for cal_name in calibration_sources:
            if cal_name in result.residuals:
                residuals = result.residuals[cal_name]
                rmse = float(jnp.sqrt(jnp.mean(residuals**2)))

                # For synthetic data, RMSE should be reasonably low (< 1.0K)
                print(f"{cal_name:10s}: RMSE = {rmse:.6f} K")
                assert rmse < 1.0, (
                    f"Synthetic data calibration for '{cal_name}' has RMSE {rmse:.6f}K, "
                    f"expected < 1.0K"
                )

    def test_frequency_masking(self, test_data_path):
        """Test that frequency masking works correctly."""
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        loader = HDF5DataLoader()
        data = loader.load_observation(str(test_data_path))

        # Test different frequency masks
        mask1 = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
        masked1 = loader.apply_frequency_mask(data, mask1)

        mask2 = (data.vna_frequencies >= 60e6) & (data.vna_frequencies <= 120e6)
        masked2 = loader.apply_frequency_mask(data, mask2)

        # Check that masking reduces data size correctly
        assert len(masked1.vna_frequencies) < len(data.vna_frequencies)
        assert len(masked2.vna_frequencies) < len(masked1.vna_frequencies)

        # Fit model to different masks and verify both work
        model = LeastSquaresModel()

        model.fit(masked1)
        result1 = model.get_result()

        model.fit(masked2)
        result2 = model.get_result()

        # Both should produce valid results
        assert 'hot' in result1.residuals
        assert 'hot' in result2.residuals