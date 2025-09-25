"""
Unit tests for data module.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from src.data import (
    CalibratorData,
    CalibrationData,
    CalibrationResult,
    HDF5DataLoader
)


class TestCalibratorData:
    """Tests for CalibratorData class."""

    def test_valid_creation(self):
        """Test creating a valid CalibratorData object."""
        n_time, n_freq_psd, n_freq_vna = 10, 100, 50

        cal_data = CalibratorData(
            name="test_cal",
            psd_source=jnp.ones((n_time, n_freq_psd)),
            psd_load=jnp.ones((n_time, n_freq_psd)),
            psd_ns=jnp.ones((n_time, n_freq_psd)),
            s11_freq=jnp.linspace(50e6, 200e6, n_freq_vna),
            s11_complex=jnp.ones(n_freq_vna, dtype=complex),
            timestamps=jnp.ones((n_time, 2)),
            temperature=jnp.array(300.0)
        )

        assert cal_data.name == "test_cal"
        assert cal_data.n_time == n_time
        assert cal_data.n_freq_psd == n_freq_psd
        assert cal_data.n_freq_vna == n_freq_vna

    def test_psd_shape_mismatch(self):
        """Test that mismatched PSD shapes raise an error."""
        with pytest.raises(ValueError, match="PSD shapes must match"):
            CalibratorData(
                name="test",
                psd_source=jnp.ones((10, 100)),
                psd_load=jnp.ones((10, 100)),
                psd_ns=jnp.ones((10, 50)),  # Wrong freq dimension
                s11_freq=jnp.ones(50),
                s11_complex=jnp.ones(50, dtype=complex),
                timestamps=jnp.ones((10, 2))
            )

    def test_s11_shape_mismatch(self):
        """Test that mismatched S11 shapes raise an error."""
        with pytest.raises(ValueError, match="S11 frequency and complex"):
            CalibratorData(
                name="test",
                psd_source=jnp.ones((10, 100)),
                psd_load=jnp.ones((10, 100)),
                psd_ns=jnp.ones((10, 100)),
                s11_freq=jnp.ones(50),
                s11_complex=jnp.ones(60, dtype=complex),  # Wrong size
                timestamps=jnp.ones((10, 2))
            )

    def test_timestamp_shape_mismatch(self):
        """Test that mismatched timestamp shape raises an error."""
        with pytest.raises(ValueError, match="Timestamps must match"):
            CalibratorData(
                name="test",
                psd_source=jnp.ones((10, 100)),
                psd_load=jnp.ones((10, 100)),
                psd_ns=jnp.ones((10, 100)),
                s11_freq=jnp.ones(50),
                s11_complex=jnp.ones(50, dtype=complex),
                timestamps=jnp.ones((5, 2))  # Wrong time dimension
            )


class TestCalibrationData:
    """Tests for CalibrationData class."""

    def create_mock_calibrator(self, name: str, n_time: int = 10) -> CalibratorData:
        """Helper to create a mock calibrator."""
        return CalibratorData(
            name=name,
            psd_source=jnp.ones((n_time, 100)),
            psd_load=jnp.ones((n_time, 100)),
            psd_ns=jnp.ones((n_time, 100)),
            s11_freq=jnp.linspace(50e6, 200e6, 50),
            s11_complex=jnp.ones(50, dtype=complex),
            timestamps=jnp.ones((n_time, 2))
        )

    def test_valid_creation(self):
        """Test creating a valid CalibrationData object."""
        calibrators = {
            'hot': self.create_mock_calibrator('hot'),
            'cold': self.create_mock_calibrator('cold')
        }

        cal_data = CalibrationData(
            calibrators=calibrators,
            psd_frequencies=jnp.linspace(0, 200e6, 100),
            vna_frequencies=jnp.linspace(50e6, 200e6, 50)
        )

        assert cal_data.calibrator_names == ['hot', 'cold']
        assert cal_data.get_calibrator('hot').name == 'hot'

    def test_empty_calibrators(self):
        """Test that empty calibrators raise an error."""
        with pytest.raises(ValueError, match="at least one calibrator"):
            CalibrationData(
                calibrators={},
                psd_frequencies=jnp.ones(100),
                vna_frequencies=jnp.ones(50)
            )

    def test_inconsistent_psd_frequencies(self):
        """Test that inconsistent PSD frequencies raise an error."""
        calibrators = {
            'hot': self.create_mock_calibrator('hot'),
            'cold': self.create_mock_calibrator('cold')
        }

        with pytest.raises(ValueError, match="PSD frequency array size"):
            CalibrationData(
                calibrators=calibrators,
                psd_frequencies=jnp.ones(50),  # Wrong size
                vna_frequencies=jnp.ones(50)
            )

    def test_get_missing_calibrator(self):
        """Test that getting a missing calibrator raises an error."""
        calibrators = {'hot': self.create_mock_calibrator('hot')}

        cal_data = CalibrationData(
            calibrators=calibrators,
            psd_frequencies=jnp.linspace(0, 200e6, 100),
            vna_frequencies=jnp.linspace(50e6, 200e6, 50)
        )

        with pytest.raises(KeyError, match="Calibrator 'cold' not found"):
            cal_data.get_calibrator('cold')


class TestCalibrationResult:
    """Tests for CalibrationResult class."""

    def test_valid_creation(self):
        """Test creating a valid CalibrationResult object."""
        n_freq = 100

        result = CalibrationResult(
            noise_parameters={
                'u': jnp.ones(n_freq),
                'c': jnp.ones(n_freq),
                's': jnp.ones(n_freq),
                'NS': jnp.ones(n_freq),
                'L': jnp.ones(n_freq)
            },
            predicted_temperatures={'hot': jnp.ones(n_freq)},
            residuals={'hot': jnp.zeros(n_freq)},
            model_name='test_model'
        )

        assert result.model_name == 'test_model'
        assert 'u' in result.noise_parameters

    def test_missing_noise_parameters(self):
        """Test that missing noise parameters raise an error."""
        with pytest.raises(ValueError, match="Missing noise parameters"):
            CalibrationResult(
                noise_parameters={
                    'u': jnp.ones(100),
                    'c': jnp.ones(100)
                    # Missing s, NS, L
                },
                predicted_temperatures={},
                residuals={},
                model_name='test'
            )


class TestHDF5DataLoader:
    """Tests for HDF5DataLoader class."""

    def test_loader_creation(self):
        """Test creating a loader instance."""
        loader = HDF5DataLoader()
        assert loader.device is None

        loader_gpu = HDF5DataLoader(device='cpu')
        assert loader_gpu.device == 'cpu'

    def test_load_observation(self):
        """Test loading observation from HDF5 file."""
        # Check if the test data file exists
        test_file = Path('data/reach_observation.hdf5')
        if not test_file.exists():
            pytest.skip("Test data file not found")

        loader = HDF5DataLoader()
        data = loader.load_observation(str(test_file))

        # Check basic structure
        assert isinstance(data, CalibrationData)
        assert len(data.calibrators) > 0
        assert data.psd_frequencies is not None
        assert data.vna_frequencies is not None

        # Check a specific calibrator
        if 'hot' in data.calibrator_names:
            hot_cal = data.get_calibrator('hot')
            assert hot_cal.name == 'hot'
            assert hot_cal.psd_source.ndim == 2
            assert np.iscomplexobj(hot_cal.s11_complex)

    def test_missing_file(self):
        """Test that missing file raises appropriate error."""
        loader = HDF5DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_observation('nonexistent_file.hdf5')

    def test_apply_frequency_mask(self):
        """Test applying frequency mask to data."""
        # Create mock data
        calibrators = {
            'hot': CalibratorData(
                name='hot',
                psd_source=jnp.ones((10, 200)),
                psd_load=jnp.ones((10, 200)),
                psd_ns=jnp.ones((10, 200)),
                s11_freq=jnp.linspace(50e6, 200e6, 100),
                s11_complex=jnp.ones(100, dtype=complex),
                timestamps=jnp.ones((10, 2))
            )
        }

        data = CalibrationData(
            calibrators=calibrators,
            psd_frequencies=jnp.linspace(0, 200e6, 200),
            vna_frequencies=jnp.linspace(50e6, 200e6, 100)
        )

        # Apply mask
        loader = HDF5DataLoader()
        mask = data.vna_frequencies < 130e6
        masked_data = loader.apply_frequency_mask(data, mask)

        # Check that frequencies are reduced
        assert masked_data.vna_frequencies.shape[0] < data.vna_frequencies.shape[0]
        assert jnp.all(masked_data.vna_frequencies < 130e6)

    def test_save_results(self, tmp_path):
        """Test saving results to HDF5."""
        loader = HDF5DataLoader()

        result = CalibrationResult(
            noise_parameters={
                'u': jnp.ones(100),
                'c': jnp.ones(100),
                's': jnp.ones(100),
                'NS': jnp.ones(100),
                'L': jnp.ones(100)
            },
            predicted_temperatures={'hot': jnp.ones(100) * 300},
            residuals={'hot': jnp.zeros(100)},
            model_name='test_model',
            metadata={'test_key': 'test_value'}
        )

        output_file = tmp_path / 'test_results.hdf5'
        loader.save_results(str(output_file), result)

        # Check file was created
        assert output_file.exists()

        # Verify contents
        import h5py
        with h5py.File(output_file, 'r') as f:
            assert 'noise_parameters/u' in f
            assert 'predicted_temperatures/hot' in f
            assert 'residuals/hot' in f
            assert f.attrs['model_name'] == 'test_model'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])