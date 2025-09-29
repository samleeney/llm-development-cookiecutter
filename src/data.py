"""
Data infrastructure for JAX-based radiometer calibration pipeline.

This module provides data structures and utilities for loading, processing,
and managing radiometer calibration data from HDF5 files.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import h5py


@dataclass
class CalibratorData:
    """
    Container for single calibrator measurements.

    Attributes:
        name: Calibrator identifier (e.g., 'hot', 'cold', 'ant')
        psd_source: Power spectral density with source connected [n_time, n_freq]
        psd_load: Power spectral density with load connected [n_time, n_freq]
        psd_ns: Power spectral density with noise source [n_time, n_freq]
        s11_freq: VNA frequency points [n_freq_vna,]
        s11_complex: Complex S11 parameters [n_freq_vna,]
        timestamps: Measurement timestamps [n_time, 2] (start, end)
        temperature: Physical temperature of calibrator in Kelvin
    """
    name: str
    psd_source: jax.Array
    psd_load: jax.Array
    psd_ns: jax.Array
    s11_freq: jax.Array
    s11_complex: jax.Array
    timestamps: jax.Array
    temperature: Optional[jax.Array] = None

    def __post_init__(self):
        """Validate data consistency after initialisation."""
        # Check PSD shapes match
        if not (self.psd_source.shape == self.psd_load.shape == self.psd_ns.shape):
            raise ValueError(
                f"PSD shapes must match. Got source: {self.psd_source.shape}, "
                f"load: {self.psd_load.shape}, ns: {self.psd_ns.shape}"
            )

        # Check S11 shapes match
        if self.s11_freq.shape != self.s11_complex.shape:
            raise ValueError(
                f"S11 frequency and complex arrays must have same shape. "
                f"Got freq: {self.s11_freq.shape}, complex: {self.s11_complex.shape}"
            )

        # Check timestamps shape
        n_time = self.psd_source.shape[0]
        if self.timestamps.shape[0] != n_time:
            raise ValueError(
                f"Timestamps must match time dimension of PSD data. "
                f"Expected {n_time}, got {self.timestamps.shape[0]}"
            )

    @property
    def n_time(self) -> int:
        """Number of time samples."""
        return self.psd_source.shape[0]

    @property
    def n_freq_psd(self) -> int:
        """Number of PSD frequency channels."""
        return self.psd_source.shape[1]

    @property
    def n_freq_vna(self) -> int:
        """Number of VNA frequency points."""
        return self.s11_freq.shape[0]


@dataclass
class CalibrationData:
    """
    Complete calibration dataset containing all calibrators and metadata.

    Attributes:
        calibrators: Dictionary of calibrator measurements
        psd_frequencies: Frequency array for PSD data [Hz]
        vna_frequencies: Frequency array for VNA data [Hz]
        metadata: Configuration and observation metadata
        temperatures: Temperature sensor readings [n_time, n_sensors]
        temperature_timestamps: Temperature measurement timestamps
    """
    calibrators: Dict[str, CalibratorData]
    psd_frequencies: jax.Array
    vna_frequencies: jax.Array
    metadata: Dict[str, Any] = field(default_factory=dict)
    temperatures: Optional[jax.Array] = None
    temperature_timestamps: Optional[jax.Array] = None

    def __post_init__(self):
        """Validate data consistency."""
        if not self.calibrators:
            raise ValueError("CalibrationData must contain at least one calibrator")

        # Check all calibrators have consistent PSD frequencies
        n_freq_psd = list(self.calibrators.values())[0].n_freq_psd
        for cal in self.calibrators.values():
            if cal.n_freq_psd != n_freq_psd:
                raise ValueError(
                    f"All calibrators must have same PSD frequency dimension. "
                    f"Expected {n_freq_psd}, got {cal.n_freq_psd} for {cal.name}"
                )

        # Check frequency array dimensions
        if self.psd_frequencies.shape[0] != n_freq_psd:
            raise ValueError(
                f"PSD frequency array size {self.psd_frequencies.shape[0]} "
                f"doesn't match data dimension {n_freq_psd}"
            )

    @property
    def calibrator_names(self) -> List[str]:
        """List of available calibrator names."""
        return list(self.calibrators.keys())

    def get_calibrator(self, name: str) -> CalibratorData:
        """Get calibrator by name with error checking."""
        if name not in self.calibrators:
            raise KeyError(
                f"Calibrator '{name}' not found. "
                f"Available: {', '.join(self.calibrator_names)}"
            )
        return self.calibrators[name]


@dataclass
class CalibrationResult:
    """
    Container for calibration results.

    Attributes:
        noise_parameters: Dictionary of noise wave parameters (u, c, s, NS, L)
        predicted_temperatures: Predicted temperatures per calibrator
        residuals: Temperature residuals per calibrator
        model_name: Name of the calibration model used
        metadata: Additional metadata (model config, timing, etc.)
    """
    noise_parameters: Dict[str, jax.Array]
    predicted_temperatures: Dict[str, jax.Array]
    residuals: Dict[str, jax.Array]
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate noise parameters."""
        required_params = {'u', 'c', 's', 'NS', 'L'}
        missing_params = required_params - set(self.noise_parameters.keys())
        if missing_params:
            raise ValueError(
                f"Missing noise parameters: {', '.join(missing_params)}"
            )


class HDF5DataLoader:
    """
    Loader for REACH observation HDF5 files.

    This class handles loading calibration data from HDF5 files following
    the REACH observation format, with automatic discovery of calibrators
    and conversion to JAX arrays.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialise the data loader.

        Args:
            device: Device to place arrays on ('cpu', 'gpu', or None for default)
        """
        self.device = device

    def load_observation(self, filepath: str) -> CalibrationData:
        """
        Load complete observation from HDF5 file.

        Args:
            filepath: Path to HDF5 observation file

        Returns:
            CalibrationData object containing all calibrators and metadata
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")

        with h5py.File(filepath, 'r') as h5file:
            # Discover available calibrators
            calibrator_names = self._discover_calibrators(h5file)

            # Load each calibrator
            calibrators = {}
            vna_frequencies = None

            for cal_name in calibrator_names:
                cal_data = self._extract_calibrator(h5file, cal_name)
                calibrators[cal_name] = cal_data

                # Use first calibrator's VNA frequencies as reference
                if vna_frequencies is None:
                    vna_frequencies = cal_data.s11_freq

            # Compute PSD frequencies
            psd_frequencies = self._compute_psd_frequencies(h5file)

            # Load metadata
            metadata = self._extract_metadata(h5file)

            # Load temperature data if available
            temperatures, temp_timestamps = self._extract_temperatures(h5file)

            return CalibrationData(
                calibrators=calibrators,
                psd_frequencies=psd_frequencies,
                vna_frequencies=vna_frequencies,
                metadata=metadata,
                temperatures=temperatures,
                temperature_timestamps=temp_timestamps
            )

    def _discover_calibrators(self, h5file: h5py.File) -> List[str]:
        """
        Auto-discover calibrators in HDF5 file.

        Args:
            h5file: Open HDF5 file object

        Returns:
            List of calibrator names
        """
        if 'observation_data' not in h5file:
            raise ValueError("HDF5 file missing 'observation_data' group")

        calibrators = set()
        for key in h5file['observation_data'].keys():
            if '_spectra' in key and not any(x in key for x in ['_load', '_ns']):
                cal_name = key.replace('_spectra', '').replace('_timestamps', '')
                calibrators.add(cal_name)

        return sorted(list(calibrators))

    def _extract_calibrator(self, h5file: h5py.File, name: str) -> CalibratorData:
        """
        Extract single calibrator data from HDF5.

        Args:
            h5file: Open HDF5 file object
            name: Calibrator name

        Returns:
            CalibratorData object
        """
        obs_data = h5file['observation_data']

        # Load PSD data
        psd_source = self._to_jax_array(obs_data[f'{name}_spectra'][:])
        psd_load = self._to_jax_array(obs_data[f'{name}_load_spectra'][:])
        psd_ns = self._to_jax_array(obs_data[f'{name}_ns_spectra'][:])

        # Load S11 data
        s11_data = obs_data[f'{name}_s11'][:]
        # S11 format: [measurement, [freq, real, imag], points]
        # We take the first (and usually only) measurement
        s11_freq = self._to_jax_array(s11_data[0, 0, :])
        s11_real = s11_data[0, 1, :]
        s11_imag = s11_data[0, 2, :]
        s11_complex = self._to_jax_array(s11_real + 1j * s11_imag)

        # Load timestamps
        timestamps = self._to_jax_array(obs_data[f'{name}_timestamps'][:])

        # Extract temperature for this specific calibrator
        temperature = self._get_calibrator_temperature(h5file, name)

        return CalibratorData(
            name=name,
            psd_source=psd_source,
            psd_load=psd_load,
            psd_ns=psd_ns,
            s11_freq=s11_freq,
            s11_complex=s11_complex,
            timestamps=timestamps,
            temperature=temperature
        )

    def _get_calibrator_temperature(self, h5file: h5py.File, name: str) -> jax.Array:
        """
        Extract temperature for specific calibrator.

        Maps calibrator names to temperature sensor columns based on
        REACH observation setup.

        Args:
            h5file: Open HDF5 file object
            name: Calibrator name

        Returns:
            Temperature in Kelvin (scalar or array)

        Raises:
            ValueError: If temperature data is missing or cannot be extracted
        """
        # Temperature sensor mapping for REACH calibrators
        # Based on analysis: column 2 is hot (~372K), column 8 is cold (~271K)
        # Others are room temperature sources (~285-287K)
        sensor_mapping = {
            'hot': 2,      # Hot load at ~372K
            'cold': 8,     # Cold load at ~271K
            'ant': 0,      # Antenna
            'r25': 1,      # 25 ohm resistor (room temp)
            'r100': 3,     # 100 ohm resistor (room temp)
            'c2r27': 4,    # 2m cable with 27 ohm
            'c2r36': 5,    # 2m cable with 36 ohm
            'c2r69': 6,    # 2m cable with 69 ohm
            'c2r91': 7,    # 2m cable with 91 ohm
            'c10open': 1,  # 10m cable open (room temp)
            'c10short': 1, # 10m cable short (room temp)
            'c10r10': 1,   # 10m cable with 10 ohm (room temp)
            'c10r250': 1,  # 10m cable with 250 ohm (room temp)
        }

        # Check if we have a mapping for this calibrator
        if name not in sensor_mapping:
            raise ValueError(f"No temperature sensor mapping defined for calibrator '{name}'")

        try:
            obs_metadata = h5file['observation_metadata']
            if 'temperatures' not in obs_metadata:
                raise ValueError(f"No temperature data found in HDF5 file metadata")

            temperatures = obs_metadata['temperatures'][:]  # Shape: [n_time, n_sensors]

            # Get the sensor index for this calibrator
            sensor_idx = sensor_mapping[name]

            # Validate sensor index
            if sensor_idx >= temperatures.shape[1]:
                raise ValueError(f"Temperature sensor index {sensor_idx} out of range "
                               f"for calibrator '{name}' (only {temperatures.shape[1]} sensors)")

            # Extract temperature from the appropriate column
            cal_temp = temperatures[:, sensor_idx]

            # Take mean over time for stable temperature
            mean_temp = float(np.mean(cal_temp))

            # Validate temperature is reasonable
            if not (0 < mean_temp < 1000):
                raise ValueError(f"Invalid temperature {mean_temp}K for calibrator '{name}'")

            return self._to_jax_array(mean_temp)

        except KeyError as e:
            raise ValueError(f"Failed to extract temperature for calibrator '{name}': "
                           f"Missing data in HDF5 file - {e}")
        except IndexError as e:
            raise ValueError(f"Failed to extract temperature for calibrator '{name}': "
                           f"Invalid sensor index - {e}")

    def _compute_psd_frequencies(self, h5file: h5py.File) -> jax.Array:
        """
        Compute PSD frequency array from metadata.

        Args:
            h5file: Open HDF5 file object

        Returns:
            JAX array of frequencies in Hz
        """
        # Try to get sampling rate and number of channels from metadata
        try:
            obs_info = h5file['observation_info']
            config = obs_info.attrs.get('instrument_config_file', {})

            # Default REACH values
            sampling_rate = 400e6  # 400 MHz
            n_channels = 16384

            # Try to extract from config if available
            if isinstance(config, dict) and 'spectrometer' in config:
                spec_config = config['spectrometer']
                sampling_rate = spec_config.get('sampling_rate', 400) * 1e6
                n_channels = spec_config.get('nof_frequency_channels', 16384)
        except:
            # Use defaults if metadata extraction fails
            sampling_rate = 400e6
            n_channels = 16384
            warnings.warn(
                "Could not extract spectrometer config from metadata, "
                "using defaults (400 MHz, 16384 channels)"
            )

        # Compute frequency array
        freq_resolution = sampling_rate / (2 * n_channels)
        frequencies = jnp.arange(n_channels) * freq_resolution

        return self._to_jax_array(frequencies)

    def _extract_metadata(self, h5file: h5py.File) -> Dict[str, Any]:
        """
        Extract metadata from HDF5 file.

        Args:
            h5file: Open HDF5 file object

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        # Extract observation info attributes
        if 'observation_info' in h5file:
            obs_info = h5file['observation_info']
            for key in obs_info.attrs.keys():
                metadata[key] = obs_info.attrs[key]

        # Add file-level metadata
        metadata['hdf5_filename'] = h5file.filename
        metadata['calibrator_count'] = len(self._discover_calibrators(h5file))

        return metadata

    def _extract_temperatures(
        self, h5file: h5py.File
    ) -> Tuple[Optional[jax.Array], Optional[jax.Array]]:
        """
        Extract temperature data from HDF5 file.

        Args:
            h5file: Open HDF5 file object

        Returns:
            Tuple of (temperatures, timestamps) or (None, None) if not available
        """
        try:
            obs_metadata = h5file['observation_metadata']
            temperatures = self._to_jax_array(obs_metadata['temperatures'][:])
            temp_timestamps = self._to_jax_array(
                obs_metadata['temperature_timestamps'][:]
            )
            return temperatures, temp_timestamps
        except KeyError:
            return None, None

    def _to_jax_array(self, arr: np.ndarray) -> jax.Array:
        """
        Convert numpy array to JAX array with device placement.

        Args:
            arr: Numpy array

        Returns:
            JAX array on specified device
        """
        jax_arr = jnp.array(arr)
        if self.device:
            jax_arr = jax.device_put(jax_arr, jax.devices(self.device)[0])
        return jax_arr

    def apply_frequency_mask(
        self, data: CalibrationData, mask: jax.Array
    ) -> CalibrationData:
        """
        Apply frequency selection mask to calibration data.

        Args:
            data: CalibrationData object
            mask: Boolean mask for frequency selection

        Returns:
            New CalibrationData with masked frequencies
        """
        # Apply mask to VNA frequencies
        vna_mask = mask
        masked_vna_freq = data.vna_frequencies[vna_mask]

        # Compute PSD mask (may need interpolation if frequencies don't align)
        # For now, assume we can create a simple mask based on frequency range
        psd_freq_min = data.vna_frequencies[vna_mask][0]
        psd_freq_max = data.vna_frequencies[vna_mask][-1]
        psd_mask = (data.psd_frequencies >= psd_freq_min) & (
            data.psd_frequencies <= psd_freq_max
        )
        masked_psd_freq = data.psd_frequencies[psd_mask]

        # Apply mask to all calibrators
        masked_calibrators = {}
        for name, cal in data.calibrators.items():
            masked_calibrators[name] = CalibratorData(
                name=name,
                psd_source=cal.psd_source[:, psd_mask],
                psd_load=cal.psd_load[:, psd_mask],
                psd_ns=cal.psd_ns[:, psd_mask],
                s11_freq=cal.s11_freq[vna_mask],
                s11_complex=cal.s11_complex[vna_mask],
                timestamps=cal.timestamps,
                temperature=cal.temperature
            )

        return CalibrationData(
            calibrators=masked_calibrators,
            psd_frequencies=masked_psd_freq,
            vna_frequencies=masked_vna_freq,
            metadata=data.metadata,
            temperatures=data.temperatures,
            temperature_timestamps=data.temperature_timestamps
        )

    def save_results(self, filepath: str, result: CalibrationResult) -> None:
        """
        Save calibration results to HDF5 file.

        Args:
            filepath: Output HDF5 file path
            result: CalibrationResult object to save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as h5file:
            # Save noise parameters
            noise_group = h5file.create_group('noise_parameters')
            for param_name, param_value in result.noise_parameters.items():
                noise_group.create_dataset(
                    param_name, data=np.array(param_value)
                )

            # Save predicted temperatures
            pred_group = h5file.create_group('predicted_temperatures')
            for cal_name, temps in result.predicted_temperatures.items():
                pred_group.create_dataset(
                    cal_name, data=np.array(temps)
                )

            # Save residuals
            resid_group = h5file.create_group('residuals')
            for cal_name, resids in result.residuals.items():
                resid_group.create_dataset(
                    cal_name, data=np.array(resids)
                )

            # Save metadata
            h5file.attrs['model_name'] = result.model_name
            for key, value in result.metadata.items():
                try:
                    h5file.attrs[key] = value
                except TypeError:
                    # Skip items that can't be saved as HDF5 attributes
                    h5file.attrs[key] = str(value)