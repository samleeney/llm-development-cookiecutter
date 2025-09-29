#!/usr/bin/env python3
"""
Convert test dataset from text files to HDF5 format matching REACH observation structure.
"""

import numpy as np
import h5py
from pathlib import Path
from skrf import Network
import sys
import json

def load_psd_data(filepath):
    """Load PSD data from text file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip header lines starting with #
    data_lines = [line for line in lines if not line.startswith('#')]

    # Parse data
    data = []
    for line in data_lines:
        if line.strip():  # Skip empty lines
            values = np.array([float(x) for x in line.strip().split(',')])
            data.append(values)

    data = np.array(data)

    # Check if multiple time samples or single
    if len(data.shape) == 1:
        # Single time sample, reshape to [1, n_freq]
        return data.reshape(1, -1)
    return data

def load_s11_data(filepath):
    """Load S11 data from .s1p file."""
    ntwk = Network(filepath)
    # Extract frequency, real and imaginary parts
    freq = ntwk.f  # Frequency in Hz
    s11_complex = ntwk.s[:, 0, 0]  # S11 parameter
    s11_real = s11_complex.real
    s11_imag = s11_complex.imag

    # Format as [1 measurement, 3 (freq/real/imag), n_points]
    s11_data = np.zeros((1, 3, len(freq)))
    s11_data[0, 0, :] = freq
    s11_data[0, 1, :] = s11_real
    s11_data[0, 2, :] = s11_imag

    return s11_data

def load_temperature(filepath):
    """Load temperature from text file."""
    try:
        temp_str = Path(filepath).read_text().strip()
        # Parse comma-separated values
        temps = np.array([float(x) for x in temp_str.split(',')])
        # If we have 12288 values (frequency-dependent), return array
        if len(temps) == 12288:
            return temps
        # Otherwise return the mean as a scalar
        return float(np.mean(temps))
    except Exception as e:
        print(f"  Warning: Could not load temperature from {filepath}: {e}")
        return None

def convert_to_hdf5(input_dir, output_file):
    """
    Convert test dataset to HDF5 format.

    Args:
        input_dir: Path to test_dataset directory
        output_file: Output HDF5 file path
    """
    input_dir = Path(input_dir)
    cal_dir = input_dir / "calibration"

    # Get list of all calibrators
    calibrators = [d.name for d in cal_dir.iterdir() if d.is_dir()]
    print(f"Found {len(calibrators)} calibrators: {calibrators}")

    # First, check the data dimensions
    sample_psd = load_psd_data(cal_dir / calibrators[0] / "psd_source.txt")
    n_freq_psd = sample_psd.shape[1]  # Should be 12288
    print(f"PSD data has {n_freq_psd} frequency points")

    # Create frequency array for PSD data (0-200 MHz matching REACH format)
    # Note: test data has 12288 points, but REACH expects 16384
    # We'll pad the data to match
    n_freq_reach = 16384
    # Note: We don't store frequencies directly - the data loader computes them

    # Create HDF5 file
    with h5py.File(output_file, 'w') as h5f:
        # Create groups
        obs_data = h5f.create_group('observation_data')
        obs_info = h5f.create_group('observation_info')
        obs_metadata = h5f.create_group('observation_metadata')

        # Note: Frequency arrays are computed by the data loader, not stored in HDF5
        # The loader uses metadata to compute frequencies

        # Load and store LNA S11 data (same format as REACH observation)
        lna_file = input_dir / "lna.s1p"
        if lna_file.exists():
            print("Loading LNA S11 data...")
            lna_s11_data = load_s11_data(lna_file)
            obs_data.create_dataset('lna_s11', data=lna_s11_data)
            # Also create timestamp for consistency
            lna_timestamp = np.array([[0.0]])
            obs_data.create_dataset('lna_s11_timestamp', data=lna_timestamp)
            print(f"  Added LNA S11 data with shape {lna_s11_data.shape}")

        # Collect temperature data for metadata
        temperatures_list = []
        cal_temp_mapping = {}

        # Process each calibrator
        for cal_idx, cal_name in enumerate(sorted(calibrators)):
            print(f"Processing {cal_name}...")
            cal_path = cal_dir / cal_name

            # Load PSD data
            psd_source_file = cal_path / "psd_source.txt"
            psd_load_file = cal_path / "psd_load.txt"
            psd_noise_file = cal_path / "psd_noise.txt"

            if psd_source_file.exists():
                psd_source = load_psd_data(psd_source_file)
                psd_load = load_psd_data(psd_load_file)
                psd_ns = load_psd_data(psd_noise_file)

                # Pad PSD data from 12288 to 16384 points
                # The test data covers 50-200 MHz in 12288 points
                # We need to pad to 0-200 MHz in 16384 points
                n_time = psd_source.shape[0]
                padded_source = np.zeros((n_time, n_freq_reach))
                padded_load = np.zeros((n_time, n_freq_reach))
                padded_ns = np.zeros((n_time, n_freq_reach))

                # Calculate where the 50 MHz starts in the new array
                start_idx = int(50e6 / 200e6 * n_freq_reach)  # ~4096
                end_idx = start_idx + n_freq_psd

                # Fill in the data in the correct frequency range
                padded_source[:, start_idx:end_idx] = psd_source
                padded_load[:, start_idx:end_idx] = psd_load
                padded_ns[:, start_idx:end_idx] = psd_ns

                # For frequencies < 50 MHz, use the first value
                padded_source[:, :start_idx] = psd_source[:, 0:1]
                padded_load[:, :start_idx] = psd_load[:, 0:1]
                padded_ns[:, :start_idx] = psd_ns[:, 0:1]

                # For frequencies > data range, use the last value
                if end_idx < n_freq_reach:
                    padded_source[:, end_idx:] = psd_source[:, -1:]
                    padded_load[:, end_idx:] = psd_load[:, -1:]
                    padded_ns[:, end_idx:] = psd_ns[:, -1:]

                # Save padded PSD data
                obs_data.create_dataset(f'{cal_name}_spectra', data=padded_source)
                obs_data.create_dataset(f'{cal_name}_load_spectra', data=padded_load)
                obs_data.create_dataset(f'{cal_name}_ns_spectra', data=padded_ns)

                # Create timestamps (dummy for now)
                n_time = psd_source.shape[0]
                timestamps = np.zeros((n_time, 2))
                timestamps[:, 0] = np.arange(n_time) * 60  # Start times
                timestamps[:, 1] = timestamps[:, 0] + 58   # End times
                obs_data.create_dataset(f'{cal_name}_timestamps', data=timestamps)

            # Load S11 data
            s1p_file = cal_path / f"{cal_name}.s1p"
            if s1p_file.exists():
                s11_data = load_s11_data(s1p_file)
                obs_data.create_dataset(f'{cal_name}_s11', data=s11_data)

                # Create S11 timestamp
                s11_timestamp = np.array([[0.0]])
                obs_data.create_dataset(f'{cal_name}_s11_timestamp', data=s11_timestamp)

            # Load temperature
            temp_file = cal_path / "temperature_mean.txt"
            if temp_file.exists():
                temp = load_temperature(temp_file)
                if temp is not None:
                    temperatures_list.append(temp)
                    cal_temp_mapping[cal_name] = cal_idx

        # Create calibrator temperature arrays for easier access
        # Store actual temperatures for each calibrator
        calibrator_temperatures = {}
        for cal_name in calibrators:
            temp_file = cal_dir / cal_name / "temperature_mean.txt"
            if temp_file.exists():
                temp = load_temperature(temp_file)
                if temp is not None:
                    calibrator_temperatures[cal_name] = temp
                    if isinstance(temp, np.ndarray):
                        # Store frequency-dependent temperature
                        obs_data.create_dataset(f'{cal_name}_temperature', data=temp)
                        print(f"    {cal_name}: frequency-dependent temperature (shape {temp.shape})")
                    else:
                        # Store scalar temperature
                        obs_data.create_dataset(f'{cal_name}_temperature', data=np.array([temp]))
                        print(f"    {cal_name}: {temp:.1f} K")

        # Create temperature metadata matrix matching REACH observation format
        # The reach_observation.hdf5 has 9 sensors, we'll match that
        n_sensors = 9  # Match REACH observation
        n_time_temps = 10  # Dummy time samples

        # Temperature matrix [n_time, n_sensors]
        temp_matrix = np.zeros((n_time_temps, n_sensors))

        # Fill temperature matrix based on actual calibrator temperatures
        # Default all to room temp first
        temp_matrix[:, :] = 298.0 + np.random.randn(n_time_temps, n_sensors) * 0.1

        # Map calibrators to sensor indices (matching data.py sensor_mapping)
        # Must match the exact mapping in src/data.py
        sensor_mapping = {
            'ant': 0,      # Antenna
            'r25': 1,      # 25 ohm resistor (room temp)
            'hot': 2,      # Hot load at ~372K
            'r100': 3,     # 100 ohm resistor (room temp)
            'c2r27': 4,    # 2m cable with 27 ohm
            'c2r36': 5,    # 2m cable with 36 ohm
            'c2r69': 6,    # 2m cable with 69 ohm
            'c2r91': 7,    # 2m cable with 91 ohm
            'cold': 8,     # Cold load at ~271K
            # These map to existing sensors
            'c10open': 1,  # Maps to sensor 1 (same as r25)
            'c10short': 1, # Maps to sensor 1 (same as r25)
            'c10r10': 1,   # Maps to sensor 1 (same as r25)
            'c10r250': 1,  # Maps to sensor 1 (same as r25)
        }

        # Set temperature values based on actual calibrator temperatures
        # Process unique sensors first
        for cal_name in ['ant', 'hot', 'cold', 'r25', 'r100', 'c2r27', 'c2r36', 'c2r69', 'c2r91']:
            if cal_name in calibrator_temperatures and cal_name in sensor_mapping:
                val = calibrator_temperatures[cal_name]
                if isinstance(val, np.ndarray):
                    val = np.mean(val)
                # For antenna, keep the actual 5000K value for proper residual calculation
                sensor_idx = sensor_mapping[cal_name]
                temp_matrix[:, sensor_idx] = val + np.random.randn(n_time_temps) * 0.01  # Small variation

        # Save temperature metadata
        obs_metadata.create_dataset('temperatures', data=temp_matrix.astype(np.float32))

        # Create temperature timestamps
        temp_timestamps = np.arange(n_time_temps) * 60.0  # Every minute
        obs_metadata.create_dataset('temperature_timestamps', data=temp_timestamps)

        # Create dummy TPM temperatures (not used but keeps format consistent)
        tpm_temps = np.random.randn(n_time_temps + 1, 3) * 5 + 50  # Around 50C
        obs_metadata.create_dataset('tpm_temperatures', data=tpm_temps.astype(np.float32))
        obs_metadata.create_dataset('tpm_temperature_timestamps', data=np.arange(n_time_temps + 1) * 60.0)

        # Add metadata attributes for frequency computation
        obs_info.attrs['observation_name'] = 'test_dataset'
        obs_info.attrs['instrument'] = 'REACH'
        obs_info.attrs['date'] = '2024-01-01'

        # Add spectrometer configuration for frequency array computation
        # Store as JSON string since HDF5 can't directly store nested dicts
        config = {
            'spectrometer': {
                'sampling_rate': 400,  # MHz
                'nof_frequency_channels': 16384
            }
        }
        obs_info.attrs['instrument_config_file'] = json.dumps(config)

    print(f"\nSuccessfully created HDF5 file: {output_file}")
    print(f"File size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")

def verify_hdf5(filepath):
    """Verify the created HDF5 file structure."""
    print(f"\nVerifying HDF5 file structure...")
    with h5py.File(filepath, 'r') as h5f:
        print("\nGroups:")
        for group_name in h5f.keys():
            print(f"  {group_name}")
            group = h5f[group_name]
            if isinstance(group, h5py.Group):
                print(f"    Contains {len(group.keys())} datasets/subgroups")

        # Check calibrator data
        obs_data = h5f['observation_data']
        calibrators = set()
        for key in obs_data.keys():
            if '_spectra' in key and not '_load_' in key and not '_ns_' in key:
                cal_name = key.replace('_spectra', '')
                calibrators.add(cal_name)

        print(f"\nCalibrators found: {sorted(calibrators)}")

        # Check temperatures
        if 'observation_metadata/temperatures' in h5f:
            temps = h5f['observation_metadata/temperatures'][...]
            print(f"\nTemperature matrix shape: {temps.shape}")
            print(f"Mean temperatures per sensor:")
            for i in range(min(5, temps.shape[1])):
                print(f"  Sensor {i}: {np.mean(temps[:, i]):.1f} K")

if __name__ == "__main__":
    input_dir = "calibration_pipeline_example/test_dataset"
    output_file = "data/test_observation.hdf5"

    print("Converting test dataset to HDF5 format...")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}\n")

    convert_to_hdf5(input_dir, output_file)
    verify_hdf5(output_file)