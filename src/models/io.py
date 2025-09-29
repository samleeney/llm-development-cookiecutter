"""
I/O utilities for saving and loading calibration results.
"""

import h5py
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Optional
import json


def save_calibration_result(result, filepath: str, model=None, data=None):
    """
    Save calibration results to HDF5 file.

    Args:
        result: CalibrationResult object
        filepath: Path to save the file
        model: Optional model object to save parameters
        data: Optional CalibrationData to save metadata
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        # Save noise parameters
        params_group = f.create_group('noise_parameters')
        for key, values in result.noise_parameters.items():
            params_group.create_dataset(key, data=np.array(values))

        # Save predicted temperatures
        pred_group = f.create_group('predicted_temperatures')
        for cal_name, temps in result.predicted_temperatures.items():
            pred_group.create_dataset(cal_name, data=np.array(temps))

        # Save residuals
        res_group = f.create_group('residuals')
        for cal_name, res in result.residuals.items():
            res_group.create_dataset(cal_name, data=np.array(res))

        # Save metadata
        meta_group = f.create_group('metadata')
        meta_group.attrs['model_name'] = result.model_name

        if result.metadata:
            for key, value in result.metadata.items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
                elif isinstance(value, dict):
                    meta_group.attrs[key] = json.dumps(value)

        # Save frequencies if data provided
        if data is not None:
            freq_group = f.create_group('frequencies')
            freq_group.create_dataset('psd_frequencies',
                                     data=np.array(data.psd_frequencies))
            freq_group.create_dataset('vna_frequencies',
                                     data=np.array(data.vna_frequencies))

        # Save model configuration if provided
        if model is not None:
            model_group = f.create_group('model_config')
            config = model.get_config() if hasattr(model, 'get_config') else {}
            for key, value in config.items():
                model_group.attrs[key] = value

    print(f"Saved calibration results to {filepath}")


def load_calibration_result(filepath: str) -> Dict[str, Any]:
    """
    Load calibration results from HDF5 file.

    Args:
        filepath: Path to the HDF5 file

    Returns:
        Dictionary containing calibration results
    """
    results = {}

    with h5py.File(filepath, 'r') as f:
        # Load noise parameters
        if 'noise_parameters' in f:
            results['noise_parameters'] = {}
            for key in f['noise_parameters'].keys():
                results['noise_parameters'][key] = jnp.array(
                    f['noise_parameters'][key][...]
                )

        # Load predicted temperatures
        if 'predicted_temperatures' in f:
            results['predicted_temperatures'] = {}
            for cal_name in f['predicted_temperatures'].keys():
                results['predicted_temperatures'][cal_name] = jnp.array(
                    f['predicted_temperatures'][cal_name][...]
                )

        # Load residuals
        if 'residuals' in f:
            results['residuals'] = {}
            for cal_name in f['residuals'].keys():
                results['residuals'][cal_name] = jnp.array(
                    f['residuals'][cal_name][...]
                )

        # Load metadata
        if 'metadata' in f:
            results['metadata'] = dict(f['metadata'].attrs)
            # Parse JSON strings back to dicts
            for key, value in results['metadata'].items():
                if isinstance(value, str) and value.startswith('{'):
                    try:
                        results['metadata'][key] = json.loads(value)
                    except:
                        pass

        # Load frequencies
        if 'frequencies' in f:
            results['frequencies'] = {}
            if 'psd_frequencies' in f['frequencies']:
                results['frequencies']['psd'] = jnp.array(
                    f['frequencies']['psd_frequencies'][...]
                )
            if 'vna_frequencies' in f['frequencies']:
                results['frequencies']['vna'] = jnp.array(
                    f['frequencies']['vna_frequencies'][...]
                )

        # Load model configuration
        if 'model_config' in f:
            results['model_config'] = dict(f['model_config'].attrs)

    print(f"Loaded calibration results from {filepath}")
    return results


def save_noise_parameters_csv(params: Dict[str, jnp.ndarray],
                              frequencies: jnp.ndarray,
                              filepath: str):
    """
    Save noise wave parameters to CSV file for easy analysis.

    Args:
        params: Dictionary of noise parameters
        frequencies: Frequency array
        filepath: Output CSV file path
    """
    import pandas as pd

    # Create dataframe
    df_data = {'frequency_MHz': frequencies / 1e6}
    for key, values in params.items():
        df_data[f'T_{key}_K'] = np.array(values)

    df = pd.DataFrame(df_data)
    df.to_csv(filepath, index=False)
    print(f"Saved noise parameters to {filepath}")