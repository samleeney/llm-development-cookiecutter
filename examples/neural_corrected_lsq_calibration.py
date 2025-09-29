#!/usr/bin/env python3
"""
Example script demonstrating neural-corrected least squares calibration.

This script loads REACH observation data and performs calibration using
the hybrid physics-ML approach that combines analytical least squares
with neural network corrections.
"""

import sys
from pathlib import Path
import jax.numpy as jnp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import HDF5DataLoader, CalibrationData
from src.models.neural_corrected_lsq import NeuralCorrectedLSQModel
from src.models.least_squares import LeastSquaresModel
from src.visualization.calibration_plots import CalibrationPlotter


def main():
    """Run neural-corrected least squares calibration on observation data."""
    obs_file = Path('data/test_observation.hdf5')

    # Load and prepare data
    print("=" * 70)
    print("NEURAL-CORRECTED LEAST SQUARES CALIBRATION")
    print("=" * 70)
    print("\nLoading observation data...")
    loader = HDF5DataLoader()
    data = loader.load_observation(str(obs_file))

    # Apply frequency mask (50-130 MHz for cleaner results)
    mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
    masked_data = loader.apply_frequency_mask(data, mask)

    # Filter to specific calibrators
    calibrators_to_use = ['hot', 'cold', 'c10open', 'c10short', 'r100', 'ant']
    filtered_calibrators = {
        name: cal for name, cal in masked_data.calibrators.items()
        if name in calibrators_to_use
    }
    filtered_data = CalibrationData(
        calibrators=filtered_calibrators,
        psd_frequencies=masked_data.psd_frequencies,
        vna_frequencies=masked_data.vna_frequencies,
        lna_s11=masked_data.lna_s11,
        metadata=masked_data.metadata
    )

    print(f"Loaded {len(filtered_calibrators)} calibrators")
    print(f"Frequency range: {float(filtered_data.vna_frequencies[0])/1e6:.1f} - "
          f"{float(filtered_data.vna_frequencies[-1])/1e6:.1f} MHz")
    print(f"Number of frequency channels: {len(filtered_data.vna_frequencies)}")

    # Fit pure least squares model for comparison
    print("\n" + "=" * 70)
    print("STAGE 0: Pure Least Squares (for comparison)")
    print("=" * 70)
    lsq_model = LeastSquaresModel({'regularisation': 0.0})
    lsq_model.fit(filtered_data)
    lsq_result = lsq_model.get_result()

    print("\nPure LSQ Calibration Source RMSE:")
    for cal_name in calibrators_to_use[:-1]:  # Exclude antenna
        if cal_name in lsq_result.residuals:
            rmse = float(jnp.sqrt(jnp.mean(lsq_result.residuals[cal_name]**2)))
            print(f"  {cal_name:10s}: {rmse:8.4f} K")

    # Fit neural-corrected model
    print("\n" + "=" * 70)
    print("NEURAL-CORRECTED LEAST SQUARES MODEL")
    print("=" * 70)

    config = {
        'regularisation': 0.0,
        'use_gamma_weighting': False,
        'hidden_layers': [32, 32],
        'learning_rate': 1e-3,
        'n_iterations': 1000,
        'correction_regularization': 0.01
    }

    print("\nConfiguration:")
    print(f"  Hidden layers: {config['hidden_layers']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Training iterations: {config['n_iterations']}")
    print(f"  Correction regularization: {config['correction_regularization']}")

    model = NeuralCorrectedLSQModel(config)
    model.fit(filtered_data)
    result = model.get_result()

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nCalibration source temperature statistics:")
    for cal_name in calibrators_to_use:
        T_pred = result.predicted_temperatures[cal_name]
        print(f"  {cal_name:10s}: "
              f"mean={float(jnp.mean(T_pred)):8.2f} K, "
              f"std={float(jnp.std(T_pred)):7.2f} K")

    print("\nCalibration source RMSE (Neural-Corrected):")
    for cal_name in calibrators_to_use[:-1]:  # Exclude antenna from RMSE (not in fitting)
        if cal_name in result.residuals:
            rmse = float(jnp.sqrt(jnp.mean(result.residuals[cal_name]**2)))
            print(f"  {cal_name:10s}: {rmse:8.4f} K")

    # Neural network correction statistics
    print("\nNeural Network Correction Statistics:")
    correction_stats = model.get_correction_magnitude()
    print(f"  Mean:  {correction_stats['mean']:7.4f} K")
    print(f"  Std:   {correction_stats['std']:7.4f} K")
    print(f"  RMS:   {correction_stats['rms']:7.4f} K")
    print(f"  Range: [{correction_stats['min']:7.4f}, {correction_stats['max']:7.4f}] K")

    # Comparison with pure LSQ
    print("\n" + "=" * 70)
    print("COMPARISON: Pure LSQ vs Neural-Corrected LSQ")
    print("=" * 70)

    print("\nRMSE Comparison:")
    print(f"{'Calibrator':<12} {'Pure LSQ':>12} {'Neural LSQ':>12} {'Improvement':>12}")
    print("-" * 50)
    for cal_name in calibrators_to_use[:-1]:
        if cal_name in lsq_result.residuals and cal_name in result.residuals:
            lsq_rmse = float(jnp.sqrt(jnp.mean(lsq_result.residuals[cal_name]**2)))
            neural_rmse = float(jnp.sqrt(jnp.mean(result.residuals[cal_name]**2)))
            improvement = (lsq_rmse - neural_rmse) / lsq_rmse * 100

            print(f"{cal_name:<12} {lsq_rmse:12.4f} {neural_rmse:12.4f} {improvement:11.2f}%")

    # Note about synthetic data
    print("\n" + "=" * 70)
    print("NOTE: Expected Behavior on Synthetic Data")
    print("=" * 70)
    print("""
For this synthetic test dataset, the physical model (least squares) is
sufficient to describe the data perfectly. Therefore, the neural network
corrections should be very small (near zero), as there are no unmodeled
systematic effects to correct.

On real observational data with imperfect switches, cable temperature
gradients, and other instrumental effects, the neural network corrections
would be larger and would capture these systematic effects.
    """)

    # Create plots
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    print("\nCreating plots (saved to results/)...")

    plotter = CalibrationPlotter(output_dir=Path("results"), save=True, show=False)

    # Plot all calibrators
    plotter.plot_all_calibrators(filtered_data, model, result, antenna_validation=False)

    # Plot noise parameters
    plotter.plot_noise_parameters(filtered_data, model, param_smoothing=50)

    print("\nPlots saved to results/ directory:")
    print("  - calibration_summary_*.png")
    print("  - noise_parameters_*.png")
    print("\nCalibration complete!")


if __name__ == '__main__':
    main()