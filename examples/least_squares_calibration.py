#!/usr/bin/env python3
"""
Example script demonstrating least squares calibration with REACH data.

This script loads REACH observation data and performs calibration using
the least squares method to extract noise wave parameters.
"""

import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import HDF5DataLoader
from src.models.least_squares import LeastSquaresModel
from src.visualization.calibration_plots import CalibrationPlotter


def main():
    """Run least squares calibration on observation data."""
    # Get path from command line or use default
    if len(sys.argv) > 1:
        obs_file = Path(sys.argv[1])
        print(f"Using data file: {obs_file}")
    else:
        obs_file = Path('data/reach_observation.hdf5')
        print(f"Using default data file: {obs_file}")

    if not obs_file.exists():
        print(f"Error: Observation file not found at {obs_file}")
        print("Please ensure the data file is in the correct location")
        return

    print("=" * 70)
    print("LEAST SQUARES CALIBRATION EXAMPLE")
    print("=" * 70)

    # Load data
    print("\n1. Loading observation data...")
    loader = HDF5DataLoader()
    data = loader.load_observation(str(obs_file))

    print(f"   - Loaded {len(data.calibrators)} calibrators")
    print(f"   - Frequency range: {data.psd_frequencies[0]/1e6:.1f} - "
          f"{data.psd_frequencies[-1]/1e6:.1f} MHz")
    print(f"   - Number of channels: {len(data.psd_frequencies)}")

    # Apply frequency mask (50-130 MHz for cleaner results)
    print("\n2. Applying frequency mask (50-130 MHz)...")
    mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
    masked_data = loader.apply_frequency_mask(data, mask)
    print(f"   - Masked to {len(masked_data.vna_frequencies)} frequency points")

    # Create and configure model
    print("\n3. Creating least squares model...")
    config = {
        'regularisation': 0.0,  # No regularisation
        'use_gamma_weighting': False
    }
    model = LeastSquaresModel(config)
    print(f"   - Configuration: {config}")

    # Fit the model
    print("\n4. Fitting model to calibration data...")
    start_time = time.time()
    model.fit(masked_data)
    fit_time = time.time() - start_time
    print(f"   - Fitting completed in {fit_time:.3f} seconds")

    # Get fitted parameters
    print("\n5. Extracting noise wave parameters...")
    params = model.get_parameters()

    # Print parameter statistics
    print("\n   Noise Wave Parameter Statistics:")
    print("   " + "-" * 45)
    param_names = {'u': 'Uncorrelated', 'c': 'Cosine', 's': 'Sine',
                   'NS': 'Noise Source', 'L': 'Load'}

    for key, name in param_names.items():
        values = params[key]
        print(f"   {name:15s} ({key:2s}): "
              f"mean={float(jnp.mean(values)):8.2f}, "
              f"std={float(jnp.std(values)):7.2f}, "
              f"range=[{float(jnp.min(values)):7.2f}, {float(jnp.max(values)):7.2f}]")

    # Generate predictions
    print("\n6. Generating temperature predictions...")
    result = model.get_result()

    # Calculate residuals statistics
    print("\n   Residual Statistics (Predicted - Measured):")
    print("   " + "-" * 45)
    for cal_name in ['hot', 'cold', 'ant']:
        if cal_name in result.residuals:
            residuals = result.residuals[cal_name]
            print(f"   {cal_name:5s}: "
                  f"mean={float(jnp.mean(residuals)):7.2f} K, "
                  f"std={float(jnp.std(residuals)):7.2f} K, "
                  f"max|res|={float(jnp.max(jnp.abs(residuals))):7.2f} K")

    # Create comprehensive visualisation plots
    print("\n7. Creating visualisation plots...")
    plotter = CalibrationPlotter(output_dir=Path("plots"), save=True, show=False)

    # Create all plots
    plotter.plot_all_calibrators(masked_data, model, result, antenna_validation=False)
    plotter.plot_noise_parameters(masked_data, model, param_smoothing=50)
    plotter.plot_residuals_summary(result)
    plotter.plot_antenna_temperature(masked_data, result)
    plotter.create_summary_plot(masked_data, model, result)

    print("  Plots saved to plots/ directory")

    # Test with different configurations
    print("\n8. Testing alternative configurations...")
    test_configurations(masked_data)

    print("\n✓ Calibration completed successfully!")
    print("  All plots saved to plots/ directory")


def create_plots(data, model, result):
    """Create visualisation plots for calibration results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Least Squares Calibration Results', fontsize=14, fontweight='bold')

    freq_mhz = data.psd_frequencies / 1e6
    params = model.get_parameters()

    # Plot 1: Noise wave parameters
    ax = axes[0, 0]
    ax.plot(freq_mhz, params['u'], label='u (uncorrelated)', alpha=0.8)
    ax.plot(freq_mhz, params['c'], label='c (cosine)', alpha=0.8)
    ax.plot(freq_mhz, params['s'], label='s (sine)', alpha=0.8)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Noise Temperature (K)')
    ax.set_title('Correlated/Uncorrelated Noise Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Source and loss parameters
    ax = axes[0, 1]
    ax.plot(freq_mhz, params['NS'], label='NS (noise source)', alpha=0.8, color='red')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Noise Source Temp (K)', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(freq_mhz, params['L'], label='L (loss)', alpha=0.8, color='blue')
    ax2.set_ylabel('Load Temperature', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax.set_title('Noise Source & Load Temperature')

    # Plot 3: Temperature predictions for hot calibrator
    ax = axes[0, 2]
    if 'hot' in result.predicted_temperatures:
        hot_cal = data.get_calibrator('hot')
        T_pred = result.predicted_temperatures['hot']
        T_meas = hot_cal.temperature

        ax.plot(freq_mhz, T_pred, label='Predicted', alpha=0.8)
        if T_meas.ndim == 0:
            ax.axhline(float(T_meas), color='red', linestyle='--',
                      label=f'Measured ({float(T_meas):.1f} K)')
        else:
            ax.plot(freq_mhz, T_meas, 'r--', label='Measured', alpha=0.8)

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Hot Calibrator Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Temperature predictions for cold calibrator
    ax = axes[1, 0]
    if 'cold' in result.predicted_temperatures:
        cold_cal = data.get_calibrator('cold')
        T_pred = result.predicted_temperatures['cold']
        T_meas = cold_cal.temperature

        ax.plot(freq_mhz, T_pred, label='Predicted', alpha=0.8)
        if T_meas is not None:
            if T_meas.ndim == 0:
                ax.axhline(float(T_meas), color='blue', linestyle='--',
                          label=f'Measured ({float(T_meas):.1f} K)')
            else:
                ax.plot(freq_mhz, T_meas, 'b--', label='Measured', alpha=0.8)

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Cold Calibrator Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 5: Antenna temperature prediction
    ax = axes[1, 1]
    if 'ant' in result.predicted_temperatures:
        T_ant = result.predicted_temperatures['ant']
        ax.plot(freq_mhz, T_ant, color='green', alpha=0.8)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Antenna Temperature Prediction')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_T = float(jnp.mean(T_ant))
        std_T = float(jnp.std(T_ant))
        ax.text(0.05, 0.95, f'Mean: {mean_T:.1f} K\nStd: {std_T:.1f} K',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 6: Residuals for all calibrators
    ax = axes[1, 2]
    colors = {'hot': 'red', 'cold': 'blue', 'ant': 'green'}
    for cal_name in ['hot', 'cold', 'ant']:
        if cal_name in result.residuals:
            residuals = result.residuals[cal_name]
            ax.plot(freq_mhz, residuals, label=cal_name, alpha=0.7,
                   color=colors.get(cal_name, 'black'))

    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Residual (K)')
    ax.set_title('Temperature Residuals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('least_squares_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_configurations(data):
    """Test different model configurations."""
    configs = [
        {'name': 'Standard', 'config': {}},
        {'name': 'With Regularisation', 'config': {'regularisation': 0.001}},
        {'name': 'Gamma Weighted', 'config': {'use_gamma_weighting': True}},
    ]

    print("\n   Configuration Comparison:")
    print("   " + "-" * 60)

    for cfg in configs:
        try:
            model = LeastSquaresModel(cfg['config'])
            start = time.time()
            model.fit(data)
            fit_time = time.time() - start

            params = model.get_parameters()
            u_mean = float(jnp.mean(params['u']))
            ns_mean = float(jnp.mean(params['NS']))

            print(f"   {cfg['name']:20s}: "
                  f"time={fit_time:.3f}s, "
                  f"u_mean={u_mean:7.2f} K, "
                  f"NS_mean={ns_mean:7.2f} K")
        except Exception as e:
            print(f"   {cfg['name']:20s}: Failed - {str(e)[:40]}")


if __name__ == '__main__':
    main()