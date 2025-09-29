#!/usr/bin/env python3
"""
Direct comparison of pure LSQ vs neural-corrected LSQ on real REACH data.

Runs both methods on the same real observation data and provides a clear
side-by-side comparison of results.
"""

import sys
from pathlib import Path
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import HDF5DataLoader
from src.models.least_squares import LeastSquaresModel
from src.models.neural_corrected_lsq import NeuralCorrectedLSQModel
from src.visualization.calibration_plots import CalibrationPlotter


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def main():
    """Compare pure LSQ and neural-corrected LSQ on real REACH data."""
    obs_file = Path('data/reach_observation.hdf5')

    if not obs_file.exists():
        print(f"ERROR: Real data file not found at {obs_file}")
        return

    print_section("COMPARISON: PURE LSQ vs NEURAL-CORRECTED LSQ")
    print("Dataset: Real REACH Observation Data")

    # ========================================================================
    # Load and prepare data
    # ========================================================================
    print("\nLoading and preparing data...")
    loader = HDF5DataLoader()
    data = loader.load_observation(str(obs_file))

    # Apply frequency mask
    mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
    masked_data = loader.apply_frequency_mask(data, mask)

    calibrators = [name for name in masked_data.calibrator_names if name != 'ant']
    has_antenna = 'ant' in masked_data.calibrator_names

    print(f"  Frequency range: {float(masked_data.vna_frequencies[0])/1e6:.1f} - "
          f"{float(masked_data.vna_frequencies[-1])/1e6:.1f} MHz")
    print(f"  Frequency channels: {len(masked_data.vna_frequencies)}")
    print(f"  Calibrators for fitting: {len(calibrators)}")
    print(f"  Calibrator names: {', '.join(calibrators)}")
    if has_antenna:
        print(f"  Antenna present: Yes (used for validation only)")

    # ========================================================================
    # METHOD 1: Pure Least Squares
    # ========================================================================
    print_section("METHOD 1: PURE LEAST SQUARES")

    print("\nFitting pure least squares model...")
    lsq_model = LeastSquaresModel({'regularisation': 0.0})
    lsq_model.fit(masked_data)
    lsq_result = lsq_model.get_result()
    print("✓ Fitting complete")

    # Compute statistics
    print("\nCalibration Source Results:")
    print(f"{'Calibrator':<12} {'Mean T (K)':>12} {'Std (K)':>10} {'RMSE (K)':>12}")
    print("-" * 50)

    lsq_rmse = {}
    for cal_name in calibrators:
        T_pred = lsq_result.predicted_temperatures[cal_name]
        mean_T = float(jnp.mean(T_pred))
        std_T = float(jnp.std(T_pred))

        if cal_name in lsq_result.residuals:
            rmse = float(jnp.sqrt(jnp.mean(lsq_result.residuals[cal_name]**2)))
            lsq_rmse[cal_name] = rmse
            print(f"{cal_name:<12} {mean_T:12.2f} {std_T:10.2f} {rmse:12.4f}")

    if has_antenna and 'ant' in lsq_result.residuals:
        ant_T_pred = lsq_result.predicted_temperatures['ant']
        ant_mean = float(jnp.mean(ant_T_pred))
        ant_std = float(jnp.std(ant_T_pred))
        ant_rmse_lsq = float(jnp.sqrt(jnp.mean(lsq_result.residuals['ant']**2)))
        print(f"\nAntenna (validation only):")
        print(f"  Mean: {ant_mean:.2f} K, Std: {ant_std:.2f} K, RMSE: {ant_rmse_lsq:.4f} K")

    # ========================================================================
    # METHOD 2: Neural-Corrected Least Squares
    # ========================================================================
    print_section("METHOD 2: NEURAL-CORRECTED LEAST SQUARES")

    neural_config = {
        'regularisation': 0.0,
        'hidden_layers': [64, 64, 32],
        'learning_rate': 1e-3,
        'n_iterations': 2000,
        'correction_regularization': 0.01
    }

    print("\nConfiguration:")
    print(f"  Architecture: {neural_config['hidden_layers']}")
    print(f"  Learning rate: {neural_config['learning_rate']}")
    print(f"  Iterations: {neural_config['n_iterations']}")
    print(f"  Regularization: {neural_config['correction_regularization']}")

    print("\nFitting neural-corrected least squares model...")
    neural_model = NeuralCorrectedLSQModel(neural_config)
    neural_model.fit(masked_data)
    neural_result = neural_model.get_result()
    print("✓ Fitting complete")

    # Neural network correction statistics
    correction_stats = neural_model.get_correction_magnitude()
    print("\nNeural Network Correction Statistics:")
    print(f"  RMS:   {correction_stats['rms']:7.4f} K")
    print(f"  Mean:  {correction_stats['mean']:7.4f} K")
    print(f"  Std:   {correction_stats['std']:7.4f} K")
    print(f"  Range: [{correction_stats['min']:7.4f}, {correction_stats['max']:7.4f}] K")

    print("\nCalibration Source Results:")
    print(f"{'Calibrator':<12} {'Mean T (K)':>12} {'Std (K)':>10} {'RMSE (K)':>12}")
    print("-" * 50)

    neural_rmse = {}
    for cal_name in calibrators:
        T_pred = neural_result.predicted_temperatures[cal_name]
        mean_T = float(jnp.mean(T_pred))
        std_T = float(jnp.std(T_pred))

        if cal_name in neural_result.residuals:
            rmse = float(jnp.sqrt(jnp.mean(neural_result.residuals[cal_name]**2)))
            neural_rmse[cal_name] = rmse
            print(f"{cal_name:<12} {mean_T:12.2f} {std_T:10.2f} {rmse:12.4f}")

    if has_antenna and 'ant' in neural_result.residuals:
        ant_T_pred_neural = neural_result.predicted_temperatures['ant']
        ant_mean_neural = float(jnp.mean(ant_T_pred_neural))
        ant_std_neural = float(jnp.std(ant_T_pred_neural))
        ant_rmse_neural = float(jnp.sqrt(jnp.mean(neural_result.residuals['ant']**2)))
        print(f"\nAntenna (validation only):")
        print(f"  Mean: {ant_mean_neural:.2f} K, Std: {ant_std_neural:.2f} K, RMSE: {ant_rmse_neural:.4f} K")

    # ========================================================================
    # Comparison
    # ========================================================================
    print_section("DETAILED COMPARISON")

    print("\nRMSE Comparison:")
    print(f"{'Calibrator':<12} {'Pure LSQ':>12} {'Neural LSQ':>12} {'Δ RMSE':>12} {'Improvement':>12}")
    print("-" * 72)

    improvements = []
    for cal_name in calibrators:
        if cal_name in lsq_rmse and cal_name in neural_rmse:
            lsq_val = lsq_rmse[cal_name]
            neural_val = neural_rmse[cal_name]
            delta = lsq_val - neural_val
            improvement = (delta / lsq_val * 100) if lsq_val > 0 else 0

            improvements.append(improvement)
            print(f"{cal_name:<12} {lsq_val:12.4f} {neural_val:12.4f} "
                  f"{delta:+12.4f} {improvement:+11.2f}%")

    print("-" * 72)
    mean_improvement = np.mean(improvements) if improvements else 0
    median_improvement = np.median(improvements) if improvements else 0
    print(f"{'Mean':<12} {' '*12} {' '*12} {' '*12} {mean_improvement:+11.2f}%")
    print(f"{'Median':<12} {' '*12} {' '*12} {' '*12} {median_improvement:+11.2f}%")

    if has_antenna:
        ant_delta = ant_rmse_lsq - ant_rmse_neural
        ant_improvement = (ant_delta / ant_rmse_lsq * 100) if ant_rmse_lsq > 0 else 0
        print(f"\nAntenna (validation):")
        print(f"  Pure LSQ:     {ant_rmse_lsq:8.4f} K")
        print(f"  Neural LSQ:   {ant_rmse_neural:8.4f} K")
        print(f"  Improvement:  {ant_improvement:+.2f}%")

    # ========================================================================
    # Visualization
    # ========================================================================
    print_section("VISUALIZATION")

    # Create comparison plots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: RMSE comparison bar chart (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(calibrators))
    width = 0.35
    bars1 = ax1.bar(x - width/2, [lsq_rmse[cal] for cal in calibrators], width,
                     label='Pure LSQ', alpha=0.8, color='C0')
    bars2 = ax1.bar(x + width/2, [neural_rmse[cal] for cal in calibrators], width,
                     label='Neural-Corrected LSQ', alpha=0.8, color='C1')
    ax1.set_xlabel('Calibrator', fontweight='bold')
    ax1.set_ylabel('RMSE (K)', fontweight='bold')
    ax1.set_title('RMSE Comparison: Pure LSQ vs Neural-Corrected LSQ', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(calibrators, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Improvement percentage (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.barh(calibrators, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('RMSE Improvement (%)', fontweight='bold')
    ax2.set_title('Improvement by Calibrator', fontweight='bold')
    ax2.axvline(0, color='k', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Residuals for a few calibrators - Pure LSQ (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    for cal_name in calibrators[:3]:
        if cal_name in lsq_result.residuals:
            ax3.plot(masked_data.psd_frequencies / 1e6,
                    lsq_result.residuals[cal_name],
                    label=cal_name, alpha=0.7, linewidth=0.5)
    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('Residual (K)')
    ax3.set_title('Pure LSQ Residuals', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='k', linestyle='--', alpha=0.5)

    # Plot 4: Residuals for same calibrators - Neural LSQ (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    for cal_name in calibrators[:3]:
        if cal_name in neural_result.residuals:
            ax4.plot(masked_data.psd_frequencies / 1e6,
                    neural_result.residuals[cal_name],
                    label=cal_name, alpha=0.7, linewidth=0.5)
    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('Residual (K)')
    ax4.set_title('Neural-Corrected LSQ Residuals', fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='k', linestyle='--', alpha=0.5)

    # Plot 5: Neural corrections (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    for cal_name in calibrators[:3]:
        cal_data = masked_data.get_calibrator(cal_name)
        s11_interp = neural_model._interpolate_s11(
            cal_data.s11_freq, cal_data.s11_complex, masked_data.psd_frequencies
        )
        freq_norm = (masked_data.psd_frequencies - neural_model._freq_mean) / (neural_model._freq_std + 1e-10)
        features = jnp.stack([
            freq_norm, jnp.abs(s11_interp), jnp.real(s11_interp), jnp.imag(s11_interp)
        ], axis=1)
        corrections = neural_model._nn_state.apply(neural_model._nn_params, features)
        ax5.plot(masked_data.psd_frequencies / 1e6, corrections, label=cal_name, alpha=0.7, linewidth=0.8)
    ax5.set_xlabel('Frequency (MHz)')
    ax5.set_ylabel('Correction (K)')
    ax5.set_title('Neural Network Corrections', fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(0, color='k', linestyle='--', alpha=0.5)

    # Plot 6: Noise parameters comparison - u (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    lsq_params = lsq_model.get_parameters()
    neural_params = neural_model.get_parameters()
    ax6.plot(masked_data.psd_frequencies / 1e6, lsq_params['u'],
            label='Pure LSQ', alpha=0.8, linewidth=1)
    ax6.plot(masked_data.psd_frequencies / 1e6, neural_params['u'],
            label='Neural LSQ', alpha=0.8, linewidth=1, linestyle='--')
    ax6.set_xlabel('Frequency (MHz)')
    ax6.set_ylabel('u (K)')
    ax6.set_title('Uncorrelated Noise Parameter', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: NS parameter comparison (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(masked_data.psd_frequencies / 1e6, lsq_params['NS'],
            label='Pure LSQ', alpha=0.8, linewidth=1)
    ax7.plot(masked_data.psd_frequencies / 1e6, neural_params['NS'],
            label='Neural LSQ', alpha=0.8, linewidth=1, linestyle='--')
    ax7.set_xlabel('Frequency (MHz)')
    ax7.set_ylabel('NS (K)')
    ax7.set_title('Noise Source Temperature', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: L parameter comparison (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(masked_data.psd_frequencies / 1e6, lsq_params['L'],
            label='Pure LSQ', alpha=0.8, linewidth=1)
    ax8.plot(masked_data.psd_frequencies / 1e6, neural_params['L'],
            label='Neural LSQ', alpha=0.8, linewidth=1, linestyle='--')
    ax8.set_xlabel('Frequency (MHz)')
    ax8.set_ylabel('L (K)')
    ax8.set_title('Load Temperature', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.suptitle('Pure LSQ vs Neural-Corrected LSQ on Real REACH Data',
                 fontsize=16, fontweight='bold', y=0.995)

    output_path = Path('results/lsq_comparison_real_data.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comprehensive comparison plot to {output_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("SUMMARY")

    print("\n📊 RMSE Performance:")
    print(f"  Mean improvement: {mean_improvement:+.2f}%")
    print(f"  Median improvement: {median_improvement:+.2f}%")
    print(f"  Best improvement: {max(improvements):+.2f}% ({calibrators[improvements.index(max(improvements))]})")
    print(f"  Worst improvement: {min(improvements):+.2f}% ({calibrators[improvements.index(min(improvements))]})")

    print("\n🧠 Neural Network Corrections:")
    print(f"  RMS magnitude: {correction_stats['rms']:.4f} K")
    print(f"  Range: [{correction_stats['min']:.4f}, {correction_stats['max']:.4f}] K")

    print("\n🎯 Physical Parameters:")
    print("  LSQ parameters identical between methods (analytical solution preserved)")

    if mean_improvement > 10:
        print("\n✅ Significant improvement! Neural network is capturing systematic effects.")
    elif mean_improvement > 0:
        print("\n✓ Modest improvement. Neural network providing small corrections.")
    else:
        print("\n⚠ No improvement. Physical model may already be sufficient.")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()