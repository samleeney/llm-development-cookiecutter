#!/usr/bin/env python3
"""
Analysis script for neural-corrected LSQ on real REACH observation data.

This script compares pure least squares with neural-corrected LSQ on real
observational data, where we expect to see larger corrections due to
unmodeled systematic effects.
"""

import sys
from pathlib import Path
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import HDF5DataLoader, CalibrationData
from src.models.neural_corrected_lsq import NeuralCorrectedLSQModel
from src.models.least_squares import LeastSquaresModel
from src.visualization.calibration_plots import CalibrationPlotter


def main():
    """Run neural-corrected LSQ analysis on real REACH observation data."""
    obs_file = Path('data/reach_observation.hdf5')

    if not obs_file.exists():
        print(f"ERROR: Real data file not found at {obs_file}")
        print("Please ensure reach_observation.hdf5 is in the data/ directory")
        return

    print("=" * 70)
    print("NEURAL-CORRECTED LSQ: REAL REACH OBSERVATION DATA ANALYSIS")
    print("=" * 70)

    # Load and prepare data
    print("\nLoading REACH observation data...")
    loader = HDF5DataLoader()
    data = loader.load_observation(str(obs_file))

    print(f"Loaded {len(data.calibrator_names)} calibrators: {', '.join(data.calibrator_names)}")
    print(f"Full frequency range: {float(data.vna_frequencies[0])/1e6:.1f} - "
          f"{float(data.vna_frequencies[-1])/1e6:.1f} MHz")

    # Apply frequency mask (50-130 MHz for cleaner results)
    print("\nApplying frequency mask (50-130 MHz)...")
    mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
    masked_data = loader.apply_frequency_mask(data, mask)
    print(f"Filtered to {len(masked_data.vna_frequencies)} frequency channels")

    # Determine which calibrators to use for fitting (exclude antenna)
    calibrators_for_fitting = [name for name in masked_data.calibrator_names if name != 'ant']
    print(f"\nCalibrators used for fitting: {', '.join(calibrators_for_fitting)}")

    # ========================================================================
    # PART 1: Pure Least Squares (Baseline)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 1: PURE LEAST SQUARES (BASELINE)")
    print("=" * 70)

    lsq_config = {
        'regularisation': 0.0,
        'use_gamma_weighting': False
    }
    lsq_model = LeastSquaresModel(lsq_config)
    lsq_model.fit(masked_data)
    lsq_result = lsq_model.get_result()

    print("\nPure LSQ - Calibration Source RMSE:")
    lsq_rmse_dict = {}
    for cal_name in calibrators_for_fitting:
        if cal_name in lsq_result.residuals:
            rmse = float(jnp.sqrt(jnp.mean(lsq_result.residuals[cal_name]**2)))
            lsq_rmse_dict[cal_name] = rmse
            print(f"  {cal_name:10s}: {rmse:8.4f} K")

    # Antenna RMSE (if available)
    if 'ant' in lsq_result.residuals:
        ant_lsq_rmse = float(jnp.sqrt(jnp.mean(lsq_result.residuals['ant']**2)))
        print(f"  {'ant':10s}: {ant_lsq_rmse:8.4f} K (validation)")
    else:
        ant_lsq_rmse = None

    # ========================================================================
    # PART 2: Neural-Corrected Least Squares
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 2: NEURAL-CORRECTED LEAST SQUARES")
    print("=" * 70)

    neural_config = {
        'regularisation': 0.0,
        'use_gamma_weighting': False,
        'hidden_layers': [64, 64, 32],  # Slightly larger for real data
        'learning_rate': 1e-3,
        'n_iterations': 2000,  # More iterations for real data
        'correction_regularization': 0.01
    }

    print("\nConfiguration:")
    print(f"  Hidden layers: {neural_config['hidden_layers']}")
    print(f"  Learning rate: {neural_config['learning_rate']}")
    print(f"  Training iterations: {neural_config['n_iterations']}")
    print(f"  Correction regularization: {neural_config['correction_regularization']}")

    neural_model = NeuralCorrectedLSQModel(neural_config)
    neural_model.fit(masked_data)
    neural_result = neural_model.get_result()

    print("\nNeural-Corrected LSQ - Calibration Source RMSE:")
    neural_rmse_dict = {}
    for cal_name in calibrators_for_fitting:
        if cal_name in neural_result.residuals:
            rmse = float(jnp.sqrt(jnp.mean(neural_result.residuals[cal_name]**2)))
            neural_rmse_dict[cal_name] = rmse
            print(f"  {cal_name:10s}: {rmse:8.4f} K")

    # Antenna RMSE (if available)
    if 'ant' in neural_result.residuals:
        ant_neural_rmse = float(jnp.sqrt(jnp.mean(neural_result.residuals['ant']**2)))
        print(f"  {'ant':10s}: {ant_neural_rmse:8.4f} K (validation)")
    else:
        ant_neural_rmse = None

    # ========================================================================
    # PART 3: Neural Network Correction Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 3: NEURAL NETWORK CORRECTION ANALYSIS")
    print("=" * 70)

    correction_stats = neural_model.get_correction_magnitude()
    print("\nCorrection Statistics (all calibrators):")
    print(f"  Mean:  {correction_stats['mean']:7.4f} K")
    print(f"  Std:   {correction_stats['std']:7.4f} K")
    print(f"  RMS:   {correction_stats['rms']:7.4f} K")
    print(f"  Range: [{correction_stats['min']:7.4f}, {correction_stats['max']:7.4f}] K")

    # ========================================================================
    # PART 4: Comparison and Improvement Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 4: COMPARISON - Pure LSQ vs Neural-Corrected LSQ")
    print("=" * 70)

    print("\nRMSE Comparison:")
    print(f"{'Calibrator':<12} {'Pure LSQ':>12} {'Neural LSQ':>12} {'Improvement':>12} {'Abs Change':>12}")
    print("-" * 72)

    improvements = []
    for cal_name in calibrators_for_fitting:
        if cal_name in lsq_rmse_dict and cal_name in neural_rmse_dict:
            lsq_rmse = lsq_rmse_dict[cal_name]
            neural_rmse = neural_rmse_dict[cal_name]
            abs_change = lsq_rmse - neural_rmse

            if lsq_rmse > 0:
                improvement = (abs_change / lsq_rmse) * 100
            else:
                improvement = 0.0

            improvements.append(improvement)

            print(f"{cal_name:<12} {lsq_rmse:12.4f} {neural_rmse:12.4f} "
                  f"{improvement:11.2f}% {abs_change:12.4f}K")

    if improvements:
        mean_improvement = sum(improvements) / len(improvements)
        print(f"\nMean RMSE improvement: {mean_improvement:+.2f}%")

    # Antenna comparison
    if ant_lsq_rmse is not None and ant_neural_rmse is not None:
        ant_improvement = ((ant_lsq_rmse - ant_neural_rmse) / ant_lsq_rmse) * 100
        print(f"\nAntenna (validation) RMSE improvement: {ant_improvement:+.2f}%")
        print(f"  Pure LSQ:    {ant_lsq_rmse:.4f} K")
        print(f"  Neural LSQ:  {ant_neural_rmse:.4f} K")

    # ========================================================================
    # PART 5: Detailed Correction Visualization
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 5: VISUALIZATION")
    print("=" * 70)

    # Create detailed correction plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Neural Network Corrections on Real REACH Data', fontsize=14, fontweight='bold')

    # Plot 1: Correction magnitude vs frequency for each calibrator
    ax = axes[0, 0]
    for cal_name in calibrators_for_fitting:
        cal_data = masked_data.get_calibrator(cal_name)
        s11_interp = neural_model._interpolate_s11(
            cal_data.s11_freq,
            cal_data.s11_complex,
            masked_data.psd_frequencies
        )

        freq_norm = (masked_data.psd_frequencies - neural_model._freq_mean) / (neural_model._freq_std + 1e-10)
        features = jnp.stack([
            freq_norm,
            jnp.abs(s11_interp),
            jnp.real(s11_interp),
            jnp.imag(s11_interp)
        ], axis=1)

        corrections = neural_model._nn_state.apply(neural_model._nn_params, features)
        ax.plot(masked_data.psd_frequencies / 1e6, corrections, label=cal_name, alpha=0.7)

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Correction A(freq, Γ) (K)')
    ax.set_title('Neural Network Corrections by Calibrator')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: RMSE comparison bar chart
    ax = axes[0, 1]
    x = range(len(calibrators_for_fitting))
    width = 0.35
    lsq_rmses = [lsq_rmse_dict.get(cal, 0) for cal in calibrators_for_fitting]
    neural_rmses = [neural_rmse_dict.get(cal, 0) for cal in calibrators_for_fitting]

    ax.bar([i - width/2 for i in x], lsq_rmses, width, label='Pure LSQ', alpha=0.8)
    ax.bar([i + width/2 for i in x], neural_rmses, width, label='Neural LSQ', alpha=0.8)
    ax.set_xlabel('Calibrator')
    ax.set_ylabel('RMSE (K)')
    ax.set_title('RMSE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(calibrators_for_fitting, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Residual improvement (LSQ residuals - Neural LSQ residuals)
    ax = axes[1, 0]
    for cal_name in calibrators_for_fitting[:5]:  # Show first 5 for clarity
        if cal_name in lsq_result.residuals and cal_name in neural_result.residuals:
            improvement = lsq_result.residuals[cal_name] - neural_result.residuals[cal_name]
            ax.plot(masked_data.psd_frequencies / 1e6, improvement, label=cal_name, alpha=0.7)

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Residual Reduction (K)')
    ax.set_title('Residual Improvement (Pure LSQ - Neural LSQ)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)

    # Plot 4: Correction vs |Gamma| scatter
    ax = axes[1, 1]
    all_gamma_mag = []
    all_corrections = []

    for cal_name in calibrators_for_fitting:
        cal_data = masked_data.get_calibrator(cal_name)
        s11_interp = neural_model._interpolate_s11(
            cal_data.s11_freq,
            cal_data.s11_complex,
            masked_data.psd_frequencies
        )

        freq_norm = (masked_data.psd_frequencies - neural_model._freq_mean) / (neural_model._freq_std + 1e-10)
        features = jnp.stack([
            freq_norm,
            jnp.abs(s11_interp),
            jnp.real(s11_interp),
            jnp.imag(s11_interp)
        ], axis=1)

        corrections = neural_model._nn_state.apply(neural_model._nn_params, features)
        all_gamma_mag.extend(jnp.abs(s11_interp).tolist())
        all_corrections.extend(corrections.tolist())

    ax.scatter(all_gamma_mag, all_corrections, alpha=0.3, s=1)
    ax.set_xlabel('|Γ_cal|')
    ax.set_ylabel('Correction A(freq, Γ) (K)')
    ax.set_title('Correction vs Reflection Coefficient Magnitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path('results/neural_lsq_real_data_analysis.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved detailed analysis plot to {output_path}")

    # Create standard calibration plots
    print("\nCreating standard calibration plots...")
    plotter = CalibrationPlotter(output_dir=Path("results"), save=True, show=False)

    # Plot all calibrators with neural-corrected model
    plotter.plot_all_calibrators(masked_data, neural_model, neural_result, antenna_validation=True)
    plotter.plot_noise_parameters(masked_data, neural_model, param_smoothing=50)

    print("\nAll plots saved to results/ directory")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nCorrection magnitude: RMS = {correction_stats['rms']:.4f} K")

    if correction_stats['rms'] > 0.1:
        print("✓ Significant corrections detected - neural network is capturing systematic effects")
    else:
        print("✓ Small corrections - physical model is already quite good")

    if improvements and mean_improvement > 0:
        print(f"✓ Mean RMSE improvement: {mean_improvement:.2f}%")
    elif improvements and mean_improvement < 0:
        print(f"⚠ Mean RMSE change: {mean_improvement:.2f}% (slight increase)")
        print("  This may indicate overfitting or that more training is needed")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()