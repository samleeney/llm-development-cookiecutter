#!/usr/bin/env python3
"""
Generate standard calibration temperature plots for both methods on real REACH data.

Creates side-by-side comparison of the standard calibration plots for:
- Pure Least Squares
- Neural-Corrected Least Squares
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import HDF5DataLoader
from src.models.least_squares import LeastSquaresModel
from src.models.neural_corrected_lsq import NeuralCorrectedLSQModel
from src.visualization.calibration_plots import CalibrationPlotter


def main():
    """Generate calibration temperature plots for both methods."""
    obs_file = Path('data/reach_observation.hdf5')

    if not obs_file.exists():
        print(f"ERROR: Real data file not found at {obs_file}")
        return

    print("=" * 70)
    print("GENERATING CALIBRATION TEMPERATURE PLOTS")
    print("=" * 70)

    # Load and prepare data
    print("\nLoading REACH observation data...")
    loader = HDF5DataLoader()
    data = loader.load_observation(str(obs_file))

    # Apply frequency mask (50-130 MHz)
    mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
    masked_data_full = loader.apply_frequency_mask(data, mask)

    # Filter data: use c2r91 as validation (exclude from training, like antenna)
    # Remove ant entirely for now
    from src.data import CalibrationData
    calibrators_for_analysis = [name for name in masked_data_full.calibrator_names
                                if name != 'ant']

    filtered_calibrators = {
        name: cal for name, cal in masked_data_full.calibrators.items()
        if name in calibrators_for_analysis
    }

    masked_data = CalibrationData(
        calibrators=filtered_calibrators,
        psd_frequencies=masked_data_full.psd_frequencies,
        vna_frequencies=masked_data_full.vna_frequencies,
        lna_s11=masked_data_full.lna_s11,
        metadata=masked_data_full.metadata
    )

    print(f"Frequency range: {float(masked_data.vna_frequencies[0])/1e6:.1f} - "
          f"{float(masked_data.vna_frequencies[-1])/1e6:.1f} MHz")
    print(f"Total calibrators: {len(masked_data.calibrator_names)}")
    print(f"Validation calibrator: c2r91 (excluded from training)")
    print(f"Training calibrators: {len([n for n in masked_data.calibrator_names if n != 'c2r91'])}")

    # ========================================================================
    # Method 1: Pure Least Squares
    # ========================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: PURE LEAST SQUARES")
    print("=" * 70)

    print("\nFitting pure least squares model...")
    # Exclude c2r91 from training (use as validation)
    training_calibrators = {
        name: cal for name, cal in masked_data.calibrators.items()
        if name != 'c2r91'
    }
    training_data = CalibrationData(
        calibrators=training_calibrators,
        psd_frequencies=masked_data.psd_frequencies,
        vna_frequencies=masked_data.vna_frequencies,
        lna_s11=masked_data.lna_s11,
        metadata=masked_data.metadata
    )

    lsq_model = LeastSquaresModel({'regularisation': 0.0})
    lsq_model.fit(training_data)
    lsq_result = lsq_model.get_result()

    # Add c2r91 validation predictions
    # Temporarily extend model's data reference for prediction
    lsq_model._data = masked_data
    c2r91_pred = lsq_model.predict(masked_data.psd_frequencies, 'c2r91')
    c2r91_true = masked_data.calibrators['c2r91'].temperature
    lsq_result.predicted_temperatures['c2r91'] = c2r91_pred
    lsq_result.residuals['c2r91'] = c2r91_pred - c2r91_true
    # Restore training data reference
    lsq_model._data = training_data

    print("✓ Fitting complete")

    print("\nGenerating calibration temperature plots...")
    # Use full data for plotting (includes c2r91 validation)
    lsq_model._data = masked_data
    plotter_lsq = CalibrationPlotter(output_dir=Path("results/pure_lsq"), save=True, show=False)
    plotter_lsq.plot_all_calibrators(
        masked_data,
        lsq_model,
        lsq_result,
        antenna_validation=False
    )
    print("✓ Saved to results/pure_lsq/calibrator_temperatures_*.png")

    # ========================================================================
    # Method 2: Neural-Corrected Least Squares
    # ========================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: NEURAL-CORRECTED LEAST SQUARES")
    print("=" * 70)

    neural_config = {
        'regularisation': 0.0,
        'hidden_layers': [64, 64, 32],
        'learning_rate': 1e-3,
        'n_iterations': 2000,
        'correction_regularization': 0.01
    }

    print("\nFitting neural-corrected least squares model...")
    neural_model = NeuralCorrectedLSQModel(neural_config)
    neural_model.fit(training_data)
    neural_result = neural_model.get_result()

    # Add c2r91 validation predictions
    # Temporarily extend model's data reference for prediction
    neural_model._data = masked_data
    c2r91_pred_neural = neural_model.predict(masked_data.psd_frequencies, 'c2r91')
    c2r91_true = masked_data.calibrators['c2r91'].temperature
    neural_result.predicted_temperatures['c2r91'] = c2r91_pred_neural
    neural_result.residuals['c2r91'] = c2r91_pred_neural - c2r91_true
    # Restore training data reference
    neural_model._data = training_data

    print("✓ Fitting complete")

    print("\nGenerating calibration temperature plots...")
    # Use full data for plotting (includes c2r91)
    neural_model._data = masked_data
    plotter_neural = CalibrationPlotter(output_dir=Path("results/neural_corrected_lsq"), save=True, show=False)
    plotter_neural.plot_all_calibrators(
        masked_data,
        neural_model,
        neural_result,
        antenna_validation=False
    )
    print("✓ Saved to results/neural_corrected_lsq/calibrator_temperatures_*.png")

    print("\nGenerating neural network corrections plot...")
    plotter_neural.plot_neural_corrections(
        masked_data,
        neural_model
    )
    print("✓ Saved to results/neural_corrected_lsq/neural_corrections_*.png")

    print("\nGenerating Fourier transform of corrections plot...")
    plotter_neural.plot_correction_fourier_transforms(
        masked_data,
        neural_model
    )
    print("✓ Saved to results/neural_corrected_lsq/correction_fft_*.png")

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  1. Pure LSQ:                results/pure_lsq/calibrator_temperatures_*.png")
    print("  2. Neural-Corrected LSQ:    results/neural_corrected_lsq/calibrator_temperatures_*.png")
    print("  3. Neural Corrections:      results/neural_corrected_lsq/neural_corrections_*.png")
    print("  4. Correction FFT:          results/neural_corrected_lsq/correction_fft_*.png")
    print("\nPlots 1 & 2: Compare to see improvement in fit quality")
    print("Plot 3:      Correction terms A(freq, Γ_cal) in frequency domain")
    print("Plot 4:      FFT of corrections reveals periodic systematic effects")


if __name__ == '__main__':
    main()