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
    masked_data = loader.apply_frequency_mask(data, mask)

    print(f"Frequency range: {float(masked_data.vna_frequencies[0])/1e6:.1f} - "
          f"{float(masked_data.vna_frequencies[-1])/1e6:.1f} MHz")
    print(f"Number of calibrators: {len(masked_data.calibrator_names)}")

    # ========================================================================
    # Method 1: Pure Least Squares
    # ========================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: PURE LEAST SQUARES")
    print("=" * 70)

    print("\nFitting pure least squares model...")
    lsq_model = LeastSquaresModel({'regularisation': 0.0})
    lsq_model.fit(masked_data)
    lsq_result = lsq_model.get_result()
    print("✓ Fitting complete")

    print("\nGenerating calibration temperature plots...")
    plotter_lsq = CalibrationPlotter(output_dir=Path("results/pure_lsq"), save=True, show=False)
    plotter_lsq.plot_all_calibrators(
        masked_data,
        lsq_model,
        lsq_result,
        antenna_validation=True
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
    neural_model.fit(masked_data)
    neural_result = neural_model.get_result()
    print("✓ Fitting complete")

    print("\nGenerating calibration temperature plots...")
    plotter_neural = CalibrationPlotter(output_dir=Path("results/neural_corrected_lsq"), save=True, show=False)
    plotter_neural.plot_all_calibrators(
        masked_data,
        neural_model,
        neural_result,
        antenna_validation=True
    )
    print("✓ Saved to results/neural_corrected_lsq/calibrator_temperatures_*.png")

    print("\nGenerating neural network corrections plot...")
    plotter_neural.plot_neural_corrections(
        masked_data,
        neural_model
    )
    print("✓ Saved to results/neural_corrected_lsq/neural_corrections_*.png")

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  1. Pure LSQ:                results/pure_lsq/calibrator_temperatures_*.png")
    print("  2. Neural-Corrected LSQ:    results/neural_corrected_lsq/calibrator_temperatures_*.png")
    print("  3. Neural Corrections:      results/neural_corrected_lsq/neural_corrections_*.png")
    print("\nCompare plots 1 and 2 to see the improvement in fit quality!")
    print("Plot 3 shows the correction terms A(freq, Γ_cal) learned by the neural network.")


if __name__ == '__main__':
    main()