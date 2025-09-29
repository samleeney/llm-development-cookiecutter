#!/usr/bin/env python3
"""
Example script demonstrating least squares calibration with REACH data.

This script loads REACH observation data and performs calibration using
the least squares method to extract noise wave parameters.
"""

import sys
from pathlib import Path
import jax.numpy as jnp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import HDF5DataLoader, CalibrationData
from src.models.least_squares import LeastSquaresModel
from src.visualization.calibration_plots import CalibrationPlotter


def main():
    """Run least squares calibration on observation data."""
    obs_file = Path('data/test_observation.hdf5')

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

    # Apply frequency mask (50-130 MHz)
    print("\n2. Applying frequency mask (50-130 MHz)...")
    mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
    masked_data = loader.apply_frequency_mask(data, mask)
    print(f"   - Masked to {len(masked_data.vna_frequencies)} frequency points")

    # Filter to specific calibrators
    print("\n3. Filtering to specific calibrators...")
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
    print(f"   - Using {len(filtered_calibrators)} calibrators")

    # Create and fit model
    print("\n4. Fitting least squares model...")
    model = LeastSquaresModel({'regularisation': 0.0})
    model.fit(filtered_data)
    print("   - Fitting completed")

    # Print parameter statistics
    print("\n5. Noise wave parameter statistics:")
    params = model.get_parameters()
    param_names = {'u': 'Uncorrelated', 'c': 'Cosine', 's': 'Sine',
                   'NS': 'Noise Source', 'L': 'Load'}

    for key, name in param_names.items():
        values = params[key]
        print(f"   {name:15s} ({key:2s}): "
              f"mean={float(jnp.mean(values)):8.2f}, "
              f"std={float(jnp.std(values)):7.2f}")

    # Create visualisation plots
    print("\n6. Creating visualisation plots...")
    result = model.get_result()
    plotter = CalibrationPlotter(output_dir=Path("plots"), save=True, show=False)

    plotter.plot_all_calibrators(filtered_data, model, result, antenna_validation=False)
    plotter.plot_noise_parameters(filtered_data, model, param_smoothing=50)

    print("   - Plots saved to plots/ directory")
    print("\n✓ Calibration completed successfully!")


if __name__ == '__main__':
    main()