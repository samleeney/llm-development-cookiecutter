#!/usr/bin/env python3
"""
Example script demonstrating how to load and explore REACH observation data.

This script shows the basic usage of the HDF5DataLoader to load calibration
data and access various components.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import src module
sys.path.append(str(Path(__file__).parent.parent))

from src.data import HDF5DataLoader


def main():
    """Main example function."""
    # Path to observation file
    obs_file = Path('data/reach_observation.hdf5')

    if not obs_file.exists():
        print(f"Error: Observation file not found at {obs_file}")
        print("Please ensure the data file is in the correct location")
        return

    print("=" * 60)
    print("REACH Observation Data Loader Example")
    print("=" * 60)

    # Create data loader
    print("\n1. Creating HDF5 data loader...")
    loader = HDF5DataLoader()

    # Load observation data
    print(f"\n2. Loading observation from {obs_file}...")
    data = loader.load_observation(str(obs_file))

    # Display basic information
    print(f"\n3. Observation Summary:")
    print(f"   - Number of calibrators: {len(data.calibrators)}")
    print(f"   - Calibrator names: {', '.join(data.calibrator_names)}")
    print(f"   - PSD frequency channels: {len(data.psd_frequencies)}")
    print(f"   - PSD frequency range: {data.psd_frequencies[0]/1e6:.1f} - "
          f"{data.psd_frequencies[-1]/1e6:.1f} MHz")
    print(f"   - VNA frequency points: {len(data.vna_frequencies)}")
    print(f"   - VNA frequency range: {data.vna_frequencies[0]/1e6:.1f} - "
          f"{data.vna_frequencies[-1]/1e6:.1f} MHz")

    # Access specific calibrator
    print("\n4. Accessing 'hot' calibrator data:")
    if 'hot' in data.calibrator_names:
        hot_cal = data.get_calibrator('hot')
        print(f"   - Name: {hot_cal.name}")
        print(f"   - PSD data shape: {hot_cal.psd_source.shape}")
        print(f"   - Number of time samples: {hot_cal.n_time}")
        print(f"   - S11 data points: {hot_cal.n_freq_vna}")
        print(f"   - Timestamp shape: {hot_cal.timestamps.shape}")

        # Calculate mean power levels
        mean_source = np.mean(hot_cal.psd_source)
        mean_load = np.mean(hot_cal.psd_load)
        mean_ns = np.mean(hot_cal.psd_ns)
        print(f"\n   Mean power levels:")
        print(f"   - Source: {mean_source:.2e}")
        print(f"   - Load: {mean_load:.2e}")
        print(f"   - Noise source: {mean_ns:.2e}")

    # Display metadata
    print("\n5. Observation Metadata:")
    print(f"   - Observation name: {data.metadata.get('observation_name', 'N/A')}")
    print(f"   - Start time: {data.metadata.get('start_time', 'N/A')}")
    print(f"   - HDF5 file: {data.metadata.get('hdf5_filename', 'N/A')}")

    # Apply frequency mask
    print("\n6. Applying frequency mask (50-130 MHz):")
    mask = (data.vna_frequencies >= 50e6) & (data.vna_frequencies <= 130e6)
    masked_data = loader.apply_frequency_mask(data, mask)
    print(f"   - Original VNA points: {len(data.vna_frequencies)}")
    print(f"   - Masked VNA points: {len(masked_data.vna_frequencies)}")
    print(f"   - New frequency range: {masked_data.vna_frequencies[0]/1e6:.1f} - "
          f"{masked_data.vna_frequencies[-1]/1e6:.1f} MHz")

    # Plot example data
    print("\n7. Creating visualization plots...")
    create_plots(data)

    print("\n✓ Example completed successfully!")


def create_plots(data):
    """Create visualization plots for the loaded data."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('REACH Observation Data Overview', fontsize=14)

    # Plot 1: PSD spectra for hot calibrator
    if 'hot' in data.calibrator_names:
        ax = axes[0, 0]
        hot_cal = data.get_calibrator('hot')

        # Average over time for cleaner plot
        mean_source = np.mean(hot_cal.psd_source, axis=0)
        mean_load = np.mean(hot_cal.psd_load, axis=0)
        mean_ns = np.mean(hot_cal.psd_ns, axis=0)

        freq_mhz = data.psd_frequencies / 1e6
        ax.semilogy(freq_mhz, mean_source, label='Source', alpha=0.7)
        ax.semilogy(freq_mhz, mean_load, label='Load', alpha=0.7)
        ax.semilogy(freq_mhz, mean_ns, label='Noise Source', alpha=0.7)

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Hot Calibrator PSD (Time-Averaged)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 200)

    # Plot 2: S11 parameters
    if 'hot' in data.calibrator_names:
        ax = axes[0, 1]
        hot_cal = data.get_calibrator('hot')

        freq_mhz = hot_cal.s11_freq / 1e6
        s11_mag = np.abs(hot_cal.s11_complex)
        s11_phase = np.angle(hot_cal.s11_complex, deg=True)

        color = 'tab:blue'
        ax.plot(freq_mhz, s11_mag, color=color, label='|S11|')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('|S11|', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(50, 200)

        ax2 = ax.twinx()
        color = 'tab:orange'
        ax2.plot(freq_mhz, s11_phase, color=color, alpha=0.7, label='Phase')
        ax2.set_ylabel('Phase (degrees)', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax.set_title('S11 Parameters (Hot Calibrator)')

    # Plot 3: Calibrator comparison
    ax = axes[1, 0]
    calibrators_to_plot = ['hot', 'cold', 'ant']

    for cal_name in calibrators_to_plot:
        if cal_name in data.calibrator_names:
            cal = data.get_calibrator(cal_name)
            mean_psd = np.mean(cal.psd_source, axis=0)
            ax.semilogy(data.psd_frequencies / 1e6, mean_psd,
                       label=cal_name, alpha=0.7)

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Calibrator PSD Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)

    # Plot 4: Time variation
    if 'ant' in data.calibrator_names:
        ax = axes[1, 1]
        ant_cal = data.get_calibrator('ant')

        # Select a few frequency channels to plot
        freq_indices = [1000, 4000, 8000, 12000]
        for idx in freq_indices:
            freq_val = data.psd_frequencies[idx] / 1e6
            time_series = ant_cal.psd_source[:, idx]
            ax.plot(range(len(time_series)), time_series,
                   label=f'{freq_val:.1f} MHz', alpha=0.7)

        ax.set_xlabel('Time Sample')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title('Antenna PSD Time Variation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('observation_overview.png', dpi=150, bbox_inches='tight')
    print("   Plots saved to 'observation_overview.png'")
    plt.show()


if __name__ == '__main__':
    main()