"""
Visualization module for calibration results.

This module provides comprehensive plotting functionality for calibration results,
including temperature predictions, residuals, noise parameters, and validation plots.
Based on the REACH calibration pipeline plotting approach.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


class CalibrationPlotter:
    """
    Comprehensive plotting class for calibration results.

    Provides methods to visualize:
    - Temperature predictions for all calibrators
    - Residuals from target temperatures
    - Noise wave parameters
    - Validation metrics
    """

    def __init__(self,
                 output_dir: Optional[Path] = None,
                 show: bool = False,
                 save: bool = True,
                 dpi: int = 300):
        """
        Initialize the calibration plotter.

        Args:
            output_dir: Directory to save plots
            show: Whether to display plots
            save: Whether to save plots to disk
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.show = show
        self.save = save
        self.dpi = dpi
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Ensure output directory exists
        if self.save:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_filename(self, description: str) -> Path:
        """Generate timestamped filename following YYYY-MM-DD_HH-MM-SS_description.ext format."""
        return self.output_dir / f"{self.timestamp}_{description}.png"

    def plot_all_calibrators(self,
                            data,
                            model,
                            result,
                            antenna_validation: bool = True,
                            bin_size: int = 100):
        """
        Plot temperature predictions for all calibrators in a grid.

        Args:
            data: CalibrationData object
            model: Fitted model
            result: CalibrationResult object
            antenna_validation: Whether to show antenna validation
            bin_size: Size for binning/smoothing
        """
        # Get list of calibrators
        cal_names = sorted(data.calibrators.keys())
        n_cals = len(cal_names)

        # Determine grid layout
        width = 4
        height = (n_cals + width - 1) // width

        fig, axes = plt.subplots(height, width,
                                 figsize=(width*4, height*3.5),
                                 dpi=self.dpi)

        # Flatten axes array for easier iteration
        if height == 1:
            axes = axes.reshape(1, -1)

        freq_mhz = data.psd_frequencies / 1e6

        for idx, cal_name in enumerate(cal_names):
            row = idx // width
            col = idx % width
            ax = axes[row, col]

            # Get predictions and measurements
            if cal_name in result.predicted_temperatures:
                T_pred = result.predicted_temperatures[cal_name]
                # Get measured temperature from calibrator data
                cal_data = data.calibrators[cal_name]
                T_meas = cal_data.temperature if cal_data.temperature is not None else 0.0
                residuals = result.residuals[cal_name]

                # Calculate RMSE
                rmse = float(jnp.sqrt(jnp.mean(residuals**2)))

                # Plot scatter points
                ax.scatter(freq_mhz, T_pred, s=0.5, alpha=0.5,
                          color='C0', label=f'Predicted')

                # Add smoothed line
                if len(T_pred) > bin_size:
                    smoothed = jnp.convolve(T_pred, jnp.ones(bin_size)/bin_size, mode='valid')
                    smooth_freq = jnp.convolve(freq_mhz, jnp.ones(bin_size)/bin_size, mode='valid')
                    ax.plot(smooth_freq, smoothed, 'k-', linewidth=1, alpha=0.8)

                # Add target temperature line if not antenna
                if cal_name != 'ant' or antenna_validation:
                    ax.axhline(y=float(T_meas), color='r', linestyle='--',
                              label=f'Target: {float(T_meas):.1f} K')

                # Add title with RMSE
                if cal_name == 'ant' and not antenna_validation:
                    ax.set_title(f'{cal_name.upper()} (Validation)')
                else:
                    ax.set_title(f'{cal_name.upper()} - RMSE: {rmse:.2f} K')

                ax.set_xlabel('Frequency (MHz)')
                ax.set_ylabel('Temperature (K)')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{cal_name}\n(No data)',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(cal_name.upper())

        # Hide unused subplots
        for idx in range(n_cals, height * width):
            row = idx // width
            col = idx % width
            axes[row, col].axis('off')

        plt.suptitle('Calibration Temperature Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if self.save:
            filepath = self._get_filename('calibrator_temperatures')
            plt.savefig(filepath, bbox_inches='tight')

        if self.show:
            plt.show()

        plt.close()

    def plot_neural_corrections(self,
                               data,
                               neural_model,
                               bin_size: int = 100):
        """
        Plot neural network corrections for all calibrators in a grid.

        This method extracts and visualizes the correction term A(freq, Γ_cal)
        learned by the neural network for each calibrator.

        Args:
            data: CalibrationData object
            neural_model: Fitted NeuralCorrectedLSQModel
            bin_size: Size for binning/smoothing
        """
        # Check if model has neural network parameters
        if not hasattr(neural_model, '_nn_params') or neural_model._nn_params is None:
            raise ValueError("Model must be a fitted NeuralCorrectedLSQModel")

        # Get list of calibrators
        cal_names = sorted(data.calibrators.keys())
        n_cals = len(cal_names)

        # Determine grid layout
        width = 4
        height = (n_cals + width - 1) // width

        fig, axes = plt.subplots(height, width,
                                 figsize=(width*4, height*3.5),
                                 dpi=self.dpi)

        # Flatten axes array for easier iteration
        if height == 1:
            axes = axes.reshape(1, -1)

        freq_mhz = data.psd_frequencies / 1e6

        # Compute corrections for each calibrator
        all_corrections = []
        for cal_name in cal_names:
            cal_data = data.calibrators[cal_name]
            s11_interp = neural_model._interpolate_s11(
                cal_data.s11_freq,
                cal_data.s11_complex,
                data.psd_frequencies
            )

            freq_norm = (data.psd_frequencies - neural_model._freq_mean) / (neural_model._freq_std + 1e-10)
            features = jnp.stack([
                freq_norm,
                jnp.abs(s11_interp),    # Magnitude of S11
                jnp.angle(s11_interp)   # Phase of S11
            ], axis=1)

            corrections = neural_model._nn_state.apply(neural_model._nn_params, features, deterministic=True)
            all_corrections.append(corrections)

        # Plot each calibrator
        for idx, cal_name in enumerate(cal_names):
            row = idx // width
            col = idx % width
            ax = axes[row, col]

            # Get correction for this calibrator
            corrections = all_corrections[idx]

            # Calculate statistics
            mean_corr = float(jnp.mean(corrections))
            std_corr = float(jnp.std(corrections))
            rms_corr = float(jnp.sqrt(jnp.mean(corrections**2)))

            # Plot scatter points
            ax.scatter(freq_mhz, corrections, s=0.5, alpha=0.5,
                      color='C1', label='Correction')

            # Add smoothed line
            if len(corrections) > bin_size:
                smoothed = jnp.convolve(corrections, jnp.ones(bin_size)/bin_size, mode='valid')
                smooth_freq = jnp.convolve(freq_mhz, jnp.ones(bin_size)/bin_size, mode='valid')
                ax.plot(smooth_freq, smoothed, 'k-', linewidth=1.5, alpha=0.8)

            # Add zero line for reference
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

            # Add title with statistics
            ax.set_title(f'{cal_name.upper()} - RMS: {rms_corr:.3f} K\n'
                        f'Mean: {mean_corr:+.3f} K, Std: {std_corr:.3f} K',
                        fontsize=9)

            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Correction (K)')
            ax.grid(True, alpha=0.3)

            # Use individual y-axis scaling for each calibrator (no margin)
            local_min = float(jnp.min(corrections))
            local_max = float(jnp.max(corrections))
            ax.set_ylim(local_min, local_max)

        # Hide unused subplots
        for idx in range(n_cals, height * width):
            row = idx // width
            col = idx % width
            axes[row, col].axis('off')

        plt.suptitle('Neural Network Corrections A(freq, Γ_cal)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if self.save:
            filepath = self._get_filename('neural_corrections')
            plt.savefig(filepath, bbox_inches='tight')

        if self.show:
            plt.show()

        plt.close()

    def plot_correction_fourier_transforms(self,
                                          data,
                                          neural_model):
        """
        Plot Fourier transforms of neural network corrections for all calibrators.

        This reveals the frequency content of the learned corrections, helping
        identify periodic systematic effects.

        Args:
            data: CalibrationData object
            neural_model: Fitted NeuralCorrectedLSQModel
        """
        # Check if model has neural network parameters
        if not hasattr(neural_model, '_nn_params') or neural_model._nn_params is None:
            raise ValueError("Model must be a fitted NeuralCorrectedLSQModel")

        # Get list of calibrators
        cal_names = sorted(data.calibrators.keys())
        n_cals = len(cal_names)

        # Determine grid layout
        width = 4
        height = (n_cals + width - 1) // width

        fig, axes = plt.subplots(height, width,
                                 figsize=(width*4, height*3.5),
                                 dpi=self.dpi)

        # Flatten axes array for easier iteration
        if height == 1:
            axes = axes.reshape(1, -1)

        # Plot each calibrator
        for idx, cal_name in enumerate(cal_names):
            row = idx // width
            col = idx % width
            ax = axes[row, col]

            # Get correction for this calibrator
            cal_data = data.calibrators[cal_name]
            s11_interp = neural_model._interpolate_s11(
                cal_data.s11_freq,
                cal_data.s11_complex,
                data.psd_frequencies
            )

            freq_norm = (data.psd_frequencies - neural_model._freq_mean) / (neural_model._freq_std + 1e-10)
            features = jnp.stack([
                freq_norm,
                jnp.abs(s11_interp),    # Magnitude of S11
                jnp.angle(s11_interp)   # Phase of S11
            ], axis=1)

            corrections = neural_model._nn_state.apply(neural_model._nn_params, features, deterministic=True)

            # Compute FFT
            fft_values = jnp.fft.fft(corrections)
            fft_magnitude = jnp.abs(fft_values)

            # Frequency axis for FFT
            n_samples = len(corrections)
            freq_spacing = float(data.psd_frequencies[1] - data.psd_frequencies[0])
            fft_freqs = jnp.fft.fftfreq(n_samples, freq_spacing)

            # Only plot positive frequencies
            positive_freq_mask = fft_freqs >= 0
            fft_freqs_positive = fft_freqs[positive_freq_mask]
            fft_magnitude_positive = fft_magnitude[positive_freq_mask]

            # Convert to period (1/frequency) for interpretability
            # Avoid division by zero for DC component
            periods = jnp.where(fft_freqs_positive > 0, 1.0 / fft_freqs_positive, jnp.inf)

            # Plot magnitude spectrum vs period
            ax.semilogy(periods / 1e6, fft_magnitude_positive, linewidth=1, alpha=0.8)

            # Calculate dominant period (excluding DC)
            if len(fft_magnitude_positive) > 1:
                dominant_idx = jnp.argmax(fft_magnitude_positive[1:]) + 1  # Skip DC
                dominant_period = float(periods[dominant_idx]) / 1e6  # MHz
                dominant_mag = float(fft_magnitude_positive[dominant_idx])

                if not jnp.isinf(dominant_period):
                    ax.axvline(dominant_period, color='r', linestyle='--',
                              alpha=0.5, linewidth=0.8)
                    ax.text(0.98, 0.95, f'Peak: {dominant_period:.1f} MHz',
                           transform=ax.transAxes, ha='right', va='top',
                           fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Period (MHz)')
            ax.set_ylabel('FFT Magnitude')
            ax.set_title(f'{cal_name.upper()}')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim(left=float(1.0 / fft_freqs_positive[-1]) / 1e6 if fft_freqs_positive[-1] > 0 else 0.1)

        # Hide unused subplots
        for idx in range(n_cals, height * width):
            row = idx // width
            col = idx % width
            axes[row, col].axis('off')

        plt.suptitle('Fourier Transform of Neural Network Corrections',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if self.save:
            filepath = self._get_filename('correction_fft')
            plt.savefig(filepath, bbox_inches='tight')

        if self.show:
            plt.show()

        plt.close()

    def plot_noise_parameters(self,
                             data,
                             model,
                             param_smoothing: Optional[int] = None):
        """
        Plot the LNA noise wave parameters.

        Args:
            data: CalibrationData object
            model: Fitted model
            param_smoothing: Number of points for smoothing
        """
        params = model.get_parameters()
        freq_mhz = data.psd_frequencies / 1e6

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=self.dpi)

        # Parameter labels and colors
        param_info = [
            ('u', 'Uncorrelated', 'C0'),
            ('c', 'Cosine', 'C1'),
            ('s', 'Sine', 'C2'),
            ('NS', 'Noise Source', 'C3'),
            ('L', 'Load', 'C4'),
        ]

        for idx, (key, label, color) in enumerate(param_info):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            values = params[key]

            # Apply smoothing if requested
            if param_smoothing and len(values) > param_smoothing:
                smoothed = jnp.convolve(values,
                                        jnp.ones(param_smoothing)/param_smoothing,
                                        mode='valid')
                smooth_freq = freq_mhz[param_smoothing//2:-(param_smoothing//2)+1]
                ax.plot(smooth_freq, smoothed, color=color, linewidth=2,
                       label=f'{label} (smoothed)')
                ax.plot(freq_mhz, values, color=color, alpha=0.3, linewidth=0.5)
            else:
                ax.plot(freq_mhz, values, color=color, linewidth=1.5, label=label)

            # Add statistics
            mean_val = float(jnp.mean(values))
            std_val = float(jnp.std(values))
            ax.text(0.05, 0.95, f'μ={mean_val:.1f} K\nσ={std_val:.1f} K',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Temperature (K)')
            ax.set_title(f'{label} ({key})')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

        # Hide the last subplot if we only have 5 parameters
        axes[1, 2].axis('off')

        plt.suptitle('Noise Wave Parameters', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if self.save:
            filepath = self._get_filename('noise_parameters')
            plt.savefig(filepath, bbox_inches='tight')

        if self.show:
            plt.show()

        plt.close()

    def plot_residuals_summary(self, result):
        """
        Plot summary of residuals for all calibrators.

        Args:
            result: CalibrationResult object
        """
        cal_names = sorted(result.residuals.keys())

        # Skip antenna if present
        cal_names = [c for c in cal_names if c != 'ant']

        # Calculate statistics
        rmse_values = []
        mean_values = []
        std_values = []

        for cal_name in cal_names:
            residuals = result.residuals[cal_name]
            rmse_values.append(float(jnp.sqrt(jnp.mean(residuals**2))))
            mean_values.append(float(jnp.mean(residuals)))
            std_values.append(float(jnp.std(residuals)))

        # Create bar plots
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=self.dpi)

        x_pos = np.arange(len(cal_names))

        # RMSE plot
        axes[0].bar(x_pos, rmse_values, color='C0', alpha=0.7)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(cal_names, rotation=45, ha='right')
        axes[0].set_ylabel('RMSE (K)')
        axes[0].set_title('Root Mean Square Error')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Mean residual plot
        axes[1].bar(x_pos, mean_values, color='C1', alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(cal_names, rotation=45, ha='right')
        axes[1].set_ylabel('Mean Residual (K)')
        axes[1].set_title('Mean Residual (Bias)')
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')

        # Std deviation plot
        axes[2].bar(x_pos, std_values, color='C2', alpha=0.7)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(cal_names, rotation=45, ha='right')
        axes[2].set_ylabel('Std Deviation (K)')
        axes[2].set_title('Residual Standard Deviation')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Calibration Residuals Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if self.save:
            filepath = self._get_filename('residuals_summary')
            plt.savefig(filepath, bbox_inches='tight')

        if self.show:
            plt.show()

        plt.close()

    def plot_antenna_temperature(self,
                                 data,
                                 result,
                                 ylim: Optional[Tuple[float, float]] = None):
        """
        Plot dedicated antenna temperature prediction.

        Args:
            data: CalibrationData object
            result: CalibrationResult object
            ylim: Optional y-axis limits
        """
        if 'ant' not in result.predicted_temperatures:
            print("No antenna data available")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       dpi=self.dpi, sharex=True)

        freq_mhz = data.psd_frequencies / 1e6
        T_pred = result.predicted_temperatures['ant']

        # Top panel: Temperature prediction
        ax1.scatter(freq_mhz, T_pred, s=0.5, alpha=0.3, color='C0')

        # Add smoothed line
        bin_size = 100
        if len(T_pred) > bin_size:
            smoothed = jnp.convolve(T_pred, jnp.ones(bin_size)/bin_size, mode='valid')
            smooth_freq = jnp.convolve(freq_mhz, jnp.ones(bin_size)/bin_size, mode='valid')
            ax1.plot(smooth_freq, smoothed, 'k-', linewidth=1.5,
                    label='Smoothed (100 pt)')

        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Antenna Temperature Prediction')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        if ylim:
            ax1.set_ylim(ylim)

        # Bottom panel: Residuals if validation data available
        if 'ant' in result.residuals:
            residuals = result.residuals['ant']

            ax2.scatter(freq_mhz, residuals, s=0.5, alpha=0.3, color='C1')
            ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

            # Add RMS line
            rmse = float(jnp.sqrt(jnp.mean(residuals**2)))
            ax2.axhline(y=rmse, color='r', linestyle='--',
                       label=f'RMSE: {rmse:.2f} K')
            ax2.axhline(y=-rmse, color='r', linestyle='--')

            ax2.set_ylabel('Residual (K)')
            ax2.set_title('Antenna Temperature Residuals')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No validation data available',
                    ha='center', va='center', transform=ax2.transAxes)

        ax2.set_xlabel('Frequency (MHz)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save:
            filepath = self._get_filename('antenna_temperature')
            plt.savefig(filepath, bbox_inches='tight')

        if self.show:
            plt.show()

        plt.close()

    def create_summary_plot(self, data, model, result):
        """
        Create a comprehensive summary plot with all key results.

        Args:
            data: CalibrationData object
            model: Fitted model
            result: CalibrationResult object
        """
        fig = plt.figure(figsize=(16, 10), dpi=self.dpi)
        gs = GridSpec(3, 4, figure=fig)

        freq_mhz = data.psd_frequencies / 1e6
        params = model.get_parameters()

        # Top row: Noise parameters
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(freq_mhz, params['u'], label='u (uncorrelated)', alpha=0.8)
        ax1.plot(freq_mhz, params['c'], label='c (cosine)', alpha=0.8)
        ax1.plot(freq_mhz, params['s'], label='s (sine)', alpha=0.8)
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Correlated/Uncorrelated Noise Parameters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(freq_mhz, params['NS'], label='NS (Noise Source)', color='C3', alpha=0.8)
        ax2.plot(freq_mhz, params['L'], label='L (Load)', color='C4', alpha=0.8)
        ax2.set_ylabel('Temperature (K)')
        ax2.set_title('Noise Source & Load Parameters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Middle row: Key calibrator fits
        for idx, cal_name in enumerate(['hot', 'cold', 'r100', 'c2r36'][:4]):
            ax = fig.add_subplot(gs[1, idx])

            if cal_name in result.predicted_temperatures and cal_name in data.calibrators:
                T_pred = result.predicted_temperatures[cal_name]
                cal_data = data.calibrators[cal_name]
                T_meas = cal_data.temperature if cal_data.temperature is not None else 0.0

                ax.scatter(freq_mhz[::10], T_pred[::10], s=1, alpha=0.5, color='C0')
                ax.axhline(y=float(T_meas), color='r', linestyle='--', linewidth=1)

                rmse = float(jnp.sqrt(jnp.mean(result.residuals[cal_name]**2)))
                ax.set_title(f'{cal_name.upper()}\nRMSE: {rmse:.2f} K', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'{cal_name}\nNo data',
                       ha='center', va='center', transform=ax.transAxes)

            ax.set_xlabel('Freq (MHz)', fontsize=9)
            ax.set_ylabel('Temp (K)', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3)

        # Bottom row: Residuals statistics
        ax_res = fig.add_subplot(gs[2, :])

        cal_names = sorted([c for c in result.residuals.keys() if c != 'ant'])
        rmse_values = [float(jnp.sqrt(jnp.mean(result.residuals[c]**2))) for c in cal_names]

        x_pos = np.arange(len(cal_names))
        bars = ax_res.bar(x_pos, rmse_values, color='C0', alpha=0.7)

        # Color-code bars by value
        for i, (bar, val) in enumerate(zip(bars, rmse_values)):
            if val < 5:
                bar.set_color('green')
            elif val < 15:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        ax_res.set_xticks(x_pos)
        ax_res.set_xticklabels(cal_names, rotation=45, ha='right')
        ax_res.set_ylabel('RMSE (K)')
        ax_res.set_title('Calibration Residuals (RMSE) by Source')
        ax_res.axhline(y=5, color='g', linestyle='--', alpha=0.5, label='Good (<5K)')
        ax_res.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='OK (<15K)')
        ax_res.legend(loc='upper right')
        ax_res.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Calibration Results Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if self.save:
            filepath = self._get_filename('calibration_summary')
            plt.savefig(filepath, bbox_inches='tight')

        if self.show:
            plt.show()

        plt.close()
    def plot_s11_components(self, data):
        """
        Plot real and imaginary components of S11 for all calibrators.

        Args:
            data: CalibrationData object
        """
        cal_names = sorted(data.calibrators.keys())

        fig, (ax_real, ax_imag) = plt.subplots(2, 1, figsize=(12, 10), dpi=self.dpi)

        # Color cycle for different calibrators
        colors = plt.cm.tab20(np.linspace(0, 1, len(cal_names)))

        for idx, cal_name in enumerate(cal_names):
            cal_data = data.calibrators[cal_name]

            # Get S11 data
            freq_mhz = cal_data.s11_freq / 1e6
            s11_complex = cal_data.s11_complex

            # Plot real part
            ax_real.plot(freq_mhz, jnp.real(s11_complex),
                        label=cal_name.upper(), color=colors[idx], linewidth=1.5, alpha=0.8)

            # Plot imaginary part
            ax_imag.plot(freq_mhz, jnp.imag(s11_complex),
                        label=cal_name.upper(), color=colors[idx], linewidth=1.5, alpha=0.8)

        # Configure real plot
        ax_real.set_xlabel('Frequency (MHz)')
        ax_real.set_ylabel('Re(S11)')
        ax_real.set_title('S11 Real Component - All Calibrators')
        ax_real.legend(loc='best', ncol=2, fontsize=8)
        ax_real.grid(True, alpha=0.3)

        # Configure imaginary plot
        ax_imag.set_xlabel('Frequency (MHz)')
        ax_imag.set_ylabel('Im(S11)')
        ax_imag.set_title('S11 Imaginary Component - All Calibrators')
        ax_imag.legend(loc='best', ncol=2, fontsize=8)
        ax_imag.grid(True, alpha=0.3)

        plt.suptitle('S11 Components for All Calibrators', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if self.save:
            filepath = self._get_filename('s11_components')
            plt.savefig(filepath, bbox_inches='tight')

        if self.show:
            plt.show()

        plt.close()
