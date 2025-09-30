"""
Least squares calibration model implementation.

This module implements the least squares method for radiometer calibration,
solving for noise wave parameters (u, c, s, NS, L) that characterise the
receiver system. The method uses a linear least squares approach to fit
the calibration equation:

    T_ant = T_u * x_u + T_c * x_c + T_s * x_s + T_NS * x_NS + T_L * x_L

where x_i are the basis functions derived from S-parameters and power
measurements, and T_i are the noise wave parameters to be determined.

Mathematical formulation:
    For each frequency channel, solve: X @ θ = T_cal
    where:
    - X is the design matrix [n_calibrators, 5]
    - θ is the parameter vector [T_u, T_c, T_s, T_NS, T_L]
    - T_cal is the calibrator temperature vector [n_calibrators,]
"""

from typing import Dict, Any, Optional, Tuple
import jax
import jax.numpy as jnp
from functools import partial

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.base import BaseModel
from data import CalibrationData, CalibratorData


class LeastSquaresModel(BaseModel):
    """
    Least squares calibration model using JAX.

    This model implements a vectorised least squares solver for extracting
    noise wave parameters from radiometer calibration data. It uses JAX
    for GPU acceleration and automatic differentiation support.

    Noise wave parameters:
        u: Uncorrelated noise
        c: Correlated noise (cosine component)
        s: Correlated noise (sine component)
        NS: Noise source temperature
        L: Load temperature

    Attributes:
        config: Model configuration with optional parameters:
            - regularisation: Regularisation parameter (default: 0.0)
            - use_gamma_weighting: Apply reflection coefficient weighting (default: False)
            - observation_index: Which time observation to use (default: 0)
            - use_radiometric_weighting: Weight by 1/T_cal for radiometric noise (default: True)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise the least squares model.

        Args:
            config: Optional configuration dictionary with:
                - regularisation: L2 regularisation parameter
                - use_gamma_weighting: Boolean for gamma weighting
                - observation_index: Which time observation to use (0-indexed)
                - use_radiometric_weighting: Weight by inverse temperature (default: True)
        """
        super().__init__(config)

        # Extract configuration
        self.regularisation = self.config.get('regularisation', 0.0)
        self.use_gamma_weighting = self.config.get('use_gamma_weighting', False)
        self.observation_index = self.config.get('observation_index', 0)
        self.use_radiometric_weighting = self.config.get('use_radiometric_weighting', True)

        # Internal state
        self._X_matrices = None  # Cache for design matrices
        self._frequencies = None  # Cache for frequency array

    def fit(self, data: CalibrationData) -> None:
        """
        Fit the model to calibration data.

        Extracts measurements from all calibrators (excluding antenna), constructs design matrices,
        and solves the least squares problem for each frequency channel.

        Args:
            data: CalibrationData containing measurements for all calibrators

        Raises:
            ValueError: If data is insufficient or invalid
            RuntimeError: If fitting fails
        """
        self._data = data
        self._frequencies = data.psd_frequencies

        # Validate we have required calibrators
        required_cals = ['hot', 'cold']
        for cal in required_cals:
            if cal not in data.calibrators:
                raise ValueError(f"Required calibrator '{cal}' not found in data")

        # Build design matrices for all calibrators EXCEPT antenna
        X_list = []
        T_list = []

        for cal_name, cal_data in data.calibrators.items():
            # Skip antenna - it's what we're calibrating FOR
            if cal_name == 'ant':
                continue

            # Build X matrix for this calibrator
            X_cal = self._build_X_matrix(cal_data, data.psd_frequencies)
            X_list.append(X_cal)

            # Get calibrator temperature
            T_cal = self._get_calibrator_temperature(cal_data, len(data.psd_frequencies))
            T_list.append(T_cal)

        # Stack matrices: [n_freq, n_calibrators, 5]
        X = jnp.stack(X_list, axis=1)
        # Stack temperatures: [n_freq, n_calibrators]
        T = jnp.stack(T_list, axis=1)

        # Store for later use
        self._X_matrices = X

        # Compute radiometric weights if enabled
        weights = None
        if self.use_radiometric_weighting:
            # Weight by inverse temperature: w = 1/T
            # Radiometric noise variance is proportional to temperature
            weights = 1.0 / T
            # Normalize weights (sum to n_calibrators per frequency)
            weights = weights / jnp.mean(weights, axis=1, keepdims=True)

        # Solve least squares for each frequency using vmap
        self._parameters = self._solve_vectorised(X, T, weights)

        self.fitted = True
        self._result = None  # Reset cached result

    @partial(jax.jit, static_argnums=(0,))
    def _solve_vectorised(self, X: jax.Array, T: jax.Array,
                         weights: Optional[jax.Array] = None) -> Dict[str, jax.Array]:
        """
        Vectorised least squares solver across frequencies.

        For each frequency, solves the linear system:
            X @ θ = T (unweighted)
        or minimizes ||√W(X @ θ - T)||² (weighted)

        Args:
            X: Design matrices [n_freq, n_calibration_sources, 5]
            T: Temperature measurements [n_freq, n_calibration_sources]
            weights: Optional weights [n_freq, n_calibration_sources] for radiometric noise

        Returns:
            Dictionary of noise wave parameters
        """
        # Define single frequency solver
        def solve_single_freq(X_freq, T_freq, w_freq):
            """Solve (weighted) least squares for single frequency."""

            # Weighted least squares: θ = (X^T W X)^-1 X^T W T
            # When w=1, this reduces to standard least squares
            # W is diagonal, so we compute directly
            W_X = X_freq * w_freq[:, None]  # Weight each row
            W_T = T_freq * w_freq

            XTX = X_freq.T @ W_X
            XTT = X_freq.T @ W_T

            if self.regularisation > 0:
                XTX = XTX + self.regularisation * jnp.eye(5)

            theta = jnp.linalg.solve(XTX, XTT)
            return theta

        # Prepare weights (use ones if not provided)
        if weights is None:
            weights = jnp.ones_like(T)

        # Apply to all frequencies in parallel
        theta_all = jax.vmap(solve_single_freq)(X, T, weights)

        # Split into individual parameters
        return {
            'u': theta_all[:, 0],
            'c': theta_all[:, 1],
            's': theta_all[:, 2],
            'NS': theta_all[:, 3],
            'L': theta_all[:, 4]
        }

    def _build_X_matrix(self, cal_data: CalibratorData, frequencies: jax.Array) -> jax.Array:
        """
        Build the design matrix X for a calibrator.

        The X matrix contains the basis functions derived from S-parameters
        and power measurements.

        Args:
            cal_data: Calibrator data containing measurements
            frequencies: Frequency array for interpolation

        Returns:
            Design matrix X [n_freq, 5]
        """
        # Select observation index (handle calibrators with fewer observations)
        obs_idx = min(self.observation_index, cal_data.psd_source.shape[0] - 1)

        # Check if we need to interpolate PSD data to new frequencies
        if len(frequencies) != cal_data.psd_source.shape[1]:
            # Interpolate PSD measurements to target frequencies
            # Use the stored PSD frequencies for interpolation
            psd_freq = self._data.psd_frequencies

            # Select single observation
            P_cal_orig = cal_data.psd_source[obs_idx]
            P_L_orig = cal_data.psd_load[obs_idx]
            P_NS_orig = cal_data.psd_ns[obs_idx]

            # Interpolate to target frequencies
            P_cal = jnp.interp(frequencies, psd_freq, P_cal_orig)
            P_L = jnp.interp(frequencies, psd_freq, P_L_orig)
            P_NS = jnp.interp(frequencies, psd_freq, P_NS_orig)
        else:
            # Select single observation from PSD measurements
            P_cal = cal_data.psd_source[obs_idx]
            P_L = cal_data.psd_load[obs_idx]
            P_NS = cal_data.psd_ns[obs_idx]

        # Interpolate S11 to target frequencies
        s11_interp = self._interpolate_s11(
            cal_data.s11_freq,
            cal_data.s11_complex,
            frequencies
        )

        # Get receiver S11 (LNA S11) - REQUIRED
        if self._data.lna_s11 is None:
            raise ValueError("LNA S11 data is required for calibration but was not found. "
                           "Ensure the HDF5 file contains 'lna_s11' dataset.")

        # Interpolate LNA S11 to target frequencies
        Gamma_rec = self._interpolate_s11(
            self._data.vna_frequencies,
            self._data.lna_s11,
            frequencies
        )

        Gamma_cal = s11_interp

        # Compute X matrix components
        if self.use_gamma_weighting:
            # Gamma-weighted formulation
            x_u = -jnp.abs(Gamma_cal)**2
            x_L = jnp.abs(1 - Gamma_cal * Gamma_rec)**2
            weight = 1 - jnp.abs(Gamma_cal)**2
        else:
            # Standard formulation
            denominator = 1 - jnp.abs(Gamma_cal)**2
            x_u = -jnp.abs(Gamma_cal)**2 / denominator
            x_L = jnp.abs(1 - Gamma_cal * Gamma_rec)**2 / denominator
            weight = 1.0

        # Sinusoidal components for correlated noise
        sinusoid = (Gamma_cal / (1 - Gamma_cal * Gamma_rec)) * \
                   (x_L / jnp.sqrt(1 - jnp.abs(Gamma_rec)**2 + 1e-10))
        x_c = -jnp.real(sinusoid)
        x_s = -jnp.imag(sinusoid)

        # Noise source component
        x_NS = ((P_cal - P_L) / (P_NS - P_L + 1e-10)) * x_L

        # Stack into design matrix [n_freq, 5]
        X = jnp.stack([x_u, x_c, x_s, x_NS, x_L], axis=-1)

        return X


    def _interpolate_s11(self, vna_freq: jax.Array, s11_complex: jax.Array,
                        target_freq: jax.Array) -> jax.Array:
        """
        Interpolate S11 parameters to target frequencies.

        Args:
            vna_freq: VNA frequency points
            s11_complex: Complex S11 values
            target_freq: Target frequency array

        Returns:
            Interpolated S11 values
        """
        # Interpolate real and imaginary parts separately
        s11_real = jnp.interp(target_freq, vna_freq, s11_complex.real)
        s11_imag = jnp.interp(target_freq, vna_freq, s11_complex.imag)

        return s11_real + 1j * s11_imag

    def _get_calibrator_temperature(self, cal_data: CalibratorData,
                                   n_freq: int) -> jax.Array:
        """
        Get calibrator temperature array.

        Args:
            cal_data: Calibrator data
            n_freq: Number of frequency points

        Returns:
            Temperature array [n_freq,]

        Raises:
            ValueError: If temperature data is missing
        """
        if cal_data.temperature is None:
            raise ValueError(f"Calibrator '{cal_data.name}' has no temperature data. "
                           f"Temperature measurements are required for calibration.")

        if cal_data.temperature.ndim == 0:
            # Scalar temperature - broadcast to frequency dimension
            return jnp.full(n_freq, float(cal_data.temperature))
        else:
            # Array temperature (frequency-dependent)
            if len(cal_data.temperature) != n_freq:
                raise ValueError(f"Temperature array length {len(cal_data.temperature)} "
                               f"doesn't match frequency dimension {n_freq}")
            return cal_data.temperature

    def predict(self, frequencies: jax.Array, calibrator: str) -> jax.Array:
        """
        Predict calibrated temperatures for a calibrator.

        Uses the linear model: T = X @ θ
        where X is the design matrix and θ are the fitted parameters.

        Args:
            frequencies: Frequency points for prediction
            calibrator: Name of calibrator

        Returns:
            Predicted temperatures [n_freq,]

        Raises:
            RuntimeError: If model not fitted
            KeyError: If calibrator not found
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self._validate_calibrator(calibrator)
        self._validate_frequencies(frequencies)

        # Get calibrator data
        cal_data = self._data.get_calibrator(calibrator)

        # Check if we need to interpolate parameters
        if not jnp.array_equal(frequencies, self._frequencies):
            # Interpolate parameters to requested frequencies
            params_interp = {}
            for key, values in self._parameters.items():
                params_interp[key] = jnp.interp(frequencies, self._frequencies, values)
        else:
            params_interp = self._parameters

        # Build X matrix for this calibrator at requested frequencies
        X_cal = self._build_X_matrix(cal_data, frequencies)

        # Stack parameters into parameter vector [n_freq, 5]
        theta = jnp.stack([
            params_interp['u'],
            params_interp['c'],
            params_interp['s'],
            params_interp['NS'],
            params_interp['L']
        ], axis=1)

        # Compute prediction using linear model: T = X @ θ
        # X_cal is [n_freq, 5], theta is [n_freq, 5]
        # We need element-wise multiplication and sum over parameter dimension
        T_pred = jnp.sum(X_cal * theta, axis=1)

        return T_pred

    def get_parameters(self) -> Dict[str, jax.Array]:
        """
        Return fitted noise wave parameters.

        Returns:
            Dictionary with keys 'u', 'c', 's', 'NS', 'L'

        Raises:
            RuntimeError: If model not fitted
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting parameters")

        return self._parameters

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'regularisation': self.regularisation,
            'use_gamma_weighting': self.use_gamma_weighting
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.fitted else "not fitted"
        return (f"LeastSquaresModel(status={status}, "
                f"regularisation={self.regularisation}, "
                f"use_gamma_weighting={self.use_gamma_weighting})")