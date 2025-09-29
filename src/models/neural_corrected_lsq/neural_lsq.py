"""
Neural-corrected least squares calibration model implementation.

This module implements a hybrid physics-ML calibration approach that combines
analytical least squares fitting with neural network corrections. The model
solves for noise wave parameters using standard least squares (preserving
physical interpretability), then trains a neural network to learn frequency-
dependent corrections for systematic effects not captured by the physical model.

Two-stage fitting process:
    Stage 1: Solve analytically for noise wave parameters θ = (X^T X)^-1 X^T T
    Stage 2: Train neural network to predict correction A(freq, Γ_cal) on residuals

Final prediction: T = F(θ, measurements, freq) + A(freq, Γ_cal)
"""

from typing import Dict, Any, Optional, Tuple, Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from functools import partial

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.base import BaseModel
from data import CalibrationData, CalibratorData


class CorrectionNetwork(nn.Module):
    """
    Neural network for learning frequency-dependent calibration corrections.

    Inputs:
        - frequency (normalised)
        - |Γ_cal| (magnitude of reflection coefficient)
        - Re(Γ_cal) (real part of reflection coefficient)
        - Im(Γ_cal) (imaginary part of reflection coefficient)

    Output:
        - A(freq, Γ_cal): scalar correction value

    Attributes:
        hidden_layers: Tuple of hidden layer sizes (e.g., (32, 32))
        dropout_rate: Dropout rate (default: 0.0, no dropout)
    """
    hidden_layers: Sequence[int] = (32, 32)
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        """
        Forward pass through the network.

        Args:
            x: Input features [batch, 3] containing [freq, |Γ|, arg(Γ)]
            deterministic: If True, disable dropout (inference mode)

        Returns:
            Correction values [batch,]
        """
        for hidden_size in self.hidden_layers:
            x = nn.Dense(hidden_size)(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        # Output layer - no activation (can be positive or negative)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


class NeuralCorrectedLSQModel(BaseModel):
    """
    Neural-corrected least squares calibration model using JAX.

    This model combines physics-based least squares fitting with neural network
    corrections to capture systematic effects not modelled by the physical equations.
    The least squares solution is computed analytically and frozen, preserving the
    physical interpretation of noise wave parameters.

    Two-stage fitting:
        1. Analytical least squares: solve for noise wave parameters θ
        2. Neural network training: learn correction A(freq, Γ_cal) on residuals

    Attributes:
        config: Model configuration with parameters:
            - regularisation: L2 regularisation for least squares (default: 0.0)
            - use_gamma_weighting: Gamma weighting for least squares (default: False)
            - hidden_layers: Neural network architecture (default: (32, 32))
            - learning_rate: Adam learning rate (default: 1e-3)
            - n_iterations: Training iterations (default: 1000)
            - correction_regularization: Penalty on |A|² (default: 0.01)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise the neural-corrected least squares model.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)

        # Least squares configuration
        self.regularisation = self.config.get('regularisation', 0.0)
        self.use_gamma_weighting = self.config.get('use_gamma_weighting', False)

        # Neural network configuration
        self.hidden_layers = tuple(self.config.get('hidden_layers', [32, 32]))
        self.learning_rate = self.config.get('learning_rate', 1e-3)
        self.n_iterations = self.config.get('n_iterations', 1000)
        self.correction_regularization = self.config.get('correction_regularization', 0.01)
        self.dropout_rate = self.config.get('dropout_rate', 0.0)

        # Validation and early stopping configuration
        self.validation_check_interval = self.config.get('validation_check_interval', 50)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.min_delta = self.config.get('min_delta', 1e-4)  # Minimum improvement threshold

        # Internal state
        self._theta_ls = None  # Frozen least squares parameters
        self._nn_params = None  # Trained neural network parameters
        self._nn_state = None  # Neural network architecture/state
        self._X_matrices = None  # Cache for design matrices
        self._frequencies = None  # Cache for frequency array
        self._freq_mean = None  # For frequency normalisation
        self._freq_std = None  # For frequency normalisation
        self._validation_data = None  # Optional validation data

    def fit(self, data: CalibrationData, validation_data: Optional[CalibrationData] = None) -> None:
        """
        Fit the model to calibration data using two-stage process.

        Stage 1: Solve least squares analytically for noise wave parameters
        Stage 2: Train neural network on residuals with optional validation monitoring

        Args:
            data: CalibrationData containing measurements for all calibrators
            validation_data: Optional CalibrationData for validation monitoring and early stopping

        Raises:
            ValueError: If data is insufficient or invalid
            RuntimeError: If fitting fails
        """
        self._data = data
        self._frequencies = data.psd_frequencies
        self._validation_data = validation_data

        # Validate required calibrators
        required_cals = ['hot', 'cold']
        for cal in required_cals:
            if cal not in data.calibrators:
                raise ValueError(f"Required calibrator '{cal}' not found in data")

        # Stage 1: Analytical least squares (frozen)
        print("Stage 1: Solving least squares analytically...")
        self._theta_ls = self._solve_least_squares(data)

        # Stage 2: Train neural network on residuals
        print("Stage 2: Training neural network on residuals...")
        if validation_data is not None:
            print(f"  Validation monitoring enabled (check every {self.validation_check_interval} iterations)")
            print(f"  Early stopping patience: {self.early_stopping_patience}")
        self._nn_params, self._nn_state = self._train_neural_network(data, validation_data)

        self.fitted = True
        self._result = None  # Reset cached result
        print("Fitting complete.")

    def _solve_least_squares(self, data: CalibrationData) -> Dict[str, jax.Array]:
        """
        Stage 1: Solve least squares analytically for noise wave parameters.

        This uses the same logic as LeastSquaresModel to compute the analytical
        solution: θ = (X^T X)^-1 X^T T

        Args:
            data: CalibrationData

        Returns:
            Dictionary of noise wave parameters {u, c, s, NS, L}
        """
        # Build design matrices for all calibrators EXCEPT antenna
        X_list = []
        T_list = []

        for cal_name, cal_data in data.calibrators.items():
            if cal_name == 'ant':
                continue  # Exclude antenna from fitting

            X_cal = self._build_X_matrix(cal_data, data.psd_frequencies)
            X_list.append(X_cal)

            T_cal = self._get_calibrator_temperature(cal_data, len(data.psd_frequencies))
            T_list.append(T_cal)

        # Stack matrices: [n_freq, n_calibrators, 5]
        X = jnp.stack(X_list, axis=1)
        T = jnp.stack(T_list, axis=1)

        # Store for later use
        self._X_matrices = X

        # Solve least squares for each frequency using vmap
        return self._solve_vectorised(X, T)

    @partial(jax.jit, static_argnums=(0,))
    def _solve_vectorised(self, X: jax.Array, T: jax.Array) -> Dict[str, jax.Array]:
        """
        Vectorised least squares solver across frequencies.

        For each frequency, solves: X @ θ = T

        Args:
            X: Design matrices [n_freq, n_calibrators, 5]
            T: Temperature measurements [n_freq, n_calibrators]

        Returns:
            Dictionary of noise wave parameters
        """
        def solve_single_freq(X_freq, T_freq):
            """Solve least squares for single frequency."""
            if self.regularisation > 0:
                XTX = X_freq.T @ X_freq
                XTX = XTX + self.regularisation * jnp.eye(5)
                XTT = X_freq.T @ T_freq
                theta = jnp.linalg.solve(XTX, XTT)
            else:
                theta, residuals, rank, s = jnp.linalg.lstsq(X_freq, T_freq, rcond=None)
            return theta

        # Apply to all frequencies in parallel
        theta_all = jax.vmap(solve_single_freq)(X, T)

        # Split into individual parameters
        return {
            'u': theta_all[:, 0],
            'c': theta_all[:, 1],
            's': theta_all[:, 2],
            'NS': theta_all[:, 3],
            'L': theta_all[:, 4]
        }

    def _train_neural_network(self, data: CalibrationData,
                             validation_data: Optional[CalibrationData] = None) -> Tuple[Any, CorrectionNetwork]:
        """
        Stage 2: Train neural network to learn corrections on residuals.

        Prepares training data from residuals between least squares predictions
        and true temperatures, then optimises neural network parameters with
        optional validation monitoring and early stopping.

        Args:
            data: Training CalibrationData
            validation_data: Optional validation CalibrationData for early stopping

        Returns:
            Tuple of (trained_params, network_module)
        """
        # Helper function to prepare inputs/targets from data
        def prepare_data(cal_data_dict, frequencies):
            inputs_list = []
            targets_list = []

            for cal_name, cal_data in cal_data_dict.items():
                if cal_name == 'ant':
                    continue  # Exclude antenna from training

                # Get linear predictions from least squares
                X_cal = self._build_X_matrix(cal_data, frequencies)
                theta_stacked = jnp.stack([
                    self._theta_ls['u'],
                    self._theta_ls['c'],
                    self._theta_ls['s'],
                    self._theta_ls['NS'],
                    self._theta_ls['L']
                ], axis=1)
                T_linear = jnp.sum(X_cal * theta_stacked, axis=1)

                # Get true temperatures
                T_true = self._get_calibrator_temperature(cal_data, len(frequencies))

                # Compute residuals
                residuals = T_true - T_linear

                # Prepare input features [freq, |Γ|, arg(Γ)]
                s11_interp = self._interpolate_s11(
                    cal_data.s11_freq,
                    cal_data.s11_complex,
                    frequencies
                )

                features = jnp.stack([
                    frequencies,
                    jnp.abs(s11_interp),    # Magnitude of S11
                    jnp.angle(s11_interp)   # Phase of S11
                ], axis=1)

                inputs_list.append(features)
                targets_list.append(residuals)

            # Stack all calibrators
            inputs = jnp.concatenate(inputs_list, axis=0)
            targets = jnp.concatenate(targets_list, axis=0)
            return inputs, targets

        # Prepare training data
        train_inputs, train_targets = prepare_data(data.calibrators, data.psd_frequencies)

        # Normalise frequency dimension for better training
        self._freq_mean = jnp.mean(train_inputs[:, 0])
        self._freq_std = jnp.std(train_inputs[:, 0])
        train_inputs = train_inputs.at[:, 0].set((train_inputs[:, 0] - self._freq_mean) / (self._freq_std + 1e-10))

        # Prepare validation data if provided
        val_inputs, val_targets = None, None
        if validation_data is not None:
            val_inputs, val_targets = prepare_data(validation_data.calibrators, validation_data.psd_frequencies)
            # Apply same normalisation as training data
            val_inputs = val_inputs.at[:, 0].set((val_inputs[:, 0] - self._freq_mean) / (self._freq_std + 1e-10))

        # Initialise network
        network = CorrectionNetwork(hidden_layers=self.hidden_layers, dropout_rate=self.dropout_rate)
        rng = jax.random.PRNGKey(0)
        params = network.init(rng, train_inputs[:10], deterministic=False)  # Initialise with small batch

        # Set up optimiser
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        # Define training loss function (with regularization, dropout enabled)
        @jax.jit
        def train_loss_fn(params, inputs, targets, rng):
            predictions = network.apply(params, inputs, deterministic=False, rngs={'dropout': rng})

            # MSE loss on residuals
            mse_loss = jnp.mean((predictions - targets)**2)

            # Regularisation: penalise large corrections
            reg_loss = self.correction_regularization * jnp.mean(predictions**2)

            return mse_loss + reg_loss, {'mse': mse_loss, 'reg': reg_loss}

        # Define validation loss function (without regularization, dropout disabled)
        @jax.jit
        def val_loss_fn(params, inputs, targets):
            predictions = network.apply(params, inputs, deterministic=True)

            # MSE loss only
            mse_loss = jnp.mean((predictions - targets)**2)

            return mse_loss, {'mse': mse_loss, 'reg': 0.0}

        # Training step
        @jax.jit
        def train_step(params, opt_state, inputs, targets, rng):
            (loss_val, metrics), grads = jax.value_and_grad(train_loss_fn, has_aux=True)(params, inputs, targets, rng)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val, metrics

        # Early stopping state
        best_val_loss = float('inf')
        best_params = params
        patience_counter = 0

        # Training loop with dropout RNG
        dropout_rng = jax.random.PRNGKey(1)  # Separate RNG for dropout
        for i in range(self.n_iterations):
            dropout_rng, step_rng = jax.random.split(dropout_rng)
            params, opt_state, loss_val, metrics = train_step(params, opt_state, train_inputs, train_targets, step_rng)

            # Validation check
            check_validation = (validation_data is not None and
                              i % self.validation_check_interval == 0)

            if i % 100 == 0 or check_validation:
                train_rmse = jnp.sqrt(metrics['mse'])
                log_str = f"  Iteration {i}/{self.n_iterations}: train_rmse={train_rmse:.6f}"

                if check_validation:
                    val_loss, val_metrics = val_loss_fn(params, val_inputs, val_targets)
                    val_rmse = jnp.sqrt(val_metrics['mse'])
                    log_str += f", val_rmse={val_rmse:.6f}"

                    # Early stopping logic
                    if val_loss < best_val_loss - self.min_delta:
                        best_val_loss = float(val_loss)
                        best_params = params
                        patience_counter = 0
                        log_str += " *"  # Mark improvement
                    else:
                        patience_counter += 1

                    if patience_counter >= self.early_stopping_patience:
                        print(log_str)
                        print(f"  Early stopping triggered at iteration {i}")
                        print(f"  Best validation RMSE: {jnp.sqrt(best_val_loss):.6f}")
                        params = best_params  # Restore best parameters
                        break

                print(log_str)

        if validation_data is not None and patience_counter < self.early_stopping_patience:
            print(f"  Training completed. Best validation RMSE: {jnp.sqrt(best_val_loss):.6f}")
            params = best_params  # Use best parameters
        else:
            print(f"  Final training loss: {loss_val:.6f}")

        return params, network

    def _build_X_matrix(self, cal_data: CalibratorData, frequencies: jax.Array) -> jax.Array:
        """
        Build the design matrix X for a calibrator (reused from LeastSquaresModel).

        Args:
            cal_data: Calibrator data containing measurements
            frequencies: Frequency array for interpolation

        Returns:
            Design matrix X [n_freq, 5]
        """
        # Check if we need to interpolate PSD data
        if len(frequencies) != cal_data.psd_source.shape[1]:
            psd_freq = self._data.psd_frequencies
            P_cal_orig = jnp.mean(cal_data.psd_source, axis=0)
            P_L_orig = jnp.mean(cal_data.psd_load, axis=0)
            P_NS_orig = jnp.mean(cal_data.psd_ns, axis=0)

            P_cal = jnp.interp(frequencies, psd_freq, P_cal_orig)
            P_L = jnp.interp(frequencies, psd_freq, P_L_orig)
            P_NS = jnp.interp(frequencies, psd_freq, P_NS_orig)
        else:
            P_cal = jnp.mean(cal_data.psd_source, axis=0)
            P_L = jnp.mean(cal_data.psd_load, axis=0)
            P_NS = jnp.mean(cal_data.psd_ns, axis=0)

        # Interpolate S11 to target frequencies
        s11_interp = self._interpolate_s11(
            cal_data.s11_freq,
            cal_data.s11_complex,
            frequencies
        )

        # Get receiver S11 (LNA S11) - REQUIRED
        if self._data.lna_s11 is None:
            raise ValueError("LNA S11 data is required for calibration")

        Gamma_rec = self._interpolate_s11(
            self._data.vna_frequencies,
            self._data.lna_s11,
            frequencies
        )

        Gamma_cal = s11_interp

        # Compute X matrix components
        if self.use_gamma_weighting:
            x_u = -jnp.abs(Gamma_cal)**2
            x_L = jnp.abs(1 - Gamma_cal * Gamma_rec)**2
            weight = 1 - jnp.abs(Gamma_cal)**2
        else:
            denominator = 1 - jnp.abs(Gamma_cal)**2
            x_u = -jnp.abs(Gamma_cal)**2 / denominator
            x_L = jnp.abs(1 - Gamma_cal * Gamma_rec)**2 / denominator
            weight = 1.0

        # Sinusoidal components
        sinusoid = (Gamma_cal / (1 - Gamma_cal * Gamma_rec)) * \
                   (x_L / jnp.sqrt(1 - jnp.abs(Gamma_rec)**2 + 1e-10))
        x_c = -jnp.real(sinusoid)
        x_s = -jnp.imag(sinusoid)

        # Noise source component
        x_NS = ((P_cal - P_L) / (P_NS - P_L + 1e-10)) * x_L

        # Stack into design matrix
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
            raise ValueError(f"Calibrator '{cal_data.name}' has no temperature data")

        if cal_data.temperature.ndim == 0:
            return jnp.full(n_freq, float(cal_data.temperature))
        else:
            if len(cal_data.temperature) != n_freq:
                raise ValueError(f"Temperature array length {len(cal_data.temperature)} "
                               f"doesn't match frequency dimension {n_freq}")
            return cal_data.temperature

    def predict(self, frequencies: jax.Array, calibrator: str) -> jax.Array:
        """
        Predict calibrated temperatures including neural network correction.

        Combines linear prediction from least squares with neural network correction:
            T = X @ θ + A(freq, Γ_cal)

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

        # Interpolate parameters if needed
        if not jnp.array_equal(frequencies, self._frequencies):
            params_interp = {}
            for key, values in self._theta_ls.items():
                params_interp[key] = jnp.interp(frequencies, self._frequencies, values)
        else:
            params_interp = self._theta_ls

        # Linear component: X @ θ
        X_cal = self._build_X_matrix(cal_data, frequencies)
        theta_stacked = jnp.stack([
            params_interp['u'],
            params_interp['c'],
            params_interp['s'],
            params_interp['NS'],
            params_interp['L']
        ], axis=1)
        T_linear = jnp.sum(X_cal * theta_stacked, axis=1)

        # Neural network correction: A(freq, Γ_cal)
        s11_interp = self._interpolate_s11(
            cal_data.s11_freq,
            cal_data.s11_complex,
            frequencies
        )

        # Normalise frequency
        freq_norm = (frequencies - self._freq_mean) / (self._freq_std + 1e-10)

        features = jnp.stack([
            freq_norm,
            jnp.abs(s11_interp),    # Magnitude of S11
            jnp.angle(s11_interp)   # Phase of S11
        ], axis=1)

        A_correction = self._nn_state.apply(self._nn_params, features, deterministic=True)

        # Combined prediction
        return T_linear + A_correction

    def get_parameters(self) -> Dict[str, jax.Array]:
        """
        Return fitted noise wave parameters (from least squares only).

        Returns:
            Dictionary with keys 'u', 'c', 's', 'NS', 'L'

        Raises:
            RuntimeError: If model not fitted
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting parameters")

        return self._theta_ls

    def get_correction_magnitude(self) -> Dict[str, float]:
        """
        Get statistics about neural network corrections.

        Returns:
            Dictionary with correction statistics

        Raises:
            RuntimeError: If model not fitted
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting correction statistics")

        # Compute corrections for all calibrators
        corrections = []
        for cal_name in self._data.calibrator_names:
            if cal_name == 'ant':
                continue

            cal_data = self._data.get_calibrator(cal_name)
            s11_interp = self._interpolate_s11(
                cal_data.s11_freq,
                cal_data.s11_complex,
                self._frequencies
            )

            freq_norm = (self._frequencies - self._freq_mean) / (self._freq_std + 1e-10)
            features = jnp.stack([
                freq_norm,
                jnp.abs(s11_interp),    # Magnitude of S11
                jnp.angle(s11_interp)   # Phase of S11
            ], axis=1)

            A = self._nn_state.apply(self._nn_params, features, deterministic=True)
            corrections.append(A)

        corrections = jnp.concatenate(corrections)

        return {
            'mean': float(jnp.mean(corrections)),
            'std': float(jnp.std(corrections)),
            'min': float(jnp.min(corrections)),
            'max': float(jnp.max(corrections)),
            'rms': float(jnp.sqrt(jnp.mean(corrections**2)))
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'regularisation': self.regularisation,
            'use_gamma_weighting': self.use_gamma_weighting,
            'hidden_layers': list(self.hidden_layers),
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'correction_regularization': self.correction_regularization
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.fitted else "not fitted"
        return (f"NeuralCorrectedLSQModel(status={status}, "
                f"hidden_layers={self.hidden_layers}, "
                f"n_iterations={self.n_iterations})")