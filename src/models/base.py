"""
Abstract base model for calibration models.

This module defines the interface that all calibration models must implement,
ensuring interoperability across different calibration methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data import CalibrationData, CalibrationResult


class BaseModel(ABC):
    """
    Abstract base class for calibration models.

    All calibration models must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface
    for model fitting, prediction, and parameter extraction.

    The model operates on CalibrationData inputs and produces
    CalibrationResult outputs containing noise wave parameters and
    predicted temperatures.

    Attributes:
        config: Model configuration dictionary
        fitted: Boolean indicating if model has been fitted
        _data: Reference to the calibration data (set during fit)
        _parameters: Fitted noise wave parameters
        _result: Cached calibration result
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise the calibration model.

        Args:
            config: Optional configuration dictionary for model-specific
                   parameters (e.g., regularisation, solver options)
        """
        self.config = config or {}
        self.fitted = False
        self._data: Optional[CalibrationData] = None
        self._parameters: Optional[Dict[str, jax.Array]] = None
        self._result: Optional[CalibrationResult] = None

    @abstractmethod
    def fit(self, data: CalibrationData) -> None:
        """
        Fit the calibration model to data.

        This method should:
        1. Extract relevant measurements from calibration data
        2. Solve for noise wave parameters (u, c, s, NS, L)
        3. Store fitted parameters internally
        4. Set self.fitted = True

        Args:
            data: CalibrationData object containing PSD measurements,
                 VNA S11 parameters, and temperatures for all calibrators

        Raises:
            ValueError: If data is invalid or insufficient for calibration
            RuntimeError: If fitting algorithm fails to converge
        """
        pass

    @abstractmethod
    def predict(self, frequencies: jax.Array, calibrator: str) -> jax.Array:
        """
        Predict calibrated temperatures for a specific calibrator.

        Uses the fitted noise wave parameters to predict the antenna
        temperature at specified frequencies for the given calibrator.

        Args:
            frequencies: Array of frequencies [Hz] at which to predict
            calibrator: Name of calibrator to predict for
                       (e.g., 'hot', 'cold', 'ant')

        Returns:
            Array of predicted temperatures [K] at each frequency

        Raises:
            RuntimeError: If model has not been fitted
            KeyError: If calibrator not found in calibration data
            ValueError: If frequencies are outside valid range
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, jax.Array]:
        """
        Return fitted noise wave parameters.

        Returns:
            Dictionary containing the 5 noise wave parameters:
            - 'u': Uncorrelated noise parameter [n_freq,]
            - 'c': Correlated noise parameter [n_freq,]
            - 's': Phase parameter [n_freq,]
            - 'NS': Noise source temperature [n_freq,]
            - 'L': Loss factor [n_freq,]

        Raises:
            RuntimeError: If model has not been fitted
        """
        pass

    def get_result(self) -> CalibrationResult:
        """
        Return complete calibration result.

        Generates predictions for all calibrators and computes residuals.
        Results are cached for efficiency.

        Returns:
            CalibrationResult object containing:
            - Noise wave parameters
            - Predicted temperatures for all calibrators
            - Residuals (if measured temperatures available)
            - Model metadata

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before getting results")

        # Return cached result if available
        if self._result is not None:
            return self._result

        # Generate predictions for all calibrators
        predicted_temperatures = {}
        residuals = {}

        for cal_name, cal_data in self._data.calibrators.items():
            # Use PSD frequencies for prediction
            pred_temp = self.predict(self._data.psd_frequencies, cal_name)
            predicted_temperatures[cal_name] = pred_temp

            # Compute residuals if measured temperature available
            if cal_data.temperature is not None:
                # Handle scalar temperature (broadcast to frequency dimension)
                if cal_data.temperature.ndim == 0:
                    measured = jnp.full_like(pred_temp, cal_data.temperature)
                else:
                    measured = cal_data.temperature
                residuals[cal_name] = pred_temp - measured

        # Create result object
        self._result = CalibrationResult(
            noise_parameters=self.get_parameters(),
            predicted_temperatures=predicted_temperatures,
            residuals=residuals,
            model_name=self.__class__.__name__,
            metadata={
                'config': self.config,
                'n_calibrators': len(self._data.calibrators),
                'frequency_range': (
                    float(self._data.psd_frequencies.min()),
                    float(self._data.psd_frequencies.max())
                )
            }
        )

        return self._result

    def _validate_calibrator(self, calibrator: str) -> None:
        """
        Validate that a calibrator exists in the data.

        Args:
            calibrator: Name of calibrator to validate

        Raises:
            RuntimeError: If model has not been fitted
            KeyError: If calibrator not found in data
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before validation")

        if calibrator not in self._data.calibrators:
            available = list(self._data.calibrators.keys())
            raise KeyError(
                f"Calibrator '{calibrator}' not found. "
                f"Available calibrators: {', '.join(available)}"
            )

    def _validate_frequencies(self, frequencies: jax.Array) -> None:
        """
        Validate frequency array.

        Args:
            frequencies: Frequency array to validate

        Raises:
            ValueError: If frequencies have invalid shape or values
        """
        if frequencies.ndim != 1:
            raise ValueError(
                f"Frequencies must be 1D array, got shape {frequencies.shape}"
            )

        if jnp.any(frequencies < 0):
            raise ValueError("Frequencies must be positive")

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.fitted else "not fitted"
        return f"{self.model_type}(status={status}, config={self.config})"