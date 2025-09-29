"""
Neural-corrected least squares calibration model.

This module provides a hybrid physics-ML calibration approach that combines
analytical least squares fitting with neural network corrections to model
systematic effects not captured by the physical model.
"""

from .neural_lsq import NeuralCorrectedLSQModel

__all__ = ['NeuralCorrectedLSQModel']