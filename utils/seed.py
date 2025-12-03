"""
Seed management utilities for reproducible experiments.

This module provides functions to set random seeds across all libraries
used in the federated learning framework to ensure reproducibility.
"""

import hashlib
import os
import random

import numpy as np
import tensorflow as tf

from .logger import Logger

logger = Logger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Set random seed for all random number generators used in the framework.

    This includes:
    - Python's random module
    - NumPy
    - TensorFlow/Keras
    - Environment variables for deterministic operations

    Args:
        seed: Integer seed value for reproducibility

    Example:
        >>> set_global_seed(42)
        >>> # All subsequent random operations will be reproducible
    """
    logger.info(f"Setting global random seed to {seed}")

    # Python's built-in random
    random.seed(seed)
    logger.debug(f"  ✓ Python random seed set to {seed}")

    # NumPy
    np.random.seed(seed)
    logger.debug(f"  ✓ NumPy random seed set to {seed}")

    # TensorFlow
    tf.random.set_seed(seed)
    logger.debug(f"  ✓ TensorFlow random seed set to {seed}")

    # Set environment variables for deterministic operations
    # TF_DETERMINISTIC_OPS: Forces TensorFlow to use deterministic algorithms
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    logger.debug("  ✓ TF_DETERMINISTIC_OPS enabled")

    # PYTHONHASHSEED: Makes hash() deterministic (affects dict/set order in Python < 3.7)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug(f"  ✓ PYTHONHASHSEED set to {seed}")

    # Configure TensorFlow for determinism
    # Note: Some GPU operations may still be non-deterministic
    tf.config.experimental.enable_op_determinism()
    logger.debug("  ✓ TensorFlow op determinism enabled")

    logger.info("Global seed configuration completed")
    warning_msg = (
        "Note: Some GPU operations may still be non-deterministic. "
        + "For full reproducibility, use CPU mode (gpu: false)"
    )
    logger.warning(warning_msg)


def get_derived_seed(base_seed: int, component: str, client_id: int = 0) -> int:
    """
    Generate a derived seed for a specific component or client.

    This ensures different components/clients use different but reproducible seeds.
    Uses SHA256 for deterministic hashing across processes and machines.

    Args:
        base_seed: Base seed from configuration
        component: Component name (e.g., 'dataset', 'model', 'training')
        client_id: Client ID for client-specific operations (default: 0 for server)

    Returns:
        Derived seed as integer

    Example:
        >>> base_seed = 42
        >>> dataset_seed = get_derived_seed(base_seed, 'dataset')
        >>> client_0_seed = get_derived_seed(base_seed, 'training', client_id=0)
        >>> client_1_seed = get_derived_seed(base_seed, 'training', client_id=1)
    """
    # Use deterministic SHA256 hash instead of Python's hash()
    # Python's hash() is not deterministic across processes/machines
    seed_string = f"{base_seed}_{component}_{client_id}"
    hash_object = hashlib.sha256(seed_string.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    derived = hash_int % (2**31 - 1)  # Keep within int32 range
    return derived


def set_component_seed(base_seed: int, component: str, client_id: int = 0) -> int:
    """
    Set seed for a specific component using derived seed.

    Args:
        base_seed: Base seed from configuration
        component: Component name
        client_id: Client ID (default: 0 for server)

    Returns:
        The derived seed that was set

    Example:
        >>> set_component_seed(42, 'dataset', client_id=0)
        >>> # Dataset operations for client 0 now reproducible
    """
    derived_seed = get_derived_seed(base_seed, component, client_id)
    set_global_seed(derived_seed)
    return derived_seed
