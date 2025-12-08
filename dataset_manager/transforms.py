"""
Data transformation pipeline for federated learning datasets.

This module centralizes all data preprocessing and transformation logic,
ensuring consistency between client and server data loading.
"""

import numpy as np

from ..utils.logger import Logger

logger = Logger(__name__)


class TransformPipeline:
    """
    Centralized pipeline for data transformations.

    This class provides static methods for common data transformations
    used in federated learning, including normalization, reshaping,
    and type conversions.
    """

    @staticmethod
    def normalize_images(images: np.ndarray, method: str = "tanh") -> np.ndarray:
        """
        Normalize image data.

        Args:
            images: Array of images with values in [0, 255]
            method: Normalization method
                - "tanh": Normalize to [-1, 1] (recommended for tanh activation)
                - "sigmoid": Normalize to [0, 1] (recommended for sigmoid activation)
                - "standardize": Z-score normalization (mean=0, std=1)

        Returns:
            Normalized images as float32 array

        Raises:
            ValueError: If unknown normalization method is specified
        """
        images = images.astype("float32")

        if method == "tanh":
            # [-1, 1] range: Better for gradient descent and Xavier/He initialization
            # This centers data around 0, which works well with modern optimizers
            normalized = (images - 127.5) / 127.5
            logger.debug(
                "Normalized images to [-1, 1] range (tanh) - "
                + f"Shape: {normalized.shape}, "
                + f"Min: {normalized.min():.3f}, Max: {normalized.max():.3f}"
            )
            return normalized

        elif method == "sigmoid":
            # [0, 1] range: Simple division by max value
            normalized = images / 255.0
            logger.debug(
                "Normalized images to [0, 1] range (sigmoid) - "
                + f"Shape: {normalized.shape}, "
                + f"Min: {normalized.min():.3f}, Max: {normalized.max():.3f}"
            )
            return normalized

        elif method == "standardize":
            # Z-score: mean=0, std=1
            mean = images.mean()
            std = images.std()
            normalized = (images - mean) / (std + 1e-7)  # Add epsilon to avoid division by zero
            logger.debug(
                "Standardized images (mean=0, std=1) - "
                + f"Shape: {normalized.shape}, "
                + f"Original mean: {mean:.3f}, std: {std:.3f}"
            )
            return normalized

        else:
            raise ValueError(
                f"Unknown normalization method: '{method}'. "
                + "Supported methods: 'tanh', 'sigmoid', 'standardize'"
            )

    @staticmethod
    def add_channel_dimension(images: np.ndarray) -> np.ndarray:
        """
        Add channel dimension for CNN models.

        Converts grayscale images from (N, H, W) to (N, H, W, 1) format.
        This is required for Conv2D layers which expect 4D input:
        (batch, height, width, channels)

        Args:
            images: Image array of shape (N, H, W) or (N, H, W, C)

        Returns:
            Image array with channel dimension added if needed
        """
        if len(images.shape) == 3:
            # Add channel dimension: (N, H, W) -> (N, H, W, 1)
            reshaped = np.expand_dims(images, axis=-1)
            logger.debug(
                "Added channel dimension for CNN - "
                + f"Original shape: {images.shape} -> New shape: {reshaped.shape}"
            )
            return reshaped

        # Already has channel dimension
        logger.debug(f"Images already have channel dimension - Shape: {images.shape}")
        return images

    @staticmethod
    def apply_transforms(
        images: np.ndarray,
        normalize: bool = True,
        normalization_method: str = "tanh",
        add_channel: bool = False,
    ) -> np.ndarray:
        """
        Apply a sequence of transformations to images.

        This is a convenience method that applies common transformations
        in the correct order.

        Args:
            images: Input images
            normalize: Whether to normalize images
            normalization_method: Method to use for normalization
            add_channel: Whether to add channel dimension for CNNs

        Returns:
            Transformed images
        """
        result = images

        # Step 1: Normalization
        if normalize:
            result = TransformPipeline.normalize_images(result, normalization_method)

        # Step 2: Reshape for CNN (after normalization)
        if add_channel:
            result = TransformPipeline.add_channel_dimension(result)

        return result
