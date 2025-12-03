#!/usr/bin/env python3
"""
Script to prepare and partition federated learning datasets.

This script downloads and partitions the dataset according to the configuration
before running the federated learning experiment.
"""

import os
from pathlib import Path

from .conf.loader import load_config
from .dataset_manager.dataset_manager import DSManager
from .utils.logger import Logger
from .utils.seed import set_global_seed

logger = Logger(__name__)


def main() -> None:
    """Prepare dataset partitions for federated learning."""

    # Get the directory where this script is located (LetsFed directory)
    script_dir = Path(__file__).parent.resolve()

    # Change to the script directory to ensure relative paths work correctly
    original_cwd = os.getcwd()
    os.chdir(script_dir)
    logger.info(f"Working directory set to: {script_dir}")

    try:
        # Load configuration
        config_path = script_dir / "conf" / "config.yaml"
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)

        # Set global seed for reproducibility
        logger.info(f"Setting global seed to {config.seed} for reproducible partitioning")
        set_global_seed(config.seed)

        # Initialize dataset manager with seed
        logger.info(f"Initializing dataset manager for {config.dataset.dataset}")
        logger.info(f"Number of clients: {config.n_clients}")
        logger.info(f"Train partitioner: {config.dataset.train_partitioner.method}")
        logger.info(f"Test partitioner: {config.dataset.test_partitioner.method}")

        dm = DSManager(n_clients=config.n_clients, conf=config.dataset, seed=config.seed)

        # Download and partition data
        logger.info("Downloading and partitioning dataset...")
        logger.info("This may take a few minutes on first run...")

        # Load the federated dataset (this downloads it)
        dm.load(dataset=config.dataset.dataset)

        # Save partitions locally (will be saved relative to script_dir)
        save_path = script_dir / dm.path
        logger.info(f"Saving partitions to {save_path}")
        dm.save_locally()

        logger.info("Dataset preparation completed successfully!")
        logger.info(f"Partitions saved to: {save_path}")
        logger.info(f"Distribution plot saved to: {save_path}/partition_distributions.png")

    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
