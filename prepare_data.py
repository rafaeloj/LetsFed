#!/usr/bin/env python3
"""
Script to prepare and partition federated learning datasets.

This script downloads and partitions the dataset according to the configuration
before running the federated learning experiment.
"""

from pathlib import Path

from conf.loader import load_config
from dataset_manager.dataset_manager import DSManager
from utils.logger import Logger

logger = Logger(__name__)


def main() -> None:
    """Prepare dataset partitions for federated learning."""

    # Load configuration
    config_path = Path(__file__).parent / "conf" / "config.yaml"
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Initialize dataset manager
    logger.info(f"Initializing dataset manager for {config.dataset.dataset}")
    logger.info(f"Number of clients: {config.n_clients}")
    logger.info(f"Train partitioner: {config.dataset.train_partitioner.method}")
    logger.info(f"Test partitioner: {config.dataset.test_partitioner.method}")

    dm = DSManager(n_clients=config.n_clients, conf=config.dataset)

    # Download and partition data
    logger.info("Downloading and partitioning dataset...")
    logger.info("This may take a few minutes on first run...")

    # Load the federated dataset (this downloads it)
    dm.load(dataset=config.dataset.dataset)

    # Save partitions locally
    logger.info(f"Saving partitions to {dm.path}")
    dm.save_locally()

    logger.info("Dataset preparation completed successfully!")
    logger.info(f"Partitions saved to: {dm.path}")
    logger.info(f"Distribution plot saved to: {dm.path}/partition_distributions.png")


if __name__ == "__main__":
    main()
