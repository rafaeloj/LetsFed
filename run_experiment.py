#!/usr/bin/env python3
"""
Example script demonstrating how to use the improved DockerComposeManager.

This script shows how to orchestrate federated learning experiments using
Docker Compose with proper separation of concerns.
"""

from pathlib import Path

from .conf.loader import load_config
from .utils.docker_compose_manager import DockerComposeManager
from .utils.logger import Logger
from .utils.seed import set_global_seed

logger = Logger(__name__)


def main() -> None:
    """Main entry point for FL experiment orchestration."""

    # Load configuration
    logger.info("Loading configuration...")
    # Use path relative to this file's location
    config_path = Path(__file__).parent / "conf" / "config.yaml"
    config = load_config(config_path)

    # Set global seed BEFORE any random operations
    logger.info(f"Setting global seed to {config.seed}")
    set_global_seed(config.seed)

    # Initialize Docker Compose Manager
    logger.info("Initializing Docker Compose Manager...")
    compose_file = Path(__file__).parent / "docker-compose.yml"
    manager = DockerComposeManager(config=config, compose_file=str(compose_file))

    try:
        # Build Docker images if needed
        logger.info("Building Docker images...")
        manager.build_images(use_gpu=config.gpu)

        # Start the complete FL environment
        logger.info("Starting FL environment (server + clients)...")
        manager.start_all()

        # Check status
        logger.info("\nCurrent status:")
        logger.info(manager.get_status())

        # Wait for user input to stop
        input("\nPress Enter to stop all services...")

    except KeyboardInterrupt:
        logger.info("\nReceived interrupt signal...")

    finally:
        # Clean up: stop all services
        logger.info("Stopping all services...")
        manager.stop_all()
        logger.info("FL environment stopped successfully")


if __name__ == "__main__":
    main()
