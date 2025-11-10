#!/usr/bin/env python3
"""
Example script demonstrating how to use the improved DockerComposeManager.

This script shows how to orchestrate federated learning experiments using
Docker Compose with proper separation of concerns.
"""

from conf.loader import load_config
from utils.docker_compose_manager import DockerComposeManager
from utils.logger import logger


def main() -> None:
    """Main entry point for FL experiment orchestration."""

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config("conf/config.yaml")

    # Initialize Docker Compose Manager
    logger.info("Initializing Docker Compose Manager...")
    manager = DockerComposeManager(config=config, compose_file="docker-compose.yml")

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

        # Wait for user input to continue
        input("\nPress Enter to view server logs (Ctrl+C to exit)...")

        # Show server logs
        manager.get_logs(service="server", tail=50)

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
