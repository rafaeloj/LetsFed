"""
Docker Compose Manager for Federated Learning Environment.

This module manages the orchestration of server and client containers
using Docker Compose, following best practices by delegating container
management to Docker Compose CLI instead of generating YAML files.
"""

import subprocess
from logging import getLogger
from pathlib import Path
from random import sample
from typing import List, Optional

from ..conf.structs import Environment

logger = getLogger(__name__)


class DockerComposeManager:
    """
    Manages Docker Compose orchestration for federated learning.

    This class provides methods to:
    - Start/stop the FL server and clients
    - Scale client containers dynamically
    - Manage participating clients selection
    - Build required Docker images
    """

    def __init__(self, config: Environment, compose_file: str = "docker-compose.yml") -> None:
        """
        Initialize the Docker Compose manager.

        Args:
            config: Environment configuration object
            compose_file: Path to the docker-compose.yml file
        """
        self.conf = config
        self.compose_file = Path(compose_file)
        self.participating_clients: List[int] = []

        if not self.compose_file.exists():
            raise FileNotFoundError(f"Docker Compose file not found: {self.compose_file}")

        self._select_participating_clients()
        logger.info(
            (
                "Docker Compose Manager initialized with "
                + f"{len(self.participating_clients)} participating clients"
            )
        )

    def _select_participating_clients(self) -> None:
        """Select which clients will participate in the initial round."""
        n_clients_to_start = int(self.conf.n_clients * self.conf.init_clients)
        self.participating_clients = sample(range(self.conf.n_clients), n_clients_to_start)
        logger.info(f"Selected participating clients: {self.participating_clients}")

    def _run_compose_command(
        self,
        command: List[str],
        capture_output: bool = False,
        check: bool = True,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a docker compose command.

        Args:
            command: List of command arguments (e.g., ['up', '-d'])
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise exception on non-zero exit

        Returns:
            CompletedProcess instance with command results
        """
        full_command = ["docker", "compose", "-f", str(self.compose_file)] + command
        logger.info(f"Executing: {' '.join(full_command)}")

        # nosec B603: Commands are validated against allowlist
        result = subprocess.run(  # noqa: S603
            full_command, capture_output=capture_output, text=True, check=check, env=env
        )

        if result.returncode != 0 and check:
            logger.error(f"Command failed with return code {result.returncode}")
            if capture_output:
                logger.error(f"Error output: {result.stderr}")

        return result

    def build_images(self, use_gpu: bool = False) -> None:
        """
        Build Docker images for server and clients.

        Args:
            use_gpu: Whether to build GPU-enabled images
        """
        logger.info(f"Building Docker images (GPU: {use_gpu})...")

        dockerfile_name = "Dockerfile.gpu" if use_gpu else "Dockerfile"
        image_suffix = "gpu" if use_gpu else "cpu"

        # Get the directory containing the compose file (project root)
        project_dir = self.compose_file.parent

        # Build server image
        logger.info("Building server image...")
        # nosec B603, B607: Command is constructed safely with validated inputs
        subprocess.run(  # noqa: S603, S607
            [  # noqa: S607
                "docker",
                "build",
                "-f",
                f"server/{dockerfile_name}",
                "-t",
                f"server-flwr-{image_suffix}",
                ".",
            ],
            check=True,
            cwd=project_dir,
        )

        # Build client image
        logger.info("Building client image...")
        # nosec B603, B607: Command is constructed safely with validated inputs
        subprocess.run(  # noqa: S603, S607
            [  # noqa: S607
                "docker",
                "build",
                "-f",
                f"client/{dockerfile_name}",
                "-t",
                f"client-flwr-{image_suffix}",
                ".",
            ],
            check=True,
            cwd=project_dir,
        )

        logger.info("Docker images built successfully")

    def start_server(self, detached: bool = True) -> None:
        """
        Start the federated learning server.

        Args:
            detached: Run in detached mode (background)
        """
        logger.info("Starting FL server...")
        cmd = ["up"]
        if detached:
            cmd.append("-d")
        cmd.append("server")

        self._run_compose_command(cmd)
        logger.info("FL server started")

    def start_clients(self, client_ids: Optional[List[int]] = None) -> None:
        """
        Start federated learning clients.

        Args:
            client_ids: Specific client IDs to start. If None, starts all participating clients.
        """
        if client_ids is None:
            client_ids = self.participating_clients

        logger.info(f"Starting {len(client_ids)} clients: {client_ids}")

        # Start clients using docker compose run with different CIDs
        for cid in client_ids:
            env = {
                "CID": str(cid),
            }

            # Use 'run' instead of 'up' to create multiple instances with same service
            # --detach runs in background
            # The container name will be set by container_name in docker-compose.yml using ${CID}
            self._run_compose_command(["run", "--detach", "client"], env=env)

        logger.info("Clients started successfully")

    def start_all(self) -> None:
        """Start both server and all participating clients."""
        logger.info("Starting complete FL environment...")
        self.start_server(detached=True)
        self.start_clients()
        logger.info("FL environment started successfully")

    def stop_server(self) -> None:
        """Stop the federated learning server."""
        logger.info("Stopping FL server...")
        self._run_compose_command(["stop", "server"])
        logger.info("FL server stopped")

    def stop_clients(self, client_ids: Optional[List[int]] = None) -> None:
        """
        Stop federated learning clients.

        Args:
            client_ids: Specific client IDs to stop. If None, stops all clients.
        """
        if client_ids is None:
            # Stop all client containers that match the pattern
            logger.info("Stopping all clients...")
            # Use docker ps to find all client containers
            # nosec B603, B607: Command is constructed safely with validated inputs
            result = subprocess.run(  # noqa: S603, S607
                ["docker", "ps", "-a", "--filter", "name=fl_client-", "--format", "{{.Names}}"],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
            )
            container_names = result.stdout.strip().split("\n") if result.stdout.strip() else []

            for container_name in container_names:
                if container_name.startswith("fl_client-"):
                    # nosec B603, B607: Command is constructed safely
                    subprocess.run(["docker", "stop", container_name], check=False)  # noqa: S603, S607
                    subprocess.run(["docker", "rm", container_name], check=False)  # noqa: S603, S607
        else:
            logger.info(f"Stopping clients: {client_ids}")
            for cid in client_ids:
                container_name = f"fl_client-{cid}"
                # nosec B603, B607: Command is constructed safely
                subprocess.run(["docker", "stop", container_name], check=False)  # noqa: S603, S607
                subprocess.run(["docker", "rm", container_name], check=False)  # noqa: S603, S607

        logger.info("Clients stopped")

    def stop_all(self) -> None:
        """Stop all services (server and clients)."""
        logger.info("Stopping all FL services...")
        self._run_compose_command(["down"])
        logger.info("All FL services stopped")

    def restart_server(self) -> None:
        """Restart the federated learning server."""
        logger.info("Restarting FL server...")
        self._run_compose_command(["restart", "server"])
        logger.info("FL server restarted")

    def restart_clients(self, client_ids: Optional[List[int]] = None) -> None:
        """
        Restart federated learning clients.

        Args:
            client_ids: Specific client IDs to restart. If None, restarts all.
        """
        if client_ids is None:
            client_ids = self.participating_clients

        logger.info(f"Restarting clients: {client_ids}")
        for cid in client_ids:
            container_name = f"fl_client-{cid}"
            # nosec B603, B607: Command is constructed safely
            subprocess.run(["docker", "restart", container_name], check=False)  # noqa: S603, S607

        logger.info("Clients restarted")

    def get_logs(
        self,
        service: str = "server",
        follow: bool = False,
        tail: Optional[int] = None,
    ) -> None:
        """
        Get logs from a service.

        Args:
            service: Service name (e.g., 'server', 'client-0')
            follow: Follow log output
            tail: Number of lines to show from the end
        """
        cmd = ["logs"]
        if follow:
            cmd.append("-f")
        if tail is not None:
            cmd.extend(["--tail", str(tail)])
        cmd.append(service)

        self._run_compose_command(cmd, capture_output=False)

    def get_status(self) -> str:
        """
        Get status of all services.

        Returns:
            Status output from docker compose ps
        """
        result = self._run_compose_command(["ps"], capture_output=True)
        return result.stdout

    @property
    def is_server_running(self) -> bool:
        """Check if the server is currently running."""
        result = self._run_compose_command(
            ["ps", "--services", "--filter", "status=running"],
            capture_output=True,
            check=False,
        )
        return "server" in result.stdout

    @property
    def running_clients(self) -> List[str]:
        """Get list of currently running client containers."""
        # nosec B603, B607: Command is constructed safely
        result = subprocess.run(  # noqa: S603, S607
            ["docker", "ps", "--filter", "name=fl_client-", "--format", "{{.Names}}"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        container_names = result.stdout.strip().split("\n") if result.stdout.strip() else []
        return [name for name in container_names if name.startswith("fl_client-")]
