import logging

import flwr as fl

from ..conf.loader import load_config
from ..conf.structs import Environment
from .strategies.server_builder import ServerBuilder

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    cfg: Environment = load_config("app/conf/config.yaml")
    fl.server.start_server(
        server_address=f"{cfg.server.ip}:{cfg.server.port}",
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=ServerBuilder.create(cfg),
    )


if __name__ == "__main__":
    main()
