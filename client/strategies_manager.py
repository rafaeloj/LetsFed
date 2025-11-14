import logging
import os

import flwr as fl
from omegaconf import OmegaConf

from ..conf.structs import Environment
from .strategies.client_builder import ClientBuilder

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    cfg: Environment = OmegaConf.load("app/conf/config.yaml")
    server_address = os.environ.get("SERVER_ADDRESS")

    fl.client.start_client(
        server_address=f"{server_address}:{cfg.server.port}",
        client=ClientBuilder.create(cid=os.environ.get("CID"), config=cfg).to_client(),
    )


if __name__ == "__main__":
    main()
