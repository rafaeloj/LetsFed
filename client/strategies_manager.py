import os

import flwr as fl
from omegaconf import OmegaConf

from ..conf import Environment
from .strategies.factory import ClientBuilder


def main() -> None:
    cfg: Environment = OmegaConf.load("/client/conf/config.yaml")
    fl.client.start_client(
        server_address=f"rfl_server:{cfg.server.port}",
        client=ClientBuilder.create(cid=os.environ.get("CID"), config=cfg).to_client(),
    )


if __name__ == "__main__":
    main()
