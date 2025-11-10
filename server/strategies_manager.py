import flwr as fl

from ..conf.loader import load_config
from ..conf.structs import Environment
from .strategies.server_builder import ServerBuilder


def main() -> None:
    cfg: Environment = load_config("conf/config.yaml")
    fl.server.start_server(
        server_address=f"{cfg.server.ip}:{cfg.server.port}",
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=ServerBuilder.create(cfg),
    )


if __name__ == "__main__":
    main()
