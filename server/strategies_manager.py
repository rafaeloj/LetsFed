import flwr as fl
from strategies.factory import ServerBuilder

from ..conf import Environment
from ..conf.loader import load_config


def main() -> None:
    cfg: Environment = load_config("/app/conf/config.yaml")
    fl.server.start_server(
        server_address=f"{cfg.server.ip}:{cfg.server.port}",
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=ServerBuilder.create(cfg),
    )


if __name__ == "__main__":
    main()
