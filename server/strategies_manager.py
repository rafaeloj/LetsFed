import flwr as fl
from conf import Environment
from strategies import FLServer
from strategies.client_selection_method import DEEV, RandomSelection, POC, RoundRobinSelection, LetsFedSelection
from strategies.aggregate_method import FedAVG, MaxFL

from omegaconf import OmegaConf


def get_selection_method(selection_method: str, cfg: Environment = None):
    print(selection_method)
    if selection_method.upper() == 'POC':
        return POC()

    if selection_method.upper() == 'DEEV':
        return DEEV()

    if selection_method.upper() == 'RANDOM':
        return RandomSelection()

    if selection_method.upper() == "ROUND_ROBIN":
        return RoundRobinSelection()

    if selection_method.upper() == "LETSFED":
        return LetsFedSelection(
            participating = get_selection_method(cfg.server.selection.participating.method),
            non_participating = get_selection_method(cfg.server.selection.non_participating.method),
        )
    raise ValueError(f"Selection method not found: {selection_method}")

def get_aggregation_method(agg_method: str):
    if agg_method.upper() == 'AVG':
        return FedAVG()

    if agg_method.upper() == "MAXFL":
        return MaxFL()

def main():
    cfg: Environment = OmegaConf.load("/app/conf/config.yaml")
    fl.server.start_server(
        server_address=f'{cfg.server.ip}:{cfg.server.port}',
        config=fl.server.ServerConfig(num_rounds = cfg.rounds),
        strategy=FLServer(
            get_selection_method(cfg.server.selection.method, cfg),
            get_aggregation_method(cfg.server.aggregation.method),
            cfg,
        )
    )

if __name__ == '__main__':
    main()