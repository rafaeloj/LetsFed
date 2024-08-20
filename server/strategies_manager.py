import flwr as fl
import os
from strategies.fedavg import FedAvg
from strategies import LetsFed
from strategies import FedPOC
from strategies import FedDEEV
from strategies import FedRR
from strategies import MaxFL
from utils import my_logger, ConfigManager
from typing import Dict
import json

def get_strategy(config: ConfigManager):
        if os.environ["_STRATEGY_SID"].upper() == 'LETSFED':
            return LetsFed(config.get_letsfed_config())
        if os.environ["_STRATEGY_SID"].upper() == 'POC':
            return FedPOC(config.get_poc_config())
        if os.environ["_STRATEGY_SID"].upper() == 'DEEV':
            return FedDEEV(config.get_deev_config())
        if os.environ["_STRATEGY_SID"].upper() == 'AVG':
            return FedAvg(config.get_avg_config())
        if os.environ["_STRATEGY_SID"].upper() == "R_ROBIN":
            return FedRR(config.get_rr_config())
        if os.environ['_STRATEGY_SID'].upper() == "MAXFL":
            return MaxFL(config.get_maxfl_config())
def main():
    config_manager = ConfigManager('')

    my_logger.log(
        '/s-teste.csv',
        data = {
             'rounds': 0,
             'server': 'on'
        },
    )

    fl.server.start_server(
        server_address=os.environ['SERVER_IP'],
        config=fl.server.ServerConfig(num_rounds=int(os.environ["_ROUNDS"])),
        strategy=get_strategy(config_manager)
    )

if __name__ == '__main__':
    main()