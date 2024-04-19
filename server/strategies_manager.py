import flwr as fl
import os
from flwr.server.strategy import FedAvg
from strategies.fedcia import FedCia

def get_strategy(strategy, num_clients, rounds):
    match strategy:
        case 'CIA':
            return FedCia(num_clients, rounds)
        case _:
            return FedAvg()
def main():
    num_clients = int(os.environ['NUM_CLIENTS'])
    rounds      = int(os.environ["NUM_ROUNDS"])
    fl.server.start_server(
        server_address=os.environ['SERVER_IP'],
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=get_strategy("CIA", num_clients, rounds)
    )

if __name__ == '__main__':
    main()