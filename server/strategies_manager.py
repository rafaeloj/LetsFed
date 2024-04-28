import flwr as fl
import os
from strategies.fedavg import FedAvg
from strategies.fedcia import FedCIA
from strategies.poc import FedPOC
from strategies.deev import FedDEEV

def get_strategy(
    strategy,
    n_clients,
    rounds,
    fraction_clients,
    perc_of_clients,
    decay,
    engaged_clients,
    select_client_method,
    exploitation: float,
    exploration: float,
    least_select_factor: float,
    log_foulder:         str,
):
    match strategy:
        case 'CIA':
            return FedCIA(
                n_clients            = n_clients,
                rounds               = rounds,
                engaged_clients      = engaged_clients,
                select_client_method = select_client_method,
                exploitation         = exploitation, # Seleção randomica precisa
                exploration          = exploration,
                least_select_factor  = least_select_factor, # Seleção 'justa'
                decay                = decay,
                log_foulder          = log_foulder
            )
        case 'POC':
            return FedPOC(
                n_clients        = n_clients,
                rounds           = rounds,
                fraction_clients = fraction_clients,
                perc_of_clients  = perc_of_clients,
            )
        case 'DEEV':
            return FedDEEV(n_clients=n_clients,
            rounds=rounds,
            fraction_clients=fraction_clients,
            perc_of_clients=perc_of_clients,
            decay=decay)
        case 'AVG':
            return FedAvg(n_clients=n_clients,
            rounds=rounds)
def main():
    strategy         = os.environ['STRATEGY']
    num_clients      = int(os.environ['NUM_CLIENTS'])
    rounds           = int(os.environ["NUM_ROUNDS"])
    fraction_clients = 1
    perc_of_client   = float(os.environ["PERC_OF_CLIENTS"])
    decay            = float(os.environ["DECAY"])
    select_client_method = os.environ["SELECT_CLIENT_METHOD"]
    engaged_clients  = [int(x) for x in os.environ['ENGAGED_CLIENTS'].split(',')]
    exploitation     = float(os.environ['EXPLOITATION'])
    exploration      = float(os.environ['EXPLORATION'])
    least_select_factor = float(os.environ['LEAST_SELECT_FACTOR'])
    foulder             = os.environ['LOG_FOULDER']
    for filename in os.listdir(foulder):
        os.remove(f'{foulder}/{filename}')
    fl.server.start_server(
        server_address=os.environ['SERVER_IP'],
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=get_strategy(
            strategy             = strategy,
            n_clients            = num_clients,
            rounds               = rounds,
            fraction_clients     = fraction_clients,
            perc_of_clients      = perc_of_client,
            decay                = decay,
            select_client_method = select_client_method,
            engaged_clients      = engaged_clients,
            exploitation         = exploitation,
            exploration          = exploration,
            least_select_factor  = least_select_factor,
            log_foulder          = foulder
        )
    )

if __name__ == '__main__':
    main()