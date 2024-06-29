import flwr as fl
import os
from strategies.fedavg import FedAvg
from strategies.fedcia import FedCIA
from strategies.poc import FedPOC
from strategies.deev import FedDEEV
from utils.logger import my_logger

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
    epoch,
    dirichlet_alpha,
    no_iid,
    dataset,
    threshold,
):
        if strategy == 'CIA':
            return FedCIA(
                n_clients            = n_clients,
                rounds               = rounds,
                engaged_clients      = engaged_clients,
                select_client_method = select_client_method,
                exploitation         = exploitation, # Seleção randomica precisa
                exploration          = exploration,
                least_select_factor  = least_select_factor, # Seleção 'justa'
                decay                = decay,
                solution             = strategy,
                epoch = epoch,
                dirichlet_alpha = dirichlet_alpha,
                no_iid = no_iid,
                dataset = dataset,
                threshold = threshold,
            )
        if strategy == 'POC':
            return FedPOC(
                n_clients        = n_clients,
                rounds           = rounds,
                fraction_clients = fraction_clients,
                perc_of_clients  = perc_of_clients,
                dataset= dataset,
                dirichlet_alpha= dirichlet_alpha,
                epoch= epoch,
                no_iid=no_iid,
                threshold=threshold,
            )
        if strategy == 'DEEV':
            return FedDEEV(
                n_clients        = n_clients,
                rounds           = rounds,
                fraction_clients = fraction_clients,
                perc_of_clients  = perc_of_clients,
                decay            = decay,
                epoch = epoch,
                dirichlet_alpha = dirichlet_alpha,
                no_iid = no_iid,
                dataset = dataset,
                threshold = threshold,
            )
        if strategy == 'AVG':
            return FedAvg(
                    n_clients = n_clients,
                    rounds = rounds,
                    perc = perc_of_clients,
                epoch = epoch,
                dirichlet_alpha = dirichlet_alpha,
                no_iid = no_iid,
                dataset = dataset,
                threshold = threshold,
            )
def main():
    strategy         = os.environ['STRATEGY']
    num_clients      = int(os.environ['NUM_CLIENTS'])
    rounds           = int(os.environ["NUM_ROUNDS"])
    fraction_clients = 1
    perc_of_client   = float(os.environ["PERC_OF_CLIENTS"])
    decay            = float(os.environ["DECAY"])
    exploitation     = float(os.environ['EXPLOITATION'])
    exploration      = float(os.environ['EXPLORATION'])
    least_select_factor = float(os.environ['LEAST_SELECT_FACTOR'])
    select_client_method = os.environ["SELECT_CLIENT_METHOD"]
    engaged_clients     = [int(x) for x in os.environ['ENGAGED_CLIENTS'].split(',')]
    local_epochs        = int(os.environ['LOCAL_EPOCHS'])
    dirichlet_alpha     = float(os.environ["DIRICHLET_ALPHA"])
    no_iid              = os.environ["NO_IID"] == "True" 
    threshold           = float(os.environ['THRESHOLD'])
    dataset             = os.environ['DATASET']
    my_logger.log(
        '/s-teste.csv',
        data = [0, 'TESTE'],
        header = ['round', 'server_selection'],
    )
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
            epoch  = local_epochs,
            dirichlet_alpha = dirichlet_alpha,
            no_iid = no_iid,
            threshold = threshold,
            dataset = dataset,
        )
    )

if __name__ == '__main__':
    main()