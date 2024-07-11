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
    non_iid,
    dataset,
    threshold,
    model_type,
    init_clients,
    config_test: str,
    select_client_method_to_engaged: str,
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
                non_iid = non_iid,
                dataset = dataset,
                threshold = threshold,
                model_type = model_type,
                init_clients = init_clients,
                config_test = config_test,
                select_client_method_to_engaged = select_client_method_to_engaged,
            )
        if strategy == 'POC':
            return FedPOC(
                n_clients        = n_clients,
                rounds           = rounds,
                fraction_clients = fraction_clients,
                perc_of_clients  = exploration,
                dataset= dataset,
                dirichlet_alpha= dirichlet_alpha,
                epoch= epoch,
                non_iid=non_iid,
                threshold=threshold,
                model_type = model_type,
                init_clients = init_clients,
                config_test = config_test,
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
                non_iid = non_iid,
                dataset = dataset,
                threshold = threshold,
                model_type = model_type,
                init_clients = init_clients,
                config_test = config_test,
            )
        if strategy == 'AVG':
            return FedAvg(
                    n_clients = n_clients,
                    rounds = rounds,
                    perc = exploration,
                epoch = epoch,
                dirichlet_alpha = dirichlet_alpha,
                non_iid = non_iid,
                dataset = dataset,
                threshold = threshold,
                model_type = model_type,
                init_clients = init_clients,
                config_test = config_test,
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
    non_iid              = os.environ["non_iid"] == "True" 
    threshold           = float(os.environ['THRESHOLD'])
    dataset             = os.environ['DATASET']
    model_type          = os.environ['MODEL_TYPE']
    init_clients        = float(os.environ['INIT_CLIENTS'])
    config_test         = os.environ['CONFIG_TEST']
    select_client_method_to_engaged = os.environ['SELECT_CLIENT_METHOD_TO_ENGAGED']
    my_logger.log(
        '/s-teste.csv',
        data = {
             'rounds': 0,
             'server': 'on'
        },
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
            non_iid = non_iid,
            threshold = threshold,
            dataset = dataset,
            model_type           = model_type,
            init_clients = init_clients,
            config_test = config_test,
            select_client_method_to_engaged = select_client_method_to_engaged,
        )
    )

if __name__ == '__main__':
    main()