import os
from optparse import OptionParser
import configparser
import random

def get_log_foulder_name(select_client_method, clients, engaged, dirichlet):
    match select_client_method:
        case 'random':
            return f'/random/{clients}/engaged_{engaged}/dirichlet_{dirichlet}'
        case 'worst_performance':
            return f'/worst/{clients}/engaged_{engaged}/dirichlet_{dirichlet}'
        case 'best_performance':
            return f'/best/{clients}/engaged_{engaged}/dirichlet_{dirichlet}'
        case 'least_selected':
            return f'/fair/{clients}/engaged_{engaged}/dirichlet_{dirichlet}'
        case 'deev':
            return f'/deev/{clients}/engaged_{engaged}/dirichlet_{dirichlet}'

def add_server_info(
    clients:              int,
    rounds:               int,
    dataset:              str,
    strategy:             str,
    perc_of_clients:      float,
    decay:                float,
    engaged_clients:      str,
    select_client_method: str,
    exploitation:         float,
    exploration:          float,
    least_select_factor:  float,
    foulder_log:          str,
):
    server_str = f"  server:\n\
    image: 'server-flwr:latest'\n\
    logging:\n\
      driver: local\n\
    container_name: fl_server\n\
    environment:\n\
      - SERVER_IP=0.0.0.0:9999\n\
      - NUM_CLIENTS={clients}\n\
      - NUM_ROUNDS={rounds}\n\
      - DATASET={dataset}\n\
      - STRATEGY={strategy}\n\
      - PERC_OF_CLIENTS={perc_of_clients}\n\
      - DECAY={decay}\n\
      - ENGAGED_CLIENTS={engaged_clients}\n\
      - SELECT_CLIENT_METHOD={select_client_method}\n\
      - EXPLOITATION={exploitation}\n\
      - EXPLORATION={exploration}\n\
      - LOG_FOULDER={foulder_log}\n\
      - LEAST_SELECT_FACTOR={least_select_factor}\n\
    volumes:\n\
      - ./logs{foulder_log}:/logs{foulder_log}:rw\n\
      - ./server/strategies_manager.py:/app/strategies_manager.py:r\n\
      - ./server/strategies:/app/strategies/:r\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      placement:\n\
        constraints:\n\
          - node.role==manager\n\
    \n\n"

    return server_str

def add_client_info(
    client:          str,
    n_clients:       int,
    dataset:         str,
    strategy:        str,
    local_epochs:    int,
    no_idd:          bool,
    participate:     bool,
    dirichlet_alpha: float,
    select_client_method: str,
    foulder_log:          str,
):
    client_str = f"  client-{client}:\n\
    image: 'client-flwr:latest'\n\
    logging:\n\
      driver: local\n\
    environment:\n\
      - SERVER_IP=fl_server:9999\n\
      - CLIENT_ID={client}\n\
      - NUM_CLIENTS={n_clients}\n\
      - DATASET={dataset}\n\
      - STRATEGY={strategy}\n\
      - LOCAL_EPOCHS={local_epochs}\n\
      - NO_IDD={no_idd}\n\
      - PARTICIPATE={participate}\n\
      - DIRICHLET_ALPHA={dirichlet_alpha}\n\
      - SELECT_CLIENT_METHOD={select_client_method}\n\
      - LOG_FOULDER={foulder_log}\n\
    volumes:\n\
      - ./logs{foulder_log}:/logs{foulder_log}:rw\n\
      - ./client/strategies_manager.py:/client/strategies_manager.py:r\n\
      - ./client/strategies:/client/strategies/:r\n\
      - ./client/strategies/engagement_strategy:/client/strategies/engagement_strategy/\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      placement:\n\
        constraints:\n\
          - node.role==worker\n\
          \n\n"

    return client_str

def start_clients(n_clients, init_clients) -> list:
    n_clients_to_start = int(n_clients*init_clients)
    return random.sample(range(n_clients), n_clients_to_start)

# Penetration rate
# variar a quantidade de clientes participando

"""
  --n_clients = 50 (FIXA)
  --init-clients = [100, 0, 25]
  --dirichlet-alpha = 0.1, 0.7
  --dataset = [MNIST, CIFAR10]
  --
  ## CIA
  -> Variar métodos de seleção

  ## POC
  --poc = 0.5
  ## DEEV
  --decay = 0.005
  ## FEDAVG

"""

def main():
    parser = OptionParser()
    parser.add_option("-e", "--environment", dest="environment", type='str')

    config = configparser.ConfigParser()

    (opt, _) = parser.parse_args()
    if 'debug' in opt.environment:
        config.read('config-debug.ini')
    elif 'FedCIA' in opt.environment:
        config.read('config-cia.ini')
    elif 'FedAVG' in opt.environment:
        config.read('config-avg.ini')
    elif 'FedPOC' in opt.environment:
        config.read('config-poc.ini')
    elif 'FedDEEV' in opt.environment:
        config.read('config-deev.ini')
    else:
        print("Environment not found!!")
        return


    clients              = config.getint(opt.environment, 'clients', fallback=10)
    local_epochs         = config.getint(opt.environment,'local_epochs', fallback=1)
    dataset              = config.get(opt.environment, 'dataset')
    rounds               = config.getint(opt.environment, 'rounds', fallback=10)
    strategy             = config.get(opt.environment, 'strategy', fallback='CIA')
    no_iid               = config.getboolean(opt.environment, 'no_iid', fallback=True)
    init_clients         = config.getfloat(opt.environment, 'init_clients', fallback=0.2)
    dirichlet_alpha      = config.getfloat(opt.environment, "dirichlet_alpha", fallback=0.1),
    select_client_method = config.get(opt.environment, 'select_client_method', fallback='random')
    perc_of_clients      = config.getfloat(opt.environment, "perc_of_clients", fallback=0.0),
    decay                = config.getfloat(opt.environment, "decay", fallback=0.0),
    exploitation         = config.getfloat(opt.environment, 'exploitation', fallback=0.0)
    exploration          = config.getfloat(opt.environment, 'exploration', fallback=0.0)
    least_select_factor  = config.getfloat(opt.environment, 'least_select_factor', fallback=0.0)
    perc_of_clients      = perc_of_clients[0]
    dirichlet_alpha      = dirichlet_alpha[0]
    decay                = decay[0]
  

    participate_clients = start_clients(clients, init_clients)
    engaged_clients = ','.join([str(x) for x in participate_clients])
    foulder_log = get_log_foulder_name(
        select_client_method=select_client_method,
        clients=clients,
        engaged=init_clients,
        dirichlet=dirichlet_alpha
    )
    file_name = f'dockercompose-{strategy}-{dataset}-c{clients}-r{rounds}.yaml'
    with open(file_name, 'w') as dockercompose_file:
        header = f"version: '3.8'\nservices:\n\n"

        dockercompose_file.write(header)

        server_str = add_server_info(
            clients=clients,
            rounds=rounds,
            dataset=dataset,
            strategy=strategy,
            perc_of_clients=perc_of_clients,
            decay=decay,
            engaged_clients=engaged_clients,
            select_client_method=select_client_method,
            exploitation=exploitation,
            exploration=exploration,
            least_select_factor=least_select_factor,
            foulder_log = foulder_log
        )

        dockercompose_file.write(server_str)
        print(f'Clients select to federation is: {participate_clients}')
        for client in range(clients):
            participate = False
            if client in participate_clients:
                participate = True
            client_str = add_client_info(
                client=client,
                n_clients=clients,
                dataset=dataset,
                strategy=strategy,
                local_epochs=local_epochs,
                no_idd=no_iid,
                participate=participate,
                dirichlet_alpha=dirichlet_alpha,
                select_client_method=select_client_method,
                foulder_log = foulder_log,
            )
            
            dockercompose_file.write(client_str)
    print(f"{file_name} created")
if __name__ == "__main__":
    main()