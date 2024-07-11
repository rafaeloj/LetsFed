import os
from optparse import OptionParser
import configparser
import random
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

from utils import DSManager

def add_server_info(
    clients:              int,
    rounds:               int,
    dataset:              str,
    strategy:             str,
    perc_of_clients:      float,
    decay:                float,
    exploitation:         float,
    exploration:          float,
    least_select_factor:  float,
    select_client_method: str,
    engaged_clients:      str,
    gpu:                  bool,
    threshold:            float,
    local_epochs:    int,
    non_iid:          bool,
    dirichlet_alpha: float,
    swap:                 bool,
    model_type:                str,
    init_clients: float,
    config_test: str,
    select_client_method_to_engaged: str,
):
    server_str = f"  server:\n\
    {'image: server-flwr:latest' if gpu else 'image: server-flwr-cpu:latest'}\n\
    logging:\n\
      driver: local\n\
    {'runtime: nvidia' if gpu else ''}\n\
    container_name: rfl_server\n\
    profiles:\n\
      - server\n\
    environment:\n\
      - SERVER_IP=0.0.0.0:9999\n\
      - NUM_CLIENTS={clients}\n\
      - NUM_ROUNDS={rounds}\n\
      - DATASET={dataset}\n\
      - STRATEGY={strategy}\n\
      - LOCAL_EPOCHS={local_epochs}\n\
      - PERC_OF_CLIENTS={perc_of_clients}\n\
      - DECAY={decay}\n\
      - SWAP={swap}\n\
      - non_iid={non_iid}\n\
      - DIRICHLET_ALPHA={dirichlet_alpha}\n\
      - ENGAGED_CLIENTS={','.join([str(c) for c in engaged_clients])}\n\
      - SELECT_CLIENT_METHOD={select_client_method}\n\
      - EXPLOITATION={exploitation}\n\
      - EXPLORATION={exploration}\n\
      - LEAST_SELECT_FACTOR={least_select_factor}\n\
      - THRESHOLD={threshold}\n\
      - MODEL_TYPE={model_type}\n\
      - INIT_CLIENTS={init_clients}\n\
      - CONFIG_TEST={config_test}\n\
      - SELECT_CLIENT_METHOD_TO_ENGAGED={select_client_method_to_engaged}\n\
    volumes:\n\
      - ./logs:/logs:rw\n\
      - ./server/strategies_manager.py:/app/strategies_manager.py:r\n\
      - ./server/strategies/drivers:/server/strategies/drivers/:r\n\
      - ./server/strategies:/app/strategies/:r\n\
      - ./utils:/app/utils/:r\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      {'resources:' if gpu else ''}\n\
        {'reservations:' if gpu else ''}\n\
          {'devices:' if gpu else ''}\n\
            {'- capabilities: [gpu]' if gpu else ''}\n\
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
    non_iid:          bool,
    participate:     bool,
    dirichlet_alpha: float,
    select_client_method: str,
    swap:                 bool,
    rounds:               int,
    model_type:           str,
    gpu:                  bool,
    decay:                float,
    exploitation:         float,
    exploration:          float,
    least_select_factor:  float,
    threshold:            float,
    init_clients: float,
    config_test: str,
):
    client_str = f"  client-{client}:\n\
    {'image: client-flwr:latest' if gpu else 'image: client-flwr-cpu:latest'}\n\
    logging:\n\
      driver: local\n\
    {'runtime: nvidia' if gpu else ''}\n\
    container_name: rfl_client-{client}\n\
    profiles:\n\
      - client\n\
    environment:\n\
      - SERVER_IP=rfl_server:9999\n\
      - CLIENT_ID={client}\n\
      - NUM_CLIENTS={n_clients}\n\
      - DATASET={dataset}\n\
      - STRATEGY={strategy}\n\
      - LOCAL_EPOCHS={local_epochs}\n\
      - non_iid={non_iid}\n\
      - PARTICIPATE={participate}\n\
      - DIRICHLET_ALPHA={dirichlet_alpha}\n\
      - SELECT_CLIENT_METHOD={select_client_method}\n\
      - EXPLOITATION={exploitation}\n\
      - EXPLORATION={exploration}\n\
      - LEAST_SELECT_FACTOR={least_select_factor}\n\
      - DECAY={decay}\n\
      - SWAP={swap}\n\
      - ROUNDS={rounds}\n\
      - MODEL_TYPE={model_type}\n\
      - INIT_CLIENTS={init_clients}\n\
      - THRESHOLD={threshold}\n\
      - CONFIG_TEST={config_test}\n\
    volumes:\n\
      - ./logs:/logs:rw\n\
      - ./client/strategies:/client/strategies/:r\n\
      - ./client/strategies/drivers:/client/strategies/drivers/:r\n\
      - ./client/strategies_manager.py:/client/strategies_manager.py:r\n\
      - ./utils:/utils/:r\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      {'resources:' if gpu else ''}\n\
        {'reservations:' if gpu else ''}\n\
          {'devices:' if gpu else ''}\n\
            {'- capabilities: [gpu]' if gpu else ''}\n\
      placement:\n\
        constraints:\n\
          - node.role==worker\n\
          \n\n"

    return client_str

def start_clients(n_clients, init_clients) -> list:
    n_clients_to_start = int(n_clients*init_clients)
    return random.sample(range(n_clients), n_clients_to_start)


def load_data(dataset, non_iid, clients, dirichlet_alpha):
    if os.path.exists(f'logs/{dataset}/non_iid-{non_iid}/clients-{clients}'):
        return
    partitioner = IidPartitioner(num_partitions=clients)
    if non_iid:
      partitioner = DirichletPartitioner(
          num_partitions=clients,
          alpha=dirichlet_alpha,
          partition_by="label",
          self_balancing=False
      )
    ds = DSManager({
        'train': partitioner,
        'test': IidPartitioner(num_partitions=clients),
    })
    ds.load_hugginface(dataset)
    ds.save_locally(f'logs/{dataset}/non_iid-{non_iid}/clients-{clients}')

def main():
    parser = OptionParser()
    parser.add_option("-e", "--environment", dest="environment", type='str')
    parser.add_option("-g", "--gpu",action="store_true", dest="gpu")

    config = configparser.ConfigParser()

    (opt, _) = parser.parse_args()
    config.read('config.ini')

    clients                         = config.getint(opt.environment, 'clients', fallback=10)
    local_epochs                    = config.getint(opt.environment,'local_epochs', fallback=1)
    dataset                         = config.get(opt.environment, 'dataset')
    rounds                          = config.getint(opt.environment, 'rounds', fallback=10)
    strategy                        = config.get(opt.environment, 'strategy', fallback='CIA')
    non_iid                         = config.getboolean(opt.environment, 'non_iid', fallback=True)
    init_clients                    = config.getfloat(opt.environment, 'init_clients', fallback=0.2)
    dirichlet_alpha                 = config.getfloat(opt.environment, "dirichlet_alpha", fallback=0.1),
    select_client_method            = config.get(opt.environment, 'select_client_method', fallback='random')
    perc_of_clients                 = config.getfloat(opt.environment, "perc_of_clients", fallback=0.0),
    decay                           = config.getfloat(opt.environment, "decay", fallback=0.0),
    exploitation                    = config.getfloat(opt.environment, 'exploitation')
    exploration                     = config.getfloat(opt.environment, 'exploration')
    least_select_factor             = config.getfloat(opt.environment, 'least_select_factor', fallback=0.0)
    swap                            = config.getboolean(opt.environment, 'swap', fallback=True)
    model_type                      = config.get(opt.environment, 'model_type', fallback='dnn')
    threshold                       = config.getfloat(opt.environment, 'threshold', fallback=1)
    select_client_method_to_engaged = config.get(opt.environment, 'select_client_method_to_engaged', fallback=None)
    perc_of_clients      = perc_of_clients[0]
    dirichlet_alpha      = dirichlet_alpha[0]
    decay                = decay[0]
    # threshold            = threshold

    load_data(dataset=dataset, non_iid=non_iid, clients=clients, dirichlet_alpha=dirichlet_alpha)
    file = f'./init_clients-{init_clients}.txt'
    if os.path.exists(file):
        # print('IC from file')
        with open(file, 'r+') as f:
            lines = f.read()
            engaged_clients = [int(x) for x in lines.split(',')]
        # print(f'ic: {init_clients} -> {engaged_clients}')
    else:
        # print('Generating IC file')
        with open(file, 'w+') as f:
            engaged_clients = start_clients(clients, init_clients)
            f.write(','.join([str(c) for c in engaged_clients]))
        # print(f'ic: {init_clients} -> {engaged_clients}')
                      
    select_client_method = select_client_method if not select_client_method == None else 'none'
    file_name = f'dockercompose-{strategy}-{dataset}-{init_clients}-{select_client_method}-{select_client_method_to_engaged}-c{clients}-r{rounds}-le{local_epochs}-p{perc_of_clients:.2f}-exp{exploration:.2f}-lsf{least_select_factor:.2f}-dec{decay:.2f}-thr{threshold}.yaml'.lower()
    with open(f"{file_name}", 'w') as dockercompose_file:
        header = f"services:\n\n"
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
            gpu       = opt.gpu,
            threshold = threshold,
            local_epochs=local_epochs,
            non_iid=non_iid,
            dirichlet_alpha=dirichlet_alpha,
            model_type = model_type,
            swap = swap,
            init_clients = init_clients,
            config_test=opt.environment,
            select_client_method_to_engaged=select_client_method_to_engaged,
        )

        dockercompose_file.write(server_str)
        # print(f'Clients select to federation is: {engaged_clients}')
        for client in range(clients):
            participate = False
            if client in engaged_clients:
                participate = True
            client_str = add_client_info(
                client=client,
                n_clients=clients,
                dataset=dataset,
                strategy=strategy,
                local_epochs=local_epochs,
                non_iid=non_iid,
                participate=participate,
                dirichlet_alpha=dirichlet_alpha,
                select_client_method=select_client_method,
                swap = swap,
                rounds = rounds,
                model_type = model_type,
                gpu       = opt.gpu,
                decay = decay,
                exploitation = exploitation,
                exploration = exploration,
                least_select_factor = least_select_factor,
                threshold = threshold,
                init_clients = init_clients,
                config_test=opt.environment,
            )
            
            dockercompose_file.write(client_str)
    print(f"{file_name}")
if __name__ == "__main__":
    main()