import os
from optparse import OptionParser
import random
def add_server_info(clients, rounds, dataset):
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
    volumes:\n\
      - ./logs:/logs:rw\n\
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
    client:       str,
    n_clients:    int,
    dataset:      str,
    strategy:     str,
    local_epochs: int,
    no_idd:       bool,
    participate:  bool,
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
    volumes:\n\
      - ./logs:/logs\n\
      - ./client/strategies_manager.py:/app/strategies_manager.py:r\n\
      - ./client/strategies/client.py:/app/strategies/client.py\n\
    networks:\n\
      - default\n\
    deploy:\n\
      replicas: 1\n\
      placement:\n\
        constraints:\n\
          - node.role==worker\n\
          \n\n"

    return client_str

def start_clients(n_clients, init_clients):
    return random.sample(range(n_clients), init_clients)

def main():
    parser = OptionParser()
    parser.add_option("-c",  "--clients",      dest="clients",      default=0,        type='int')
    parser.add_option("-e",  "--local-epochs", dest="local_epochs", default=1,        type='int')
    parser.add_option("-d",  "--dataset",      dest="dataset",      default='mnist', type='string')
    parser.add_option("-r",  "--rounds",       dest="rounds",       default=100,      type='int')
    parser.add_option("-s",  "--strategy",     dest="strategy",     default="CIA",    type='str')
    parser.add_option("",    "--no-iid",       dest="no_iid",       default=False,    action='store_true')
    parser.add_option("",    "--init-clients", dest="init_clients", default=2,        type='int')
    (opt, _) = parser.parse_args()

    participate_clients = start_clients(opt.clients, opt.init_clients)
    file_name = f'dockercompose-{opt.strategy}-{opt.dataset}.yaml'
    with open(file_name, 'w') as dockercompose_file:
        header = f"version: '3.8'\nservices:\n\n"

        dockercompose_file.write(header)

        server_str = add_server_info(
            opt.clients,
            opt.rounds,
            opt.dataset,
        )

        dockercompose_file.write(server_str)
        print(f'Clients select to federation is: {participate_clients}')
        for client in range(opt.clients):
            participate = False
            if client in participate_clients:
                participate = True
            client_str = add_client_info(
                client,
                opt.clients,
                opt.dataset,
                opt.strategy,
                opt.local_epochs,
                opt.no_iid,
                participate
            )
            
            dockercompose_file.write(client_str)
    print(f"{file_name} created")
if __name__ == "__main__":
    main()