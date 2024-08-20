import os
from optparse import OptionParser
import random
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
import json
from utils import DSManager, DockercomposeManager, ConfigManager

def start_clients(n_clients, init_clients) -> list:
    n_clients_to_start = int(n_clients*init_clients)
    return random.sample(range(n_clients), n_clients_to_start)


def load_data(dataset, clients, dirichlet_alpha):
    dm = DSManager({
            "train": DirichletPartitioner(
                num_partitions = clients,
                partition_by = "label",
                alpha = dirichlet_alpha,
                self_balancing = False
            ),
            "test": IidPartitioner(num_partitions=clients)
        })
    dm = DSManager({
            "train": IidPartitioner(num_partitions=clients),
            "test": IidPartitioner(num_partitions=clients)
        })
    dm.load(dataset)
    dm.save_locally(f'logs/{dataset}/{dirichlet_alpha}/clients-{clients}')


def main():
    parser = OptionParser()
    parser.add_option("-e", "--environment", dest="environment", type='int')
    parser.add_option("-g", "--gpu",action="store_true", dest="gpu")

    (opt, _) = parser.parse_args()
    conf_manager = ConfigManager("./config.json", 'utils/conf/variables.json')
    conf_manager.select_conf(opt.environment)
    config = conf_manager.get_conf_selected()

    file = f"./init_clients-{config['init_clients']}.txt"
    if os.path.exists(file):
        with open(file, 'r+') as f:
            lines = f.read()
            participating_clients = [int(x) for x in lines.split(',')]
    else:
        with open(file, 'w+') as f:
            participating_clients = start_clients(config['clients'], config['init_clients'])
            f.write(','.join([str(c) for c in participating_clients]))

    load_data(
        dataset=config['dataset'],
        clients=config['clients'],
        dirichlet_alpha=config['dirichlet_alpha']
    )
    
    compose_manager = DockercomposeManager(config, conf_manager.variables, False)

    file_name = compose_manager.create_dockercompose(participating_clients)

    print(f"{file_name}")
if __name__ == "__main__":
    main()