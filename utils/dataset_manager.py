from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from datasets import load_from_disk
import os

from conf.config import Database

class DSManager():
    def __init__(self, n_clients: int, conf: Database):
        self.path = self._get_path_name(conf, n_clients)
  
        self.train_partitioner = self.set_train_partitioner(n_clients = n_clients, conf = conf)
        self.test_partitioner = self.set_test_partitioner(n_clients = n_clients, conf = conf)

    def _get_path_name(self, conf: Database, n_clients: int):
        path = f'{conf.path}/{conf.dataset}/{n_clients}'
        if conf.partitioner.train.method == 'dirichlet':
            path = f'{path}/{conf.partitioner.train.dirichlet_alpha}'
        if conf.partitioner.test.method == 'dirichlet':
            path = f'{path}/{conf.partitioner.test.dirichlet_alpha}'
        return path
    def set_train_partitioner(self, n_clients: int, conf: Database):
        if conf.partitioner.train.method == 'dirichlet':
            return DirichletPartitioner(
                num_partitions = n_clients,
                alpha = conf.partitioner.train.dirichlet_alpha,
                partition_by = conf.partitioner.train.partition_by,
                min_partition_size = conf.partitioner.train.min_partition_size,
                self_balancing = conf.partitioner.train.self_balacing,
                shuffle = conf.partitioner.train.shuffle,
            )
        elif conf.partitioner.train.method == 'iid':
            return IidPartitioner(num_partitions = n_clients)
        raise ValueError(f"Paritioner not implemented: {conf.partitioner.train.method}")
    
    def set_test_partitioner(self, n_clients: int, conf: Database):
        if conf.partitioner.test.method == 'dirichlet':

            return DirichletPartitioner(
                num_partitions = n_clients,
                alpha = conf.partitioner.test.dirichlet_alpha,
                partition_by = conf.partitioner.test.partition_by,
                min_partition_size = conf.partitioner.test.min_partition_size,
                self_balancing = conf.partitioner.test.self_balacing,
                shuffle = conf.partitioner.test.shuffle,
            )
        elif conf.partitioner.test.method == 'iid':
            return IidPartitioner(num_partitions = n_clients)
        raise ValueError(f"Paritioner not implemented: {conf.partitioner.test.method}")

    def load(self, dataset):
        self.fds = FederatedDataset(dataset=dataset.lower(), partitioners={'train': self.train_partitioner, 'test': self.test_partitioner})        
        self.train = self.fds.load_split('train')
        self.test = self.fds.load_split('test')
    
    def load_train_partition(self, partition_id: int):
        if not self.fds:
            raise "Dataset not loaded"
        return self.fds.load_partition(partition_id = partition_id, split = 'train')

    def load_test_partition(self, partition_id: int):
        if not self.fds:
            raise "Dataset not loaded"
        return self.fds.load_partition(partition_id = partition_id, split = 'test')

    def load_locally(self, partition_id: int, path: str = None):
        print("LOAD DATA FROM LOCAL STORAGE")
        p = f'/{self.path}'
        if path:
            p = path.lower()
        test_folder = os.path.exists(f'{p}/test/{partition_id}')
        if test_folder:
            test                           = load_from_disk(f'{p}/test/{partition_id}')
            self.test                      = test
            self.test_partitioner.dataset  = test
        train_folder = os.path.exists(f'{p}/train/{partition_id}')

        if train_folder:
            train                          = load_from_disk(f'{p}/train/{partition_id}')
            self.train                     = train
            self.train_partitioner.dataset = train

        if not train_folder or not test_folder:
            raise ValueError(f"Error into load dataset of cid: {partition_id}")
        
        return test, train
        
    def save_locally(self, path: str = None):
        p = self.path
        if path:
            p = path.lower()
        for cid in range(self.fds.partitioners['train'].num_partitions):
            print(cid)
            self.fds.load_partition(cid, 'train').with_format("numpy").save_to_disk(f'{p}/train/{cid}')
        for cid in range(self.fds.partitioners['test'].num_partitions):
            self.fds.load_partition(cid, 'test').with_format("numpy").save_to_disk(f'{p}/test/{cid}')

