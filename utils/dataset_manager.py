from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from datasets import load_from_disk
import os
from typing import Tuple
from flwr_datasets.utils import divide_dataset
from flwr_datasets.visualization import plot_comparison_label_distribution
from conf.config import Database
class DSManager():
    def __init__(self, n_clients: int, conf: Database):
        self.path = self._get_path_name(conf, n_clients)
        self.conf = conf
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

    def load_locally(self, partition_id: int, path: str = None) -> Tuple[Database,Database,Database]:
        """
        Args:
            partition_id (int): CID
            path (str, optional):

        Returns:
            Tupla[Dataset,Dataset,Dataset]: Train, Validation, Test
        """
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

        validation_folder = os.path.exists(f'{p}/validation/{partition_id}')
        if validation_folder:
            validation = load_from_disk(f'{p}/validation/{partition_id}')


        if not train_folder or not test_folder or not validation_folder:
            raise ValueError(f"Error into load dataset of cid: {partition_id}")
        
        return train, validation, test
        
    def save_locally(self, path: str = None) -> None:
        p = self.path
        if path:
            p = path.lower()
        
        for cid in range(self.fds.partitioners['train'].num_partitions):
            train, validation = divide_dataset(dataset=self.fds.load_partition(cid, 'train'), division=[0.8, 0.2])
            test = self.fds.load_partition(cid, 'test')
            test.with_format("numpy").save_to_disk(f'{p}/test/{cid}')
            train.with_format("numpy").save_to_disk(f'{p}/train/{cid}')
            validation.with_format("numpy").save_to_disk(f'{p}/validation/{cid}')
        fig, _, _ = plot_comparison_label_distribution(
            partitioner_list=[self.train_partitioner, self.test_partitioner],
            label_name="label",
            subtitle=f"Comparison of Partitioning Schemes on {self.conf.dataset.upper()}",
            titles=["Train distribution", "Test distribution"],
            legend=True,
            verbose_labels=False,
        )
        fig.savefig(f'{p}/partition_distributions.png', format='png')