from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner, DirichletPartitioner, IidPartitioner
from flwr_datasets.utils import divide_dataset
from datasets import load_dataset, load_from_disk
import os
from typing import List, Dict
class DSManager():
    def __init__(self, partitioner: Dict[str, Partitioner]):
        self.train_partitioner: Partitioner = partitioner['test']
        self.test_partitioner: Partitioner = partitioner['train']
        
    def load_csv(self, path, division=List[float]):
        self.dataset = load_dataset('csv', data_files=path)['train']
        train, test = divide_dataset(self.dataset, division=division)

        self.train_partitioner.dataset = train
        self.test_partitioner.dataset = test
        self.train = train
        self.test = test

    def load_hugginface(self, dataset):
        self.fds_train = FederatedDataset(dataset=dataset.lower(), partitioners={'train': self.train_partitioner})
        self.fds_test = FederatedDataset(dataset=dataset.lower(), partitioners={'test': self.test_partitioner})
        
        self.train = self.fds_train.load_split('train')
        self.train_partitioner.dataset = self.train
        self.test = self.fds_test.load_split('test')
        self.test_partitioner.dataset = self.test
    
    def load_train_partition(self, partition_id:int):
        return self.train_partitioner.load_partition(partition_id = partition_id)

    def load_test_partition(self, partition_id: int):
        return self.test_partitioner.load_partition(partition_id = partition_id)

    def save_locally(self, path):
        for cid in range(self.train_partitioner.num_partitions):
            self.train_partitioner.load_partition(cid).with_format("numpy").save_to_disk(f'{path}/{cid}-data-train')
        for cid in range(self.test_partitioner.num_partitions):
            self.train_partitioner.load_partition(cid).with_format("numpy").save_to_disk(f'{path}/{cid}-data-test')
if __name__ == "__main__":
    ds = DSManager({
            'train': IidPartitioner(num_partitions=10),
            'test': IidPartitioner(num_partitions=10)
        }
    )
    ds.load_hugginface('cifar10')
    # print(ds.fds_train.)
    ds.save_locally('logs')

