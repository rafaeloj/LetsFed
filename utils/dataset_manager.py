from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner, DirichletPartitioner, IidPartitioner
from flwr_datasets.utils import divide_dataset
from datasets import load_dataset, load_from_disk
import os
from typing import List, Dict
class DSManager():
    def __init__(self, partitioner: Dict[str, Partitioner]):
        self.train_partitioner: Partitioner = partitioner['train']
        self.test_partitioner: Partitioner = partitioner['test']
        self.fds = None

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

    def load_locally(self, path: str, partition_id: int):
        print("LOAD DATA FROM LOCAL STORAGE")
        path = path.lower()
        test_folder = os.path.exists(f'{path}/test/{partition_id}')
        if test_folder:
            test                           = load_from_disk(f'{path}/test/{partition_id}')
            self.test                      = test
            self.test_partitioner.dataset  = test
        train_folder = os.path.exists(f'{path}/train/{partition_id}')
        
        if train_folder:
            train                          = load_from_disk(f'{path}/train/{partition_id}')
            self.train                     = train
            self.train_partitioner.dataset = train
        if not train_folder or not test_folder:
            raise f"Error into load dataset of cid: {partition_id}"
        
        return test, train
        
    def save_locally(self, path: str):
        path = path.lower()
        for cid in range(self.fds.partitioners['train'].num_partitions):
            print(cid)
            self.fds.load_partition(cid, 'train').with_format("numpy").save_to_disk(f'{path}/train/{cid}')
        for cid in range(self.fds.partitioners['test'].num_partitions):
            self.fds.load_partition(cid, 'test').with_format("numpy").save_to_disk(f'{path}/test/{cid}')


if __name__ == "__main__":
    ds = DSManager({
            'train': IidPartitioner(num_partitions=10),
            'test': IidPartitioner(num_partitions=10)
        }
    )
    ds.load_hugginface('cifar10')
    # print(ds.fds_train.)
    ds.save_locally('logs')

