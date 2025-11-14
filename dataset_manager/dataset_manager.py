import os
from typing import Tuple

from datasets import load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr_datasets.utils import divide_dataset
from flwr_datasets.visualization import plot_comparison_label_distribution

from ..utils.logger import Logger
from .structs import DatasetConfig

logger = Logger(__name__)


class DSManager:
    def __init__(self, n_clients: int, conf: DatasetConfig) -> None:
        self.path = self._get_path_name(conf, n_clients)
        self.conf = conf
        self.train_partitioner = self.set_train_partitioner(n_clients=n_clients, conf=conf)
        self.test_partitioner = self.set_test_partitioner(n_clients=n_clients, conf=conf)
        logger.info(f"DSManager initialized for {n_clients} clients with dataset '{conf.dataset}'")

    def _get_path_name(self, conf: DatasetConfig, n_clients: int) -> str:
        """
        Get the path name for the dataset based on configuration and client number.

        Args:
            conf: Dataset configuration
            n_clients: Number of clients

        Returns:
            str: Path name
        """
        path = f"{conf.path}/{conf.dataset}/{n_clients}"
        if conf.train_partitioner.method == "dirichlet":
            path = f"{path}/{conf.train_partitioner.dirichlet_alpha}"
        if conf.test_partitioner.method == "dirichlet":
            path = f"{path}/{conf.test_partitioner.dirichlet_alpha}"
        logger.debug(f"Dataset path: {path}")
        return path

    def set_train_partitioner(
        self, n_clients: int, conf: DatasetConfig
    ) -> DirichletPartitioner | IidPartitioner:
        """
        Set the training data partitioner.

        Args:
            n_clients: Number of clients
            conf: Dataset configuration

        Returns:
            DirichletPartitioner or IidPartitioner
        """
        if conf.train_partitioner.method == "dirichlet":
            logger.info(
                "Using Dirichlet partitioner for training data "
                + f"(alpha={conf.train_partitioner.dirichlet_alpha})"
            )
            return DirichletPartitioner(
                num_partitions=n_clients,
                alpha=conf.train_partitioner.dirichlet_alpha,
                partition_by=conf.train_partitioner.partition_by,
                min_partition_size=conf.train_partitioner.min_partition_size,
                self_balancing=conf.train_partitioner.self_balancing,
                shuffle=conf.train_partitioner.shuffle,
            )
        elif conf.train_partitioner.method == "iid":
            logger.info("Using IID partitioner for training data")
            return IidPartitioner(num_partitions=n_clients)
        raise ValueError(f"Paritioner not implemented: {conf.train_partitioner.method}")

    def set_test_partitioner(
        self, n_clients: int, conf: DatasetConfig
    ) -> DirichletPartitioner | IidPartitioner:
        """
        Set the test data partitioner.

        Args:
            n_clients: Number of clients
            conf: Dataset configuration

        Returns:
            DirichletPartitioner or IidPartitioner
        """
        if conf.test_partitioner.method == "dirichlet":
            logger.info(
                "Using Dirichlet partitioner for test data "
                + f"(alpha={conf.test_partitioner.dirichlet_alpha})"
            )
            return DirichletPartitioner(
                num_partitions=n_clients,
                alpha=conf.test_partitioner.dirichlet_alpha,
                partition_by=conf.test_partitioner.partition_by,
                min_partition_size=conf.test_partitioner.min_partition_size,
                self_balancing=conf.test_partitioner.self_balancing,
                shuffle=conf.test_partitioner.shuffle,
            )
        elif conf.test_partitioner.method == "iid":
            logger.info("Using IID partitioner for test data")
            return IidPartitioner(num_partitions=n_clients)
        raise ValueError(f"Paritioner not implemented: {conf.test_partitioner.method}")

    def load(self, dataset: str) -> None:
        """
        Load dataset and apply partitioners.

        Args:
            dataset: Name of the dataset to load
        """
        logger.info(f"Loading dataset '{dataset}' with partitioners")
        self.fds = FederatedDataset(
            dataset=dataset.lower(),
            partitioners={"train": self.train_partitioner, "test": self.test_partitioner},
        )
        self.train = self.fds.load_split("train")
        self.test = self.fds.load_split("test")
        logger.info(f"Dataset '{dataset}' loaded successfully")

    def load_train_partition(self, partition_id: int) -> Tuple:
        """
        Load training partition.

        Args:
            partition_id: Partition ID (client ID)

        Returns:
            Tuple of (train_x, train_y)
        """
        if not self.fds:
            raise Exception("Dataset not loaded")
        logger.debug(f"Loading training partition {partition_id}")
        return self.fds.load_partition(partition_id=partition_id, split="train")

    def load_test_partition(self, partition_id: int) -> Tuple:
        """
        Load test partition.

        Args:
            partition_id: Partition ID (client ID)

        Returns:
            Tuple of (test_x, test_y)
        """
        if not self.fds:
            raise Exception("Dataset not loaded")
        logger.debug(f"Loading test partition {partition_id}")
        return self.fds.load_partition(partition_id=partition_id, split="test")

    def load_locally(self, partition_id: int, path: str = None) -> Tuple[object, object, object]:
        """
        Load dataset from local storage.

        Args:
            partition_id (int): CID
            path (str, optional): Custom path to load data from

        Returns:
            Tuple[Dataset,Dataset,Dataset]: Train, Validation, Test
        """
        logger.info(f"Loading data from local storage for partition {partition_id}")
        # Use relative path (works both in host and container with WORKDIR /app)
        p = self.path
        if path:
            p = path.lower()

        logger.debug(f"Dataset base path: {p}")
        test_folder = os.path.exists(f"{p}/test/{partition_id}")
        train_folder = os.path.exists(f"{p}/train/{partition_id}")
        validation_folder = os.path.exists(f"{p}/validation/{partition_id}")

        if test_folder:
            logger.debug(f"Loading test data for partition {partition_id}")
            test = load_from_disk(f"{p}/test/{partition_id}")
            self.test = test
            self.test_partitioner.dataset = test

        if train_folder:
            logger.debug(f"Loading train data for partition {partition_id}")
            train = load_from_disk(f"{p}/train/{partition_id}")
            self.train = train
            self.train_partitioner.dataset = train

        if validation_folder:
            logger.debug(f"Loading validation data for partition {partition_id}")
            validation = load_from_disk(f"{p}/validation/{partition_id}")

        if not train_folder or not test_folder or not validation_folder:
            logger.error(f"Failed to load dataset for partition {partition_id}")
            raise ValueError(f"Error into load dataset of cid: {partition_id}")

        logger.info(f"Successfully loaded local dataset for partition {partition_id}")
        return train, validation, test

    def save_locally(self, path: str = None) -> None:
        """
        Save dataset partitions to local storage.

        Args:
            path (str, optional): Custom path to save data
        """
        p = self.path
        if path:
            p = path.lower()

        logger.info(f"Saving dataset partitions to {p}")
        num_partitions = self.fds.partitioners["train"].num_partitions

        for cid in range(num_partitions):
            logger.debug(f"Saving partition {cid}/{num_partitions}")
            train, validation = divide_dataset(
                dataset=self.fds.load_partition(cid, "train"), division=[0.8, 0.2]
            )
            test = self.fds.load_partition(cid, "test")
            test.with_format("numpy").save_to_disk(f"{p}/test/{cid}")
            train.with_format("numpy").save_to_disk(f"{p}/train/{cid}")
            validation.with_format("numpy").save_to_disk(f"{p}/validation/{cid}")

        logger.info("Generating partition distribution visualization")
        fig, _, _ = plot_comparison_label_distribution(
            partitioner_list=[self.train_partitioner, self.test_partitioner],
            label_name="label",
            subtitle=f"Comparison of Partitioning Schemes on {self.conf.dataset.upper()}",
            titles=["Train distribution", "Test distribution"],
            legend=True,
            verbose_labels=False,
        )
        fig.savefig(f"{p}/partition_distributions.png", format="png")
        logger.info(f"Dataset partitions saved successfully to {p}")
