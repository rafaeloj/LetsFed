from dataclasses import dataclass
from omegaconf import MISSING
from abc import ABC
@dataclass
class Base(ABC):
    method: str = MISSING

@dataclass
class Avg(Base):
    pre_training_epochs: int = MISSING

@dataclass
class MaxFL(Base):
    g_learning_rate: float = MISSING
    epsilon: float = MISSING
    rho: float = MISSING
    pre_training_epochs: int = MISSING

@dataclass
class LetsFed(Base):
    participating: Base = MISSING
    non_participating: Base = MISSING

@dataclass
class DEEV(Base):
    decay: float = MISSING

@dataclass
class POC(Base):
    perc_of_clients: float = MISSING

@dataclass
class Random(Base):
    perc_of_clients: float = MISSING

@dataclass
class Server:
    aggregation: Avg | MaxFL = MISSING
    selection: DEEV | POC | Random | LetsFed = MISSING
    ip: str = MISSING
    port: int = MISSING
@dataclass
class Client:
    threshold: float = MISSING
    participating: bool = MISSING
    epochs: int = MISSING
    learning_rate_client: float = MISSING
    training_strategy: str = MISSING

@dataclass
class DirichletPartitioner(Base):
    dirichlet_alpha: float = MISSING
    partition_by: str = MISSING
    self_balacing: bool = MISSING
    min_partition_size: int = MISSING
    shuffle: bool = MISSING
@dataclass
class IidPartitioner(Base):
    pass

@dataclass
class Partitioner:
    train: DirichletPartitioner | IidPartitioner = MISSING
    test: DirichletPartitioner | IidPartitioner = MISSING

@dataclass
class Database:
    dataset: str = MISSING
    partitioner: Partitioner = MISSING
    path: str = MISSING


@dataclass
class Environment:
    rounds: int = MISSING
    gpu: bool = MISSING
    init_clients: float = MISSING
    n_clients: int = MISSING
    model_type: str = MISSING
    model_path: str = MISSING
    server: Server = MISSING
    client: Client = MISSING
    db: Database = MISSING