from .drivers import Driver
from .states import NonParticipationState, ParticipationState, ClientState
from utils import ModelManager, DSManager, my_logger
from .training import TrainingStrategy
from abc import ABC
from keras import Model
from typing import Dict, Tuple, List
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)
from flwr_datasets.utils import divide_dataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
import flwr as fl

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, config: Config):
        self.participating_state     : bool             = True
        self.participation_state     : ClientState      = ParticipationState()
        self.non_participation_state : ClientState      = NonParticipationState()
        self.state                   : ClientState      = self.participation_state
        self.drivers                 : List[Driver]     = []
        self.strategy                : TrainingStrategy = None
        self.conf                    : Config           = config
        self.conf['selected'] = False
        self.load_data()
        self.load_model()
    
    def load_model(self):
        self.model_type = self.conf['model_type']
        mm = ModelManager(
            model_type = self.model_type,
            input_shape = self.x_train.shape
        )
        self.model = mm.get_model(self.model_type)
        self.debug_model: Model = None
    
    def load_data(self):
        dm = DSManager({
            "train": DirichletPartitioner(
                num_partitions = self.conf['num_clients'],
                partition_by = "label",
                alpha = self.conf['dirichlet_alpha'],
                self_balancing = False
            ),
            "test": IidPartitioner(num_partitions=self.conf['num_clients'])
        })
        path = f'logs/{self.conf['dataset']}/{self.conf['dirichlet_alpha']}/clients-{self.conf['num_clients']}'
        train, test = dm.load_locally(path, self.conf['cid'])

        keys = list(test.features.keys())
        train, validation = divide_dataset(dataset=train, division=[0.8, 0.2])
        self.x_train, self.y_train, self.x_validation, self.y_validation = train[keys[0]], train[keys[1]], validation[keys[0]], validation[keys[1]]
        self.x_test, self.y_test = test[keys[0]], test[keys[1]]

    def set_state(self, state: ClientState):
        self.state = state

    def add_driver(self, driver: Driver):
        self.drivers.append(driver)

    def participate(self):
        self.state.participate(self)

    def non_participate(self):
        self.state.non_participate(self)

    def apply_drivers(self, parameters: NDArrays, config: Config):
        answers = dict()
        for driver in self.drivers:
            answers[driver.get_name()] = driver.run(self, parameters, config)
        return answers
    
    def add_drivers(self, drivers):
        for driver in drivers:
            self.drivers.append(driver)

    def get_parameters(self, config: Config):
        return self.strategy.get_parameters(self, config)

    def set_strategy(self, strategy: TrainingStrategy):
        self.strategy = strategy

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        return self.strategy.fit(self, parameters, config)

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        parameters, size, conf = self.strategy.evaluate(self, parameters, config)
        my_logger.log(
            f'/c-data-{self.conf['sid']}.csv',
            data={
                **self.conf,
                'rounds': config['rounds'],
                'participating_state': self.participating_state,
                'selected': self.conf['selected']
            }
        )
        return parameters, size, conf
