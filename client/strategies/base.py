import sys
import os
sys.path.append(os.path.abspath('../../'))

from .drivers import Driver
from .states import NonParticipationState, ParticipationState, ClientState
from utils import ModelManager, DSManager, my_logger
from .training import TrainingStrategy, LestFedClient, MaxFLClient, NormalClient
from conf import Environment
from keras import Model
from typing import Dict, Tuple, List
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)
from flwr_datasets.utils import divide_dataset
import flwr as fl

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, cid: str, config: Environment, training_strategy: TrainingStrategy = None):
        self.data_to_log = {}
        self.cid                     : str              = cid
        self.conf                    : Environment      = config
        self.load_data()
        self.load_model()
        
        
        self.participating_state     : bool             = True
        self.desired_state           : bool             = True
        self.participation_state     : ClientState      = ParticipationState()
        self.non_participation_state : ClientState      = NonParticipationState()
        self.state                   : ClientState      = self.participation_state
        self.drivers                 : List[Driver]     = []
        self.strategy                : TrainingStrategy = self.get_strategy()
        self.selected = False
        self.g_eval_acc : float = 0
        self.g_fit_acc  : float = 0 
        self.g_eval_loss: float = 0
        self.g_fit_loss : float = 0
        self.diff       : float = 0

    def get_strategy(self):
        if self.conf.client.training_strategy.lower() == "maxfl":
            return MaxFLClient(self)
        if self.conf.client.training_strategy.lower() == "letsfed":
            return LestFedClient(self)
        if self.conf.client.training_strategy.lower() == 'normal':
            return NormalClient(self)

    def load_model(self):
        self.conf.model_type
        mm = ModelManager(
            conf = self.conf,
            input_shape = self.x_train.shape
        )
        self.model = mm.get_model()
        self.debug_model: Model = None
    
    def load_data(self):
        dm = DSManager(n_clients=self.conf.n_clients, conf=self.conf.db)
        
        train, test = dm.load_locally(partition_id=int(self.cid))
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
        return self.model.get_weights()

    def set_strategy(self, strategy: TrainingStrategy):
        self.strategy = strategy

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        return self.strategy.fit(self, parameters, config)

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        loss, size, conf = self.strategy.evaluate(self, parameters, config)
        my_logger.log(
            f'/c-data-{self.cid}.csv',
            data=self.get_log_data(config)
        )
        return loss, size, conf

    def get_log_data(self, config):
        return {
            'rounds': config['rounds'],
            'participating_state': self.participating_state,
            'desired_state': self.desired_state,
            'selected': self.selected,
            'cid': self.cid,
            'g_fit_acc': self.g_fit_acc, 
            'g_fit_loss': self.g_fit_loss, 
            'g_eval_acc': self.g_eval_acc, 
            'g_eval_loss': self.g_eval_loss, 
            'selected': self.selected,
            'training_method': self.conf.client.training_strategy,
            'aggregation': self.conf.server.aggregation.method,
            'selection': self.conf.server.selection.method,
            **self.data_to_log,
        }