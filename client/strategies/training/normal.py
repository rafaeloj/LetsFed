from .base import TrainingStrategy
from conf import Environment
from client.strategies.drivers import AccuracyDriver, CuriosityDriver
from utils import ModelManager, is_select_by_server
from typing import Dict, Tuple
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client.strategies.base import FederatedClient
from client.strategies.drivers import MaxFLQkDriver, MaxFLPreTrainingDriver

import time
import numpy as np

class NormalClient(TrainingStrategy):
    def __init__(self, client: 'FederatedClient'):
        self.load_debug_model(client)
        pre_training = MaxFLPreTrainingDriver()
        client.maxfl_threshold = pre_training.run(client, None, None)
        client.data_to_log['maxfl_threshold'] = client.maxfl_threshold
        client.add_drivers(self.__get_drivers())

    def __get_drivers(self):
        drivers = [
            MaxFLQkDriver(),
        ]
        return drivers

    def load_debug_model(self, client: 'FederatedClient'):
        mm = ModelManager(
            client.conf,
            input_shape = client.x_train.shape
        )
        client.debug_model = mm.get_model()

    def fit(self, client: 'FederatedClient', parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        fit_response = {
            'cid': client.cid,
            "participating_state": True,
            "desired_state"      : True
        }
        history = client.debug_model.fit(client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0)
        client.l_fit_acc     = np.mean(history.history['accuracy'])
        client.l_fit_loss    = np.mean(history.history['loss'])
        model_size = sum([layer.nbytes for layer in parameters])
        client.model_size = model_size
        client.selected = is_select_by_server(client.cid, config['selected_by_server'].split(','))
        if client.selected:
            return self.__participating_fit(client = client, parameters=parameters, config=config), client.x_train.shape[0], fit_response
        return client.model.get_weights(), client.x_train.shape[0], fit_response
    
    def __participating_fit(self, client: 'FederatedClient', parameters: NDArrays, config: Config) -> NDArrays:        
        client.model.set_weights(parameters)
        start_time = time.time()
        history    = client.model.fit(client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0)
        acc        = np.mean(history.history['accuracy'])
        loss       = np.mean(history.history['loss'])
        new_parameters = client.model.get_weights()
        end_time = time.time()
        cost = end_time - start_time
        client.maxfl_loss = np.mean(loss)
        client.apply_drivers(parameters=parameters, config=config) # add qk
        client.data_to_log['qk'] = client.qk
        client.g_fit_acc = np.mean(acc)
        client.g_fit_loss = np.mean(loss)
        client.cost = cost

        return new_parameters

    def evaluate(self, client: 'FederatedClient', parameters: NDArrays, config: Config)-> Tuple[NDArrays, int, Dict[str, Scalar]]:
        l_loss, l_acc = client.debug_model.evaluate(client.x_test, client.y_test)
        client.l_eval_acc = np.mean(l_acc)
        client.l_eval_loss = np.mean(l_loss)

        return self.__participating_evaluate(client = client, parameters=parameters, config=config)


    def __participating_evaluate(self, client: 'FederatedClient', parameters: NDArrays, config: Config) -> Tuple[float, int, Config]:
        client.model.set_weights(parameters)
        loss, acc = client.model.evaluate(client.x_test, client.y_test)
        loss = np.mean(loss)
        acc  = np.mean(acc)

        evaluation_response = {
            "cid"                : client.cid,
            'acc'                : acc,
            "participating_state": True,
            "desired_state"      : True
        }
        client.g_eval_loss = np.mean(loss)
        client.g_eval_acc = np.mean(acc)

        return loss, client.x_test.shape[0], evaluation_response