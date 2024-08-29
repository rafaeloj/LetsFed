from .base import TrainingStrategy
from conf import Environment
import tensorflow as tf
from client.strategies.drivers import MaxFLQkDriver, MaxFLPreTrainingDriver
from utils import is_select_by_server
from typing import Dict, Tuple
import copy
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client.strategies.base import FederatedClient

import keras
import numpy as np

class MaxFLClient(TrainingStrategy):
    def __init__(self, client: 'FederatedClient'):
        pre_training = MaxFLPreTrainingDriver()
        client.maxfl_threshold = pre_training.run(client, None, None)
        client.data_to_log['maxfl_threshold'] = client.maxfl_threshold
        client.add_drivers(self.__get_drivers())

    def __get_drivers(self):
        drivers = [

            MaxFLQkDriver(),
        ]
        return drivers

    def fit(self, client: 'FederatedClient', parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        fit_response = {
            'cid': client.cid,
            "participating_state": True,
            # 'qk': 0
        }
        model_size = sum([layer.nbytes for layer in parameters])
        client.model_size = model_size
        
        client.selected = is_select_by_server(client.cid, config['selected_by_server'].split(','))

        # https://openreview.net/pdf?id=8GI1SXqJBk
        
        client.model.set_weights(parameters)
        v_loss, v_acc = client.model.evaluate(client.x_validation, client.y_validation)
        client.maxfl_loss = np.mean(v_loss)
        client.data_to_log['maxfl_loss'] = client.maxfl_loss
        client.v_fit_acc = np.mean(v_acc)
        client.v_fit_loss = np.mean(v_loss)
        client.data_to_log['v_fit_acc'] = client.v_fit_acc
        client.data_to_log['v_fit_loss'] = client.v_fit_loss

        prev_model: keras.Model = copy.deepcopy(client.model)
        history    = client.model.fit(client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0)
        acc        = np.mean(history.history['accuracy'])
        loss       = np.mean(history.history['loss'])
        
        client.g_fit_acc = np.mean(acc)
        client.g_fit_loss = np.mean(loss)
        
        client.apply_drivers(parameters=parameters, config=config) # add qk
        client.data_to_log['qk'] = client.qk
        fit_response['qk'] = client.qk
        
        delta_parameters = [
            curr - prev
            for curr, prev in zip(client.model.get_weights(), prev_model.get_weights())
        ]
        return delta_parameters, client.x_train.shape[0], fit_response


    def evaluate(self, client: 'FederatedClient', parameters: NDArrays, config: Config)-> Tuple[NDArrays, int, Dict[str, Scalar]]:
        client.selected = is_select_by_server(str(client.cid), config['selected_by_server'].split(','))
        if client.selected:
            client.model.set_weights(parameters)
            loss, acc = client.model.evaluate(client.x_test, client.y_test)
            loss = np.mean(loss)
            acc  = np.mean(acc)
            eval_resp = {
                "cid"                : client.cid,
                "participating_state": client.participating_state,
                'desired_state'      : client.participating_state,
                'acc'               : acc,
            }
            client.g_eval_loss = np.mean(loss)
            client.g_eval_acc = np.mean(acc)
        else:
            loss, acc = client.model.evaluate(client.x_test, client.y_test)
            loss = np.mean(loss)
            acc  = np.mean(acc)
            eval_resp = { 
                "cid"                : client.cid,
                "participating_state": True,
                'desired_state'      : client.participating_state,
                'acc'                : acc,
            }

        return loss, client.x_test.shape[0], eval_resp
