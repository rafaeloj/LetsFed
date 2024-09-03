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
        client.fusion_matrix = tf.Variable(tf.ones((2, len(client.labels))) / 2, dtype=tf.float32)

    def __get_drivers(self):
        drivers = [

            MaxFLQkDriver(),
        ]
        return drivers

    def fit(self, client: 'FederatedClient', parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        fit_response = {
            'cid': client.cid,
            "participating_state": True,
            # 'qk': client.qk
        }
        model_size = sum([layer.nbytes for layer in parameters])
        client.model_size = model_size


        glob_model: keras.Model = copy.deepcopy(client.model)
        glob_model.set_weights(parameters)

        local_model = self.update_fusion_weights(client, glob_model, config)

        loss, acc = local_model.evaluate(client.x_train, client.y_train)
        client.maxfl_loss = np.mean(loss)
        # https://openreview.net/pdf?id=8GI1SXqJBk
        client.apply_drivers(parameters=parameters, config=config) # add qk
        client.data_to_log['qk'] = client.qk
        fit_response['qk'] = client.qk

        client.g_fit_acc = np.mean(acc)
        client.g_fit_loss = np.mean(loss)

        client.selected = is_select_by_server(client.cid, config['selected_by_server'].split(','))
        if client.selected:
            prev_model = copy.deepcopy(local_model)
            history    = local_model.fit(client.x_train, client.y_train, epochs=client.conf.client.epochs, verbose=0)
            acc        = np.mean(history.history['accuracy'])
            loss       = np.mean(history.history['loss'])
            client.g_fit_acc = np.mean(acc)
            client.g_fit_loss = np.mean(loss)
            
            delta_parameters = [
                curr - prev
                for curr, prev in zip(local_model.get_weights(), prev_model.get_weights())
            ]
            client.model.set_weights(delta_parameters)

            return delta_parameters, client.x_train.shape[0], fit_response

        return client.model.get_weights(), client.x_train.shape[0], fit_response

    def update_fusion_weights(self, client: 'FederatedClient', glob_model: keras.Model, config: Config):
        if config['rounds'] == 0:
            return client.model
        # testar modelo global individualmente por label para pegar acurácia individual
        glob_accs = []
        for i in range(len(client.labels)):
            x_test = client.x_test[client.y_test == i]
            y_test = client.y_test[client.y_test == i]
            loss, acc = glob_model.evaluate(x_test, y_test)
            glob_accs.append(acc)
        glob_accs = np.array(glob_accs)
        glob_accs = glob_accs / glob_accs.sum()

        local_acc = []
        for i in range(len(client.labels)):
            x_test = client.x_test[client.y_test == i]
            y_test = client.y_test[client.y_test == i]
            loss, acc = client.model.evaluate(x_test, y_test)
            local_acc.append(acc)
    
        last_layer = [
            client.model.layers[-1].get_weights()[0] * local_acc + glob_model.layers[-1].get_weights()[0] * glob_accs,
            client.model.layers[-1].get_weights()[1] * local_acc + glob_model.layers[-1].get_weights()[1] * glob_accs
        ]

        the_model = copy.deepcopy(glob_model)
        the_model.layers[-1].set_weights(last_layer)

        return the_model
    
    def evaluate(self, client: 'FederatedClient', parameters: NDArrays, config: Config)-> Tuple[NDArrays, int, Dict[str, Scalar]]:        
        local_model = client.model
        client.selected = is_select_by_server(client.cid, config['selected_by_server'].split(','))
        if client.selected:
            glob_model: keras.Model = copy.deepcopy(client.model)
            glob_model.set_weights(parameters)
            local_model = self.update_fusion_weights(client, glob_model, config)
        loss, acc = local_model.evaluate(client.x_test, client.y_test)
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


        return loss, client.x_test.shape[0], eval_resp
