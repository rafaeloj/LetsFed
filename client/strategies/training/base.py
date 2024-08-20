from abc import ABC
from typing import Dict, Tuple
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)

import numpy as np

class TrainingStrategy(ABC):
    def get_parameters(self, client, config: Config) -> NDArrays:
        """ Default Method """
        return client.model.get_weights()
    
    def fit(self, client, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """ Default Method """
        client.model.set_weights(parameters)
        history = client.model.fit(client.x_train, client.y_train, epochs=client.conf['epochs'], verbose =0)
        client.conf['fit_acc']  = np.mean(history.history['accuracy'])
        client.conf['fit_loss'] = np.mean(history.history['loss'])

        return client.model.get_weights(), client.x_train.shape[0], {}

    def evaluate(self, client, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """ Default Method """
        client.model.set_weights(parameters)
        loss, acc       = self.model.evaluate(client.x_test, client.y_test)
        client.conf['fit_acc']  = acc
        client.conf['fit_loss'] = loss 
        eval_response = {
            "acc" : acc,
        }

        return loss, client.x_test.shape[0], eval_response
