from .driver import Driver
import tensorflow as tf
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np
import time
WILLING_PERC = 1.0

# https://arxiv.org/pdf/2205.14840
class MaxFLClientDriver(Driver):
    def __init__(self, rho):
        self.rho = rho
        pass

    def get_name(self):
        return "maxfl_driver"

    def q(self, loss)-> float:
        # Algorithm 1: Line 8
        x = loss - self.rho
        return 1/(1+np.exp(-x))
    
    def delta_weights(self, old_parameters, new_parameters):
        # Algorithm 1: Line 11
        return [old_layer - new_layer for old_layer, new_layer in zip(old_parameters, new_parameters)]

    def fit(self, client):
        losses, _ = client.model.evaluate(client.x_test, client.y_test)
        loss = np.mean(losses)
        q_value = self.q(loss)
        
        old_parameters = client.model.get_weights()
        start_time = time.time()
        history = client.model.fit(
            client.x_train,
            client.y_train,
            epochs = client.epochs,
            verbose = 0
        )
        end_time = time.time()
        cost = end_time - start_time
       
        new_parameters = client.model.get_weights()
        client.g_fit_acc = np.mean(history.history['accuracy'])
        client.g_fit_loss = np.mean(history.history['loss'])
        delta_weights_value = self.delta_weights(
            old_parameters = old_parameters,
            new_parameters = new_parameters,
        )
        
        client.q_value = q_value
        client.cost = cost
        return delta_weights_value

    def analyze(self, client, parameters, config):
        return super().analyze(client, parameters, config)