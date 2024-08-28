from .driver import Driver
import numpy as np

class MaxFLQkDriver(Driver):

    def get_name(self):
        return "maxfl_qk_driver"

    def sigmoid(self, x: float) -> float:
        return np.exp(x) / (1.0 + np.exp(x))

    def run(self, client, parameters, config):
        loss_weight = self.sigmoid(2*(np.sum(client.maxfl_loss) - client.maxfl_threshold)) # Line 321: algos
        client.qk = loss_weight * (1-loss_weight)
        return client.qk
