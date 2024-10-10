from .driver import Driver
import numpy as np

class MaxFLQkDriver(Driver):

    def get_name(self):
        return "maxfl_qk_driver"

    def sigmoid(self, x: float) -> float:
        temp_loss = 2*x
        return 2*np.exp(temp_loss) / (1.0 + np.exp(temp_loss))

    def run(self, client, parameters, config):
        loss_weight = self.sigmoid(np.sum(client.maxfl_loss) - client.maxfl_threshold) # Line 321: algos
        client.qk = loss_weight * (1 - loss_weight)
        # client.qk = loss_weight
        return client.qk
