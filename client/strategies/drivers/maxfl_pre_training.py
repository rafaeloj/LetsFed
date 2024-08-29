from .driver import Driver
import numpy as np
from typing import TYPE_CHECKING
from utils import ModelManager
import copy
if TYPE_CHECKING:
    from client.strategies import FederatedClient

class MaxFLPreTrainingDriver(Driver):

    def get_name(self):
        return "maxfl_pre_training_driver"
    def run(self, client: 'FederatedClient', parameters, config):
        net_tmp = copy.deepcopy(client.model)    
        net_tmp.fit(client.x_train, client.y_train, epochs=client.conf.server.aggregation.pre_training_epochs)

        loss, acc = net_tmp.evaluate(client.x_validation, client.y_validation)

        return loss