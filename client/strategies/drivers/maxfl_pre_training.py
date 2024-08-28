from .driver import Driver
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from client.strategies import FederatedClient

class MaxFLPreTrainingDriver(Driver):

    def get_name(self):
        return "maxfl_pre_training_driver"

    def run(self, client: 'FederatedClient', parameters, config):
        client.model.fit(client.x_train, client.y_train, epochs=client.conf.server.aggregation.pre_training_epochs)

        loss, acc = client.model.evaluate(client.x_validation, client.y_validation)

        return loss