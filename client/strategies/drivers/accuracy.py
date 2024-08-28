from .driver import Driver
import tensorflow as tf
from typing import Tuple
from conf import Environment
from utils import ModelManager, is_select_by_server

WILLING_PERC = 1.0

class AccuracyDriver(Driver):
    def __init__(self, input_shape: Tuple[int], conf: Environment):
        self.threshold = conf.client.threshold
        self._create_model(input_shape = input_shape, conf = conf)

    def get_name(self):
        return "accuracy_driver"

    def _create_model(self, input_shape: Tuple[int], conf: Environment):
        self.mm = ModelManager(
            conf = conf,
            input_shape = input_shape
        )

    def run(self, client, parameters, config):       
        server_round = config['rounds']

        if server_round == 1:
            return True
        
        self.mm.model.set_weights(parameters)
        g_tmp_loss, _ = self.mm.model.evaluate(
            client.x_validation,
            client.y_validation,
            verbose = 0
        )
        c_tmp_loss, _ = client.model.evaluate(
            client.x_validation,
            client.y_validation,
            verbose = 0
        )
        client.diff = c_tmp_loss / g_tmp_loss

        willing = self._better(
            global_loss = g_tmp_loss,
            client_loss = c_tmp_loss
        )
        # print(f"{client.conf['cid']} with threshold {self.threshold}: {c_tmp_loss / g_tmp_loss} - {willing}")
        return willing

    def _better(self, global_loss: float, client_loss: float):
        return (client_loss / global_loss) > self.threshold