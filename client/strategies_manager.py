import flwr as fl
import os
from typing import TYPE_CHECKING
from conf import Environment
from strategies import FederatedClient
import tensorflow as tf
from omegaconf import OmegaConf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def get_strategy(config: Environment):
    fc = FederatedClient(
        cid = int(os.environ["CID"]),
        config = config,
    )
    return fc

def main():
    cfg: Environment = OmegaConf.load('/client/conf/config.yaml')
    fl.client.start_client(
        server_address=f'rfl_server:{cfg.server.port}',
        client=get_strategy(cfg).to_client()
    )

if __name__ == "__main__":
    main()