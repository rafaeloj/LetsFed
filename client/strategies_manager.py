import flwr as fl
import os
from strategies import FederatedClient
from strategies.training import LestFedClient
from strategies.states import NonParticipationState, ParticipationState 
import tensorflow as tf
from utils import get_env_var, ConfigManager

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def get_strategy(config):
    fc = FederatedClient(config)
    fc.set_strategy(LestFedClient(fc))
    return fc
    # return MaverickClient(config) 

def main():
    cm = ConfigManager('', 'utils/conf/variables.json', client=True)
    conf = cm.get_os_config()
    print(conf)
    fl.client.start_client(
        server_address=os.environ['SERVER_IP'],
        client=get_strategy(config=conf).to_client()
    )

if __name__ == "__main__":
    main()