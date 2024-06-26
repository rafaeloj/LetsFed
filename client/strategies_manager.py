import flwr as fl
import os
from strategies.cia_client import MaverickClient
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def get_strategy(
    cid,
    num_clients,
    dataset,
    epoch,
    no_idd,
    participate,
    dirichlet_alpha,
    swap,
    rounds,
):
    return MaverickClient(
        cid=cid,
        num_clients=num_clients,
        dataset=dataset,
        no_iid=no_idd,
        epoch=epoch,
        isParticipate=participate,
        dirichlet_alpha=dirichlet_alpha,
        swap        = swap,
        rounds = rounds
    )
def main():
    cid              = int(os.environ['CLIENT_ID'])
    num_clients      = int(os.environ['NUM_CLIENTS'])
    dataset          = os.environ['DATASET']
    local_epochs     = int(os.environ['LOCAL_EPOCHS'])
    no_idd           = os.environ["NO_IDD"] == "True" 
    participate      = os.environ["PARTICIPATE"] == "True"
    dirichlet_alpha  = float(os.environ["DIRICHLET_ALPHA"])
    swap             = os.environ['SWAP'] == 'True'
    rounds            = int(os.environ['ROUNDS'])
    fl.client.start_client(
        server_address=os.environ['SERVER_IP'],
        client=get_strategy(
            cid             = cid,
            num_clients     = num_clients,
            dataset         = dataset,
            epoch           = local_epochs,
            no_idd          = no_idd,
            participate     = participate,
            dirichlet_alpha = dirichlet_alpha,
            swap            = swap,
            rounds          = rounds,
        ).to_client()
    )

if __name__ == "__main__":
    main()