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
    no_iid,
    participate,
    dirichlet_alpha,
    swap,
    rounds,
    solution,
    method,
    exploitation: float,
    exploration: float,
    least_select_factor: float,
    decay: float,
    threshold: float,
    model_type,
    init_clients: float,
    config_test: str,
):
    return MaverickClient(
        cid=cid,
        num_clients=num_clients,
        dataset=dataset,
        no_iid=no_iid,
        epoch=epoch,
        isParticipate=participate,
        dirichlet_alpha=dirichlet_alpha,
        swap        = swap,
        rounds = rounds,
        solution = solution,
        method = method,
        exploitation = exploitation,
        exploration = exploration,
        least_select_factor = least_select_factor,
        decay = decay,
        threshold = threshold,
        model_type = model_type,
        init_clients = init_clients,
        config_test = config_test,
    )
def main():
    cid                 = int(os.environ['CLIENT_ID'])
    num_clients         = int(os.environ['NUM_CLIENTS'])
    dataset             = os.environ['DATASET']
    local_epochs        = int(os.environ['LOCAL_EPOCHS'])
    no_iid              = os.environ["NO_IID"] == "True" 
    participate         = os.environ["PARTICIPATE"] == "True"
    dirichlet_alpha     = float(os.environ["DIRICHLET_ALPHA"])
    swap                = os.environ['SWAP'] == 'True'
    rounds              = int(os.environ['ROUNDS'])
    solution            = os.environ["STRATEGY"]
    method              = os.environ["SELECT_CLIENT_METHOD"]
    decay               = float(os.environ["DECAY"])
    exploitation        = float(os.environ['EXPLOITATION'])
    exploration         = float(os.environ['EXPLORATION'])
    least_select_factor = float(os.environ['LEAST_SELECT_FACTOR'])
    threshold           = float(os.environ['THRESHOLD'])
    # model             = os.environ['MODEL']
    init_clients        = float(os.environ['INIT_CLIENTS'])
    local_epochs        = int(os.environ['LOCAL_EPOCHS'])
    dirichlet_alpha     = float(os.environ["DIRICHLET_ALPHA"])
    no_iid              = os.environ["NO_IID"] == "True" 
    threshold           = float(os.environ['THRESHOLD'])
    dataset             = os.environ['DATASET']
    model_type          = os.environ['MODEL_TYPE']
    config_test         = os.environ["CONFIG_TEST"]
    
    fl.client.start_client(
        server_address=os.environ['SERVER_IP'],
        client=get_strategy(
            cid                 = cid,
            num_clients         = num_clients,
            dataset             = dataset,
            epoch               = local_epochs,
            no_iid              = no_iid,
            participate         = participate,
            dirichlet_alpha     = dirichlet_alpha,
            swap                = swap,
            rounds              = rounds,
            solution            = solution,
            method              = method,
            decay               = decay,
            exploitation        = exploitation,
            exploration         = exploration,
            least_select_factor = least_select_factor,
            threshold           = threshold,
            model_type          = model_type,
            init_clients        = init_clients,
            config_test         = config_test,
        ).to_client()
    )

if __name__ == "__main__":
    main()