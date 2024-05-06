import flwr as fl
import os
from strategies.cia_client import MaverickClient
from strategies.poc_client import PocClient
from strategies.deev_client import DeevClient
from strategies.fed_client import FedAvgClient

def get_strategy(
    strategy,
    cid,
    num_clients,
    dataset,
    epoch,
    no_idd,
    participate,
    dirichlet_alpha,
    log_foulder,
    swap,
):
    return MaverickClient(
        cid=cid,
        num_clients=num_clients,
        dataset=dataset,
        no_iid=no_idd,
        epoch=epoch,
        isParticipate=participate,
        dirichlet_alpha=dirichlet_alpha,
        log_foulder = log_foulder,
        swap        = swap,
    )
def main():
    strategy         = os.environ['STRATEGY']
    cid              = int(os.environ['CLIENT_ID'])
    num_clients      = int(os.environ['NUM_CLIENTS'])
    dataset          = os.environ['DATASET']
    local_epochs     = int(os.environ['LOCAL_EPOCHS'])
    no_idd           = os.environ["NO_IDD"] == "True" 
    participate      = os.environ["PARTICIPATE"] == "True"
    dirichlet_alpha  = float(os.environ["DIRICHLET_ALPHA"])
    foulder          = os.environ['LOG_FOULDER']
    swap             = os.environ['SWAP'] == 'True'

    fl.client.start_client(
        server_address=os.environ['SERVER_IP'],
        client=get_strategy(
            strategy        = strategy,
            cid             = cid,
            num_clients     = num_clients,
            dataset         = dataset,
            epoch           = local_epochs,
            no_idd          = no_idd,
            participate     = participate,
            dirichlet_alpha = dirichlet_alpha,
            log_foulder     = foulder,
            swap            = swap,
        ).to_client()
    )

if __name__ == "__main__":
    main()