import flwr as fl
import os
import random
from strategies.client import MaverickClient

def get_strategy(strategy, cid, num_clients, dataset, epoch, no_idd, participate):
    match strategy:
        case 'CIA':
            return MaverickClient(cid=cid, num_clients=num_clients, dataset=dataset, no_iid=no_idd, epoch=epoch, isParticipate=participate)

def main():
    strategy     = os.environ['STRATEGY']
    cid          = int(os.environ['CLIENT_ID'])
    num_clients  = int(os.environ['NUM_CLIENTS'])
    dataset      = os.environ['DATASET']
    local_epochs = int(os.environ['LOCAL_EPOCHS'])
    no_idd       = os.environ["NO_IDD"] == "True" 
    participate  = os.environ["PARTICIPATE"] == "True"
    fl.client.start_client(
        server_address=os.environ['SERVER_IP'],
        client=get_strategy(
            strategy    = strategy,
            cid         = cid,
            num_clients = num_clients,
            dataset     = dataset,
            epoch       = local_epochs,
            no_idd      = no_idd,
            participate = participate
        ).to_client()
    )

if __name__ == "__main__":
    main()