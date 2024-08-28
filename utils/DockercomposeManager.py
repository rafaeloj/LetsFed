from omegaconf import OmegaConf
from typing import List
from random import sample
from conf import Environment
class DockercomposeManager():
    def __init__(self, config: Environment):
        self.conf = config
        self.start_clients()
    def generate(self, file_name: str):
        yaml_data = {
            'services': {
                f'client-{i}': v for i, v in enumerate(self.create_clients())                
            },
        }        
        yaml_data['services']['server'] = self.create_server()
        with open(file_name, 'w') as f:
            OmegaConf.save(yaml_data,f)

    def create_server(self):
        return {
            'image': self.server_img,
            'logging': { 'driver': 'local' },
            'container_name': 'rfl_server',
            'profiles': ['server'],
            'environment': [
                f"CLIENTS={','.join([str(cid) for cid in self.participating_clients])}"
            ],
            'command': [f'server/selection={self.conf.server.selection.method}', f'server/aggregation={self.conf.server.aggregation.method}'],
            'volumes': [
                './server/strategies/client_selection_method:/server/strategies/client_selection_method/:r',
                './server/strategies/aggregate_method:/server/strategies/aggregate_method/:r',
                './server/strategies_manager.py:/app/strategies_manager.py:r',
                './server/strategies/drivers:/server/strategies/drivers/:r',
                './server/strategies:/app/strategies/:r',
                './utils:/app/utils/:r',
                './model:/app/model:rw',
                './conf:/app/conf:r',
                './logs:/logs:rw',
            ],
            'networks': ['default'],
            'deploy': {
                'replicas': 1,
                'placement': {
                    'constraints': ['node.role==manager'],
                },
            },
        }
    def start_clients(self) -> list:
        n_clients_to_start = int(self.conf.n_clients*self.conf.init_clients)
        self.participating_clients = sample(range(self.conf.n_clients), n_clients_to_start)

    def create_clients(self) -> List[dict]: 
        clients = [
            {'image': self.client_img,
            'logging': {
                'driver': 'local'
            },
            'container_name': f'rfl_client-{i}',
            'profiles': ['client'],
            'command': [f'server/selection={self.conf.server.selection.method}', f'server/aggregation={self.conf.server.aggregation.method}'],
            'environment': [
                f'CID={i}',
                f'PARTICIPATING={True if i in self.participating_clients else False}',
            ],
            'volumes': [
                './client/strategies_manager.py:/client/strategies_manager.py:r',
                './client/strategies/training:/client/strategies/training/:r',
                './client/strategies/drivers:/client/strategies/drivers/:r',
                './client/strategies/states:/client/strategies/states/:r',
                './client/strategies:/client/strategies/:r',
                './utils:/utils/:r',
                './logs:/logs:rw',
                './conf:/client/conf:r',
                './model:/model:rw',
            ],
            'networks': ['default'],
            'deploy': { 
                'replicas': 1,
                'placement': {
                    'constraints': ['node.role==manager']
                }
            }} for i in range(self.conf.n_clients)
        ]

        return clients

    @property
    def server_img(self):
        return 'server-flwr-gpu' if self.conf.gpu else 'server-flwr-cpu'

    @property
    def client_img(self):
        return 'client-flwr-gpu' if self.conf.gpu else 'client-flwr-cpu'
    