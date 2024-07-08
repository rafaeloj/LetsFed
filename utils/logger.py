import os
from typing import Dict

class Logger():
    def __init__(self, logger_foulder):
        self.logger_foulder = f"./logs{logger_foulder}"

    def _log(self, filename, data, header=None):
        file_path = f"{self.logger_foulder}{filename}"
        if header != None:
            with open(file_path, 'w') as file:
                file.write(f"{','.join(header)}\n")
                file.write(f"{','.join([f'{d}' for d in data])}\n")
            return

        with open(file_path, 'a') as file:
            file.write(f"{','.join([f'{d}' for d in data])}\n")
    def log(self, filename, data: {
        'rounds': int,
        'strategy': str,
        'model_type': str,
        'select_client_method': str,
        'n_selected': int,
        'n_engaged': int,
        'n_not_engaged': int,
        'selection': str,
        'r_intetion': str,
        'r_robin': str,
        'skip_round': bool,
        'local_epochs': int,
        'dirichlet_alpha': float,
        'no_iid': bool,
        'dataset': str,
        'exploitation': float,
        'exploration': float,
        'decay': float,
        'threshold': float,
        'init_clients': int,
    }): # type: ignore
        header = data.keys()
        data = data.values()
        if type(header) == list and len(header) != len(data.values()):
            print("Log aggregate problem")
            print(f"header ({len(header)}): {header}")
            print(f"Data: (len({data})){data}")
            exit(992)

        file_path = f"{self.logger_foulder}{filename}"
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._log(filename, data, header = header)
            return

        self._log(filename = filename, data = data)

    def fit(self, filename, data):
        """
            header: round, cid, acc, loss, dynamic_engagement, is_selected
        """
        header = ['rounds', 'cid', 'acc', 'loss', 'dynamic_engagement', 'is_selected']
        self.log(filename = filename, data = data, header = header)
    
    def evaluate(self, filename, data):
        """
            header: round, cid, acc, loss, dynamic_engagement, is_selected
        """
        header = ['rounds', 'cid', 'acc', 'loss', 'dynamic_engagement', 'is_selected']
        self.log(filename = filename, data = data, header = header)

    def aggregate_eval(self, filename, data):
        header = ['rounds', 'n_selected', 'n_engaged', 'n_not_engaged']
        self.log(filename = filename, data = data, header = header)

    def drivers(self, filename, data: Dict):
        headers = ["server_round", "cid", 'is_selected', 'willing']
        rows = [data["server_round"], data["cid"], data["is_selected"], data['willing']]
        for key, value in data["drivers"].items():
            headers.append(key)
            rows.append(value)
        
        self.log(filename = filename, data = rows, header = headers)


my_logger = Logger("")
