import os
from typing import Dict

class Logger():
    def __init__(self, logger_foulder):
        self.logger_foulder = logger_foulder

    def _log(self, filename, data, header=None):
        file_path = f"logs{self.logger_foulder}{filename}"
        # print(f"SAVE FILE: {file_path} with header: {header}")
        if header != None:
            with open(file_path, 'w') as file:
                file.write(f"{','.join(header)}\n")
                file.write(f"{','.join([f"{d}" for d in data])}\n")
            return

        with open(file_path, 'a') as file:
            file.write(f"{','.join([f"{d}" for d in data])}\n")

    def log(self, filename, data, header = None):
        if type(header) == list and len(header) != len(data):
            print("Log aggregate problem")
            exit(992)

        file_path = f"logs{self.logger_foulder}{filename}"
        if not os.path.exists(file_path):
            # print("FILE NOT EXIST")
            # print("Creating...")
            self._log(filename, data, header = header)
            return

        self._log(filename = filename, data = data)

    def fit(self, filename, data):
        """
            header: round, cid, acc, loss, dynamic_engagement, is_selected
        """
        header = ['round', 'cid', 'acc', 'loss', 'dynamic_engagement', 'is_selected']
        self.log(filename = filename, data = data, header = header)
    
    def evaluate(self, filename, data):
        """
            header: round, cid, acc, loss, dynamic_engagement, is_selected
        """
        header = ['round', 'cid', 'acc', 'loss', 'dynamic_engagement', 'is_selected']
        self.log(filename = filename, data = data, header = header)

    def aggregate_eval(self, filename, data):
        header = ['round', 'n_selected', 'n_engaged', 'n_not_engaged']
        self.log(filename = filename, data = data, header = header)

    def drivers(self, filename, data: Dict):
        headers = ["server_round", "cid", 'is_selected', 'willing']
        rows = [data["server_round"], data["cid"], data["is_selected"], data['willing']]
        for key, value in data["drivers"].items():
            headers.append(key)
            rows.append(value)
        
        self.log(filename = filename, data = rows, header = headers)

my_logger = Logger(os.environ['LOG_FOULDER'])
