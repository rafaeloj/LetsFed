class DockercomposeManager():
    def __init__(self, config_values, config_variables, gpu=False):
        self.server_config = config_values
        self.client_config = config_values
        self.config_variables = config_variables
        self.file_name = None
        self.gpu = gpu


    def filename(self, config):
        if self.file_name:
            return self.file_name
        self.file_name = f"dockercompose-{self.tdi(config, '')}.yaml".lower()

        return self.file_name

    def create_dockercompose(self, participating_clients):
        with open(f"{self.filename(self.server_config)}", 'w') as dockercompose_file:
            header = f"services:\n\n"
            dockercompose_file.write(header)
            server_str = self.s_config([str(cid) for cid in participating_clients])

            dockercompose_file.write(server_str)
            for client in range(self.client_config['clients']):
                client_str = self.c_config(
                    cid = client,
                    participating = client in participating_clients
                )
                dockercompose_file.write(client_str)

    def c_config(self, cid, participating):
        client_str = f"    client-{cid}:\n\
        image: {self.client_image}\n\
        logging:\n\
            driver: local\n\
        {self.runtime}\n\
        container_name: rfl_client-{cid}\n\
        profiles:\n\
        - client\n\
        environment:\n\
            - TID={self.tdi(self.client_config, '')}\n\
            - SERVER_IP=rfl_server:9999\n\
            - PARTICIPATING={participating}\n\
            - CID={cid}\n\
            {self.environment(config = self.client_config)}\n\
        {self.c_volumes}\n\
            {self.default}\n\
        \n"

        return client_str
    

    def tdi(self, config, v: str):
        for _, value in config.items():
            if type(value) == dict:
                v += self.tdi(value, '')
            else:
                v += f"{value}" if v == '' else f'-{value}'
        return v

    @property
    def client_image(self):
        if not self.gpu:
            return 'client-flwr-cpu:latest'
        return 'client-flwr:latest'

    def s_config(self, participating):
        server_str = f"    server:\n\
        image: {self.server_image}\n\
        logging:\n\
            driver: local\n\
        {self.runtime}\n\
        container_name: rfl_server\n\
        profiles:\n\
        - server\n\
        environment:\n\
            - SERVER_IP=0.0.0.0:9999\n\
            - TID={self.tdi(self.server_config, '')}\n\
            - PARTICIPATING={','.join(participating)}\n\
            {self.environment(config=self.server_config)}\n\
        {self.s_volumes}\n\
        {self.default}\
        \n"

        return server_str

    @property
    def server_image(self):
        if not self.gpu:
            return 'server-flwr-cpu:latest'
        return 'server-flwr:latest'

    @property
    def runtime(self):
        if self.gpu:
            return 'runtime: nvidia'
        return ''

    @property
    def default(self):
        return f"""
        networks:\n\
                - default\n\
        deploy:\n\
            replicas: 1\n\
            {self.gpu_default}\n\
            placement:\n\
                constraints:\n\
                - node.role==manager\n\
        """
    @property
    def gpu_default(self):
        if self.gpu:
            return """
                resources: \n\
                    reservations:\n\
                    devices:\n\
                        - capabilities: [gpu]\n\
            """
        return ""
    @property
    def s_volumes(self):
        return """volumes:\n\
            - ./logs:/logs:rw\n\
            - ./server/strategies_manager.py:/app/strategies_manager.py:r\n\
            - ./server/strategies/drivers:/server/strategies/drivers/:r\n\
            - ./server/strategies:/app/strategies/:r\n\
            - ./utils:/app/utils/:r
        """

    @property
    def c_volumes(self):
        return """volumes:\n\
            - ./logs:/logs:rw\n\
            - ./client/strategies:/client/strategies/:r\n\
            - ./client/strategies/drivers:/client/strategies/drivers/:r\n\
            - ./client/strategies_manager.py:/client/strategies_manager.py:r\n\
            - ./utils:/utils/:r\n\
        """

    def environment(self, config: dict, s="", pref=""):
        for key, value in config.items():
            s += f"- {self.config_variables[key]}={value}\n\
            "
        return s
