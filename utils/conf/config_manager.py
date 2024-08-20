import os
import json

class ConfigManager():
    def __init__(self, config_file='', variables_file=os.getcwd()+'app/utils/conf/variables.json', client=False):
        if config_file:
            with open(config_file) as f:
                self.config = json.load(f)
        self.client = client
        self.load_variables(variables_file)

    def load_variables(self, variables_file):
        with open(variables_file) as f:
            self.variables = json.load(f)

    def select_conf(self, id):
        with open('./config-selected.json', 'w') as f:
            json.dump(self.config[id], f)
        self.config_selected = self.config[id]

    def get_os_config(self):
        if self.client:
            return self.get_os_client()
        if os.environ["STRATEGY_SID"].upper() == 'LETSFED':
            return self.get_letsfed_config()
        if os.environ["STRATEGY_SID"].upper() == 'POC':
            return self.get_poc_config()
        if os.environ["STRATEGY_SID"].upper() == 'DEEV':
            return self.get_deev_config()
        if os.environ["STRATEGY_SID"].upper() == 'AVG':
            return self.get_avg_config()
        if os.environ["STRATEGY_SID"].upper() == "R_ROBIN":
            return self.get_rr_config()
        if os.environ['STRATEGY_SID'].upper() == "MAXFL":
            return self.get_maxfl_config()
    
    # def save_os_config(self):
        
    def get_conf_selected(self):
        if not self.config_selected:
            with open('./config-selected.json') as f:
                self.config_selected = json.load(f)
        return self.config_selected
    
    def get_env_var(self, var: str, default=''):
        if var in os.environ:
            return os.environ[var]
        return default

    def get_os_client(self):
        str_exploration     = self.get_env_var(self.variables["exploration"])
        str_exploitation    = self.get_env_var(self.variables["exploitation"])
        str_decay           = self.get_env_var(self.variables["decay"])
        str_g_learning_rate = self.get_env_var(self.variables["g_learning_rate"])
        str_epsilon         = self.get_env_var(self.variables["epsilon"])
        str_rho             = self.get_env_var(self.variables["rho"])

        conf = {
            "participating"      : bool(self.get_env_var(self.variables["participating"])),
            "cid"              : int(self.get_env_var(self.variables["cid"])),
            "tid"              : self.get_env_var(self.variables["tid"]),
            "num_clients"      : int(self.get_env_var(self.variables["clients"])),
            "epochs"           : int(self.get_env_var(self.variables["epochs"])),
            "dataset"          : self.get_env_var(self.variables["dataset"]),
            "rounds"           : int(self.get_env_var(self.variables["rounds"])),
            "init_clients"     : float(self.get_env_var(self.variables["init_clients"])),
            "dirichlet_alpha"  : float(self.get_env_var(self.variables["dirichlet_alpha"])),
            "threshold"        : float(self.get_env_var(self.variables["threshold"])),
            "model_type"       : self.get_env_var(self.variables["model_type"], default='dnn'),
            "sid"              : self.get_env_var(self.variables["strategy"]),
            "p_method"         : self.get_env_var(self.variables["p_method"]),
            "np_method"        : self.get_env_var(self.variables["np_method"]),
            "exploration"      : float(str_exploration) if str_exploration else None,
            "exploitation"     : float(str_exploitation) if str_exploitation else None,
            "decay"            : float(str_decay) if str_decay else None,
            "g_learning_rate"  : float(str_g_learning_rate) if str_g_learning_rate else None,
            "epsilon"          : float(str_epsilon) if str_epsilon else None,
            "rho"              : float(str_rho) if str_rho else None,
        }
        return conf
    def get_letsfed_config(self):
        str_decay           = self.get_env_var(self.variables["decay"])
        conf = {
            'tid'             : self.get_env_var(self.variables["tid"]),
            'clients'         : int(self.get_env_var(self.variables["clients"])),
            'epochs'          : int(self.get_env_var(self.variables["epochs"])),
            'dataset'         : self.get_env_var(self.variables["dataset"]),
            'rounds'          : int(self.get_env_var(self.variables["rounds"])),
            'init_clients'    : float(self.get_env_var(self.variables["init_clients"])),
            'dirichlet_alpha' : float(self.get_env_var(self.variables["dirichlet_alpha"])),
            'threshold'       : float(self.get_env_var(self.variables["threshold"])),
            'strategy'        : self.get_env_var(self.variables["strategy"]),
            'exploration'     : float(self.get_env_var(self.variables["exploration"])),
            'p_method'        : self.get_env_var(self.variables["p_method"]),
            'exploitation'    : float(self.get_env_var(self.variables["exploitation"])),
            'np_method'       : self.get_env_var(self.variables["np_method"]),
            "decay"            : float(str_decay) if str_decay else None,
            'model_type'      : self.get_env_var(self.variables['model_type']),
            'participating'   : [int(x) for x in self.get_env_var(self.variables['participating']).split(",")],
        }
        return conf
    
    def get_poc_config(self):
        conf = {
            'tid'             : self.get_env_var(self.variables["tid"]),
            'clients'         : int(self.get_env_var(self.variables["clients"])),
            'epochs'          : int(self.get_env_var(self.variables["local_epochs"])),
            'dataset'         : self.get_env_var(self.variables["dataset"]),
            'rounds'          : int(self.get_env_var(self.variables["rounds"])),
            'init_clients'    : float(self.get_env_var(self.variables["init_clients"])),
            'dirichlet_alpha' : float(self.get_env_var(self.variables["dirichlet_alpha"])),
            'threshold'       : float(self.get_env_var(self.variables["threshold"])),
            'strategy'        : self.get_env_var(self.variables["strategy_sid"]),
            'exploration'     : float(self.get_env_var(self.variables["strategy_exploration"])),
            'model_type'      : self.get_env_var(self.variables['model_type']),
        }
        return conf

    def get_avg_config(self):
        conf = {
            'tid'             : self.get_env_var(self.variables["tid"]),
            'clients'         : int(self.get_env_var(self.variables["clients"])),
            'epochs'          : int(self.get_env_var(self.variables["local_epochs"])),
            'dataset'         : self.get_env_var(self.variables["dataset"]),
            'rounds'          : int(self.get_env_var(self.variables["rounds"])),
            'init_clients'    : float(self.get_env_var(self.variables["init_clients"])),
            'dirichlet_alpha' : float(self.get_env_var(self.variables["dirichlet_alpha"])),
            'threshold'       : float(self.get_env_var(self.variables["threshold"])),
            'strategy'        : self.get_env_var(self.variables["strategy_sid"]),
            'exploration'     : float(self.get_env_var(self.variables["strategy_exploration"])),
            'model_type'      : self.get_env_var(self.variables['model_type']),
        }
        return conf

    def get_deev_config(self):
        conf = {
            'tid'             : self.get_env_var(self.variables["tid"]),
            'clients'         : int(self.get_env_var(self.variables["clients"])),
            'epochs'          : int(self.get_env_var(self.variables["epochs"])),
            'dataset'         : self.get_env_var(self.variables["dataset"]),
            'rounds'          : int(self.get_env_var(self.variables["rounds"])),
            'init_clients'    : float(self.get_env_var(self.variables["init_clients"])),
            'dirichlet_alpha' : float(self.get_env_var(self.variables["dirichlet_alpha"])),
            'threshold'       : float(self.get_env_var(self.variables["threshold"])),
            'strategy'        : self.get_env_var(self.variables["strategy"]),
            'decay'           : float(self.get_env_var(self.variables["decay"])),
            'model_type'      : self.get_env_var(self.variables['model_type']),
        }
        return conf

    def get_rr_config(self):
        conf = {
            'tid'             : self.get_env_var(self.variables["tid"]),
            'clients'         : int(self.get_env_var(self.variables["clients"])),
            'epochs'          : int(self.get_env_var(self.variables["local_epochs"])),
            'dataset'         : self.get_env_var(self.variables["dataset"]),
            'rounds'          : int(self.get_env_var(self.variables["rounds"])),
            'init_clients'    : float(self.get_env_var(self.variables["init_clients"])),
            'dirichlet_alpha' : float(self.get_env_var(self.variables["dirichlet_alpha"])),
            'threshold'       : float(self.get_env_var(self.variables["threshold"])),
            'strategy'        : self.get_env_var(self.variables["strategy"]),
            'exploration'     : float(self.get_env_var(self.variables["exploration"])),
            'model_type'      : self.get_env_var(self.variables['model_type']),
        }
        return conf
    
    def get_maxfl_config(self):
        conf = {
            'tid'             : self.get_env_var(self.variables["tid"]),
            'clients'         : int(self.get_env_var(self.variables["clients"])),
            'epochs'          : int(self.get_env_var(self.variables["epochs"])),
            'dataset'         : self.get_env_var(self.variables["dataset"]),
            'rounds'          : int(self.get_env_var(self.variables["rounds"])),
            'init_clients'    : float(self.get_env_var(self.variables["init_clients"])),
            'dirichlet_alpha' : float(self.get_env_var(self.variables["dirichlet_alpha"])),
            'threshold'       : float(self.get_env_var(self.variables["threshold"])),
            'strategy'        : self.get_env_var(self.variables["strategy"]),
            'exploration'     : float(self.get_env_var(self.variables["exploration"])),
            'model_type'      : self.get_env_var(self.variables['model_type']),
            'g_learning_rate' : float(self.get_env_var(self.variables["g_learning_rate"])),
            'epsilon'         : float(self.get_env_var(self.variables["epsilon"])),
            'rho'             : float(self.get_env_var(self.variables["rho"])),
        }
        return conf