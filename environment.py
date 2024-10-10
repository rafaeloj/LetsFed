from conf.config import Database
from utils import DSManager, DockercomposeManager
from conf import Environment
from omegaconf import OmegaConf


def load_data(n_clients: int, db: Database):
    dm = DSManager(n_clients=n_clients, conf=db)
    dm.load(db.dataset)
    dm.save_locally()

def main():
    cfg: Environment = OmegaConf.load("conf/config.yaml")
    print(cfg)
    a = DockercomposeManager(cfg)
    a.generate(file_name=f"dockercompose-{cfg.n_clients}-{cfg.model_type}-{cfg.rounds}-{cfg.init_clients}-{cfg.client.epochs}-{cfg.init_clients}.yaml")

    load_data(
        n_clients=cfg.n_clients,
        db=cfg.db
    )

if __name__ == "__main__":
    main()