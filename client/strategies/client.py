import flwr as fl
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr_datasets.utils import divide_dataset
from flwr.common.logger import logger
from logging import INFO
from datasets.utils.logging import disable_progress_bar
import pickle
import base64
disable_progress_bar()
class MaverickClient(fl.client.NumPyClient):

    def __init__(self, cid: int, num_clients: int, dataset: str, no_iid: bool = True, epoch: int = 1, isParticipate: bool = False):
        self.cid                                             = cid
        self.num_clients                                     = num_clients
        self.epoch                                           = epoch
        self.no_idd                                          = no_iid
        self.dirichlet_alpha                                 = 0.1
        self.dataset                                         = dataset.lower()
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.dynamic_engagement                              = isParticipate

        # Models
        self.local_model                                     = self.create_model(self.x_train.shape)
        self.federated_model                                 = self.create_model(self.x_train.shape)
        self.cia_model                                       = self.create_model(self.x_train.shape)

    def load_data(self):
        if self.no_idd:
            logger.log(INFO, "LOAD DATASET WITH DIRICHLET PARTITIONER")
            partitioner = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label",
                                    alpha=self.dirichlet_alpha, min_partition_size=self.num_clients,
                                    self_balancing=False)
        else:
            logger.log(INFO, "LOAD DATASET WITH IID PARTITIONER")
            partitioner =  IidPartitioner(num_partitions=self.num_clients)
        

        # iid_partitioner = IidPartitioner(num_partitions=self.num_clients)
        # fds_eval        = FederatedDataset(dataset=self.dataset, partitioners={"test": iid_partitioner}, )
        # test        = fds_eval.load_partition(self.cid).with_format("numpy")

        fds         = FederatedDataset(dataset=self.dataset, partitioners={"train": partitioner}, )
        # train       = fds.load_partition(self.cid).with_format("numpy")


        partition       = fds.load_partition(self.cid)
        division    = [0.8, 0.2]
        train, test = divide_dataset(dataset=partition.with_format("numpy"), division=division)

        match self.dataset:
            case 'mnist':
                return train['image'], train['label'], test['image'], test['label']
            case 'cifar10':
                return train['img'], train['label'], test['img'], test['label']
        
    def create_model(self, input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape[1:]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64,  activation='relu'),
            tf.keras.layers.Dense(32,  activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),

        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
        return model

    def get_parameters(self, config):
        return self.federated_model.get_weights()

    def fit(self, parameters, config):
        match config['round']:
            case 1:
                cia_config = parameters
            case _:
                cia_config = self._decode_message(config["cia_parameters"])
        
        fed_parameters, fed_acc = self._fed_fit(weights=parameters)
        cia_parameters, cia_acc = self._cia_fit(weights=cia_config)
        local_acc = self._local_fit()

        with open('logs/train.csv', 'a') as train_file:
            train_file.write(f"{self.cid},{fed_acc[0]},{cia_acc[0]},{local_acc[0]},{self.dynamic_engagement},{config['round']}\n")

        fit_response = {
            'cid': self.cid,
            "dynamic_engagement": self.dynamic_engagement,
            'cia_parameters': self._encode_message(cia_parameters)
        }

        return fed_parameters, self.x_train.shape[0], fit_response

    def _fed_fit(self, weights):
        self.federated_model.set_weights(weights)
        history_fed   = self.federated_model.fit(self.x_train, self.y_train, epochs=self.epoch)
        parameters    = self.federated_model.get_weights()
        fed_acc   = history_fed.history['accuracy']

        return parameters, fed_acc

    def _cia_fit(self, weights):
        self.cia_model.set_weights(weights)
        history = self.cia_model.fit(self.x_train, self.y_train, epochs=self.epoch)
        acc = history.history['accuracy']
        parameters = self.cia_model.get_weights()

        return parameters, acc

    def _local_fit(self,) -> float:
        history = self.local_model.fit(self.x_train, self.y_train, epochs=self.epoch)
        acc = history.history['accuracy']
        return acc

    def evaluate(self, parameters, config):
        fed_loss, fed_accuracy     = self._fed_evaluate(weights = parameters)
        cia_loss, cia_accuracy     = self._cia_evaluate(weights = self._decode_message(config['cia_parameters']))
        _, local_accuracy = self._local_evaluate()

        with open('logs/eval.csv', 'a') as eval_file:
            eval_file.write(f"{self.cid},{fed_accuracy},{cia_accuracy},{local_accuracy},{self.dynamic_engagement},{config['round']}\n")

        evaluation_response = {
			"cid"               : self.cid,
			"dynamic_engagement": self.dynamic_engagement,
            'cia_loss'          : cia_loss
		}
        return fed_loss, self.x_test.shape[0], evaluation_response
    
    def _fed_evaluate(self, weights):
        self.federated_model.set_weights(weights)
        fed_loss, fed_accuracy     = self.federated_model.evaluate(self.x_test, self.y_test)
        return fed_loss, fed_accuracy

    def _cia_evaluate(self, weights):
        self.cia_model.set_weights(weights)
        loss, accuracy     = self.cia_model.evaluate(self.x_test, self.y_test)
        return loss, accuracy
    
    def _local_evaluate(self):
        loss, accuracy = self.local_model.evaluate(self.x_test, self.y_test)
        return loss, accuracy
    
    def _decode_message(self, message):
        decoded_message = base64.b64decode(message)
        return pickle.loads(decoded_message)
    
    def _encode_message(self, message):
        cia_parameters_serialized = pickle.dumps(message)
        cia_parameters_bytes = base64.b64encode(cia_parameters_serialized)
        return cia_parameters_bytes
