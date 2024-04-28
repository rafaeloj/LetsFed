import flwr as fl
import tensorflow as tf
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr_datasets.federated_dataset import FederatedDataset
from flwr.common import parameters_to_ndarrays
from flwr.common.logger import logger
from flwr_datasets.utils import divide_dataset
from logging import INFO
import numpy as np
class FedAvgClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid:             int,
        num_clients:     int,
        dataset:         str,
        no_iid:          bool  = True,
        epoch:           int   = 1,
        isParticipate:   bool  = False,
        dirichlet_alpha: float = 0.1,
    ):
        self.cid                                             = cid
        self.num_clients                                     = num_clients
        self.epoch                                           = epoch
        self.no_idd                                          = no_iid
        self.dirichlet_alpha                                 = dirichlet_alpha
        self.dataset                                         = dataset.lower()
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.dynamic_engagement                              = isParticipate
        # Models
        self.model                                           = self.create_model(self.x_train.shape)
    
    def load_data(self):
        if self.no_idd:
            logger.log(INFO, "LOAD DATASET WITH DIRICHLET PARTITIONER")
            partitioner = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label",
                                    alpha=self.dirichlet_alpha, min_partition_size=self.num_clients,
                                    self_balancing=False)
        else:
            logger.log(INFO, "LOAD DATASET WITH IID PARTITIONER")
            partitioner =  IidPartitioner(num_partitions=self.num_clients)
        fds         = FederatedDataset(dataset=self.dataset, partitioners={"train": partitioner}, )
        

        iid_partitioner = IidPartitioner(num_partitions=self.num_clients)
        fds_eval        = FederatedDataset(dataset=self.dataset, partitioners={"test": iid_partitioner}, )
        test            = fds_eval.load_partition(self.cid).with_format("numpy")
        train           = fds.load_partition(self.cid).with_format("numpy")



        # partition       = fds.load_partition(self.cid)
        # division    = [0.8, 0.2]
        # train, test = divide_dataset(dataset=partition.with_format("numpy"), division=division)

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
        return self.model.get_weights()
    
    def fit(self, parameters, config):
        
        new_parameters, acc, loss = self._avg_fit(parameters = parameters)
        model_size = sum([layer.nbytes for layer in new_parameters])
        with open('logs/c-fedavg-fit.csv','a') as filename:
            filename.write(f'{config['round']},{self.cid},{acc},{loss},{model_size},{self.dynamic_engagement}\n')

        return new_parameters, self.x_train.shape[0], {}

    def _avg_fit(self, parameters):
        if self.dynamic_engagement:
            self.model.set_weights(parameters)
            history = self.model.fit(self.x_train, self.y_train, epochs=self.epoch)
            new_parameters = self.model.get_weights()
            acc = np.mean(history.history['accuracy'])
            loss = np.mean(history.history['loss'])
            return new_parameters, acc, loss
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epoch)
        acc = np.mean(history.history['accuracy'])
        loss = np.mean(history.history['loss'])
        return parameters, acc, loss

    def evaluate(self, parameters, config):
        loss, acc = self._avg_eval(parameters = parameters)
        
        model_size = sum([layer.nbytes for layer in parameters])

        
        with open('logs/c-fedavg-eval.csv', 'a') as filename:
            filename.write(f'{config['round']},{self.cid},{acc},{loss},{model_size},{self.dynamic_engagement}\n')
        return loss, self.x_test.shape[0], {'accuracy': acc}
    
    def _avg_eval(self, parameters):
        if self.dynamic_engagement:
            self.model.set_weights(parameters)
            loss, acc = self.model.evaluate(self.x_test, self.y_test)
            return loss, acc
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        return loss, acc
        

