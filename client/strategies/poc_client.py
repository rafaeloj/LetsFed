import flwr as fl
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr.common.logger import logger
from logging import INFO
from datasets.utils.logging import disable_progress_bar
import numpy as np
disable_progress_bar()
class PocClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid:           int,
        num_clients:   int,
        dataset:       str,
        no_iid:        bool = True,
        epoch:         int  = 1,
        isParticipate: bool = False
    ):
        self.cid                                             = cid
        self.num_clients                                     = num_clients
        self.epoch                                           = epoch
        self.no_idd                                          = no_iid
        self.dirichlet_alpha                                 = 0.1
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
        

        iid_partitioner = IidPartitioner(num_partitions=self.num_clients)
        fds_eval        = FederatedDataset(dataset=self.dataset, partitioners={"test": iid_partitioner}, )
        test        = fds_eval.load_partition(self.cid).with_format("numpy")

        fds         = FederatedDataset(dataset=self.dataset, partitioners={"train": partitioner}, )
        train       = fds.load_partition(self.cid).with_format("numpy")

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
        fit_response = {
            'cid': self.cid,
            "dynamic_engagement": self.dynamic_engagement,
        }
        selected_clients   = []
        if config['selected_clients'] != '':
            selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(',')]

        poc_parameters, poc_acc, poc_loss = self._poc_fit(weights=parameters, clients=selected_clients) # Atualiza o modelo global com dados do cliente
        with open('logs/poc-train.csv', 'a') as train_file:
            train_file.write(f"{self.cid},{poc_acc},{poc_loss},{self.dynamic_engagement},{config['round']}\n")

        fit_response['acc'] = poc_acc
        
        return poc_parameters, self.x_train.shape[0], fit_response

    def _poc_fit(self, weights, clients):
        # if True:
        if self.cid in clients and self.dynamic_engagement:
            self.model.set_weights(weights)
            history    = self.model.fit(self.x_train, self.y_train, verbose=1, epochs=self.epoch)
            parameters = self.model.get_weights()
            loss       = np.mean(history.history['loss'])
            acc        = np.mean(history.history['accuracy'])
            return parameters, acc, loss
        history    = self.model.fit(self.x_train, self.y_train, verbose=1, epochs=self.epoch)
        loss       = np.mean(history.history['loss'])
        acc        = np.mean(history.history['accuracy'])
        return weights, acc, loss

    def evaluate(self, parameters, config):
        evaluate_response = {
            'cid': self.cid
        }
        loss, acc = self._poc_evaluate(parameters)
        with open('logs/poc-eval.csv', 'a') as eval_file:
            eval_file.write(f"{self.cid},{acc},{config['round']}\n")    
        evaluate_response['acc'] = acc
        return loss, self.x_test.shape[0], evaluate_response

    def _poc_evaluate(self, weights):
        # if True:
        if self.dynamic_engagement:
            self.model.set_weights(weights)
            loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            return loss, acc
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, acc
