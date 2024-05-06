import flwr as fl
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr.common.logger import logger
logger.disabled = True
from logging import INFO
from datasets.utils.logging import disable_progress_bar
from strategies.engagement_strategy.base import calculate_criteria
from strategies.engagement_strategy.accuracy_comparison_strategy import AccuracyComparisonStrategy
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import time
disable_progress_bar()
class MaverickClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid:                  int,
        num_clients:          int,
        dataset:              str,
        no_iid:               bool,
        epoch:                int,
        isParticipate:        bool,
        dirichlet_alpha:      float,
        log_foulder:          str,
        swap:                 bool,
    ):
        self.miss                                            = 0 # Criar contador de quando mudou de estado
        self.cid                                             = cid
        self.num_clients                                     = num_clients
        self.epoch                                           = epoch
        self.no_idd                                          = no_iid
        self.dataset                                         = dataset.lower()
        self.dirichlet_alpha                                 = dirichlet_alpha
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.dynamic_engagement                              = isParticipate
        self.log_foulder                                     = log_foulder
        # Models
        self.model                                           = self.create_model(self.x_train.shape)
        self.cia_evaluate_model                              = self.create_model(self.x_train.shape)

        self.swap                                            = swap

    def load_data(self):
        if self.no_idd:
            logger.log(INFO, "LOAD DATASET WITH DIRICHLET PARTITIONER")
            partitioner_train = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label",
                                    alpha=self.dirichlet_alpha, min_partition_size=self.num_clients/2,
                                    self_balancing=False)
        else:
            logger.log(INFO, "LOAD DATASET WITH IID PARTITIONER")
            partitioner_train =  IidPartitioner(num_partitions=self.num_clients)
        fds              = FederatedDataset(dataset=self.dataset, partitioners={"train": partitioner_train})
        train            = fds.load_partition(self.cid).with_format("numpy")
        

        partitioner_test = IidPartitioner(num_partitions=self.num_clients)
        fds_eval         = FederatedDataset(dataset=self.dataset, partitioners={"test": partitioner_test})
        test             = fds_eval.load_partition(self.cid).with_format("numpy")



        # partition       = fds.load_partition(self.cid)
        # division    = [0.8, 0.2]
        # train, test = divide_dataset(dataset=partition.with_format("numpy"), division=division)

        if self.dataset == 'mnist':
            return train['image'], train['label'], test['image'], test['label']
        elif self.dataset == 'cifar10':
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
        if self._is_seleceted_by_server(config['selected_by_server']):
            return self._engaged_fit(parameters=parameters, config=config), self.x_train.shape[0], fit_response
        return self._not_engaged_fit(parameters=parameters, config=config, server_selection='not_selected'), self.x_train.shape[0], fit_response
    
    def _engaged_fit(self, parameters, config):
        if not self.dynamic_engagement:
            return self._not_engaged_fit(parameters=parameters, config=config, server_selection='selected') 
        
        model_size = sum([layer.nbytes for layer in parameters])
        with open(f'logs{self.log_foulder}/c-bw.csv', 'a') as filename:
            # 'round', 'cid', 'acc', 'loss', 'participate'
            filename.write(f'{config['round']},{self.cid},{model_size},{self.dynamic_engagement},selected\n')

        new_parameters, acc, loss = self._cia_fit(weights=parameters, server_round=config['round']) # Atualiza o modelo global com dados do cliente
        with open(f'logs{self.log_foulder}/c-fit.csv','a') as filename:
            filename.write(f'{config['round']},{self.cid},{acc},{loss},{self.dynamic_engagement}\n')
        return parameters

    def _not_engaged_fit(self, parameters, config, server_selection):
        acc, loss = self._cia_local_fit() # Não atualiza o modelo global
        with open(f'logs{self.log_foulder}/c-fit.csv','a') as filename:
            # 'round', 'cid', 'acc', 'loss', 'model_size', 'participate', 'selected'
            filename.write(f'{config['round']},{self.cid},{acc},{loss},{self.dynamic_engagement}\n')

        model_size = sum([layer.nbytes for layer in self.model.get_weights()])
        with open(f'logs{self.log_foulder}/c-bw.csv', 'a') as filename:
            # 'round', 'cid', 'acc', 'loss', 'participate'
            filename.write(f'{config['round']},{self.cid},{model_size},{self.dynamic_engagement},{server_selection}\n')

        return parameters

    def _cia_local_fit(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epoch, verbose=0)
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])
        return acc, loss

    def _cia_fit(self, weights, server_round):
        start_time = time.time()
        self.model.set_weights(weights)
        history    = self.model.fit(self.x_train, self.y_train, epochs=self.epoch, verbose=0)
        acc        = np.mean(history.history['accuracy'])
        loss       = np.mean(history.history['loss'])
        parameters = self.model.get_weights()
        end_time = time.time()
        cost = end_time - start_time
        with open(f'logs{self.log_foulder}/c-cost-time.csv', 'a') as filename:
            filename.write(f'{server_round},{self.cid},{cost}')
        return parameters, acc, loss

    def evaluate(self, parameters, config):
        if self._is_seleceted_by_server(config['selected_by_server']):
            self.miss+=1
            return self._engaged_evaluate(parameters=parameters, config=config)
        return self._not_engaged_evaluate(parameters=parameters, config=config, server_selection='not_selected')

    def _engaged_evaluate(self, parameters, config):
        if not self.dynamic_engagement:
            return self._not_engaged_evaluate(parameters=parameters, config=config, server_selection='selected')

        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        evaluation_response = {
            "cid"               : self.cid,
            "dynamic_engagement": self.dynamic_engagement,
            'want'              : self.dynamic_engagement,
            'acc'               : acc,
        }

        with open(f'logs{self.log_foulder}/c-eval.csv', 'a') as filename:
            # 'round', 'cid', 'acc', 'loss', 'participate'
            filename.write(f'{config['round']},{self.cid},{acc},{loss},{self.dynamic_engagement},selected\n')

        return loss, self.x_test.shape[0], evaluation_response

    def _not_engaged_evaluate(self, parameters, config, server_selection):
        g_loss, g_acc = self._g_cia_evaluate(weights=parameters)   # comentar para casos não variavel
        l_loss, l_acc = self._l_cia_evaluate()

        with open(f'logs{self.log_foulder}/c-eval.csv', 'a') as filename:
            # 'round', 'cid', 'acc', 'loss', 'participate'
            filename.write(f'{config['round']},{self.cid},{l_acc},{l_loss},{self.dynamic_engagement},{server_selection}\n')

        want = self._change_engagement(g_acc = g_acc, l_acc = l_acc, config=config)
        evaluation_response = {
            "cid"               : self.cid,
            "dynamic_engagement": self.dynamic_engagement,
            'want'              : want,
            'acc'               : l_acc
        }
        
        self.dynamic_engagement = want # comentar para casos não variavel
        return l_loss, self.x_test.shape[0], evaluation_response

    def _is_seleceted_by_server(self, select_by_server: str):
        return str(self.cid) in select_by_server.split(',')

    def _g_cia_evaluate(self, weights):
        self.cia_evaluate_model.set_weights(weights)
        cia_loss, cia_accuracy = self.cia_evaluate_model.evaluate(self.x_test, self.y_test)
        return cia_loss, cia_accuracy
    
    def _l_cia_evaluate(self):        
        cia_loss, cia_accuracy = self.model.evaluate(self.x_test, self.y_test)
        return cia_loss, cia_accuracy

    def _change_engagement(self, g_acc, l_acc, config):
        if not self.swap:
            return self.dynamic_engagement
        want = self.dynamic_engagement
        if self._is_seleceted_by_server(config['selected_by_server']) and config['round'] > 1: # comentar para casos não variavel
            want = True if g_acc > l_acc else False # comentar para casos não variavel
            if want:
                with open(f'logs{self.log_foulder}/c-deev-desired.csv', 'a') as filename: # comentar para casos não variavel
                    # 'round', 'cid', 'participate', 'desired', 'g_acc', 'l_acc' # comentar para casos não variavel
                    filename.write(f'{config['round']},{self.cid},{self.dynamic_engagement},{want},{g_acc},{l_acc},{self.miss}\n') # comentar para casos não variavel
        return want

    # def _engagement_init(self, g_acc, l_acc):
    #     return np.array([
    #         AccuracyComparisonStrategy({"g_acc": g_acc, "l_acc": l_acc})
    #     ])

    # def _calculate_engagement(self, engagements):
    #     return calculate_criteria(engagements)
