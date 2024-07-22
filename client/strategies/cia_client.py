import flwr as fl
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr.common.logger import logger
logger.disabled = True
from logging import INFO
from flwr_datasets.utils import divide_dataset
from datasets import load_from_disk
from datasets.utils.logging import disable_progress_bar
from datasets import load_from_disk
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath('../../'))

from utils.select_by_server import is_select_by_server
from utils.logger import my_logger
from typing import List, Dict
from .drivers import Driver, CuriosityDriver, AccuracyDriver, IDLE, EXPLORING, EXPLORED
import numpy as np
import time
# disable_progress_bar()


class MaverickClient(fl.client.NumPyClient):

    def __init__(
        self,
        cid:                  int,
        num_clients:          int,
        dataset:              str,
        non_iid:               bool,
        epoch:                int,
        isParticipate:        bool,
        dirichlet_alpha:      float,
        swap:                 bool,
        rounds:               int,
        solution:             str,
        method:               str,
        init_clients:         float,
        exploitation: float,
        exploration: float,
        least_select_factor: float,
        decay: float,
        threshold: float,
        model_type: str,
        config_test: str,
    ):
        self.miss                                            = 0 # Criar contador de quando mudou de estado
        self.cid                                             = cid
        self.num_clients                                     = num_clients
        self.epoch                                           = epoch
        self.non_iid                                          = non_iid
        self.dataset                                         = dataset.lower()
        self.dirichlet_alpha                                 = dirichlet_alpha
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.dynamic_engagement                              = isParticipate
        # Models
        self.model_type                                      = model_type
        self.model_path = f"logs/{self.model_type}.weights.h5"  # Adiciona model_type ao nome do arquivo
        self.model                                           = self.create_model(self.x_train.shape)
        self.debug_model                                     = self.create_model(self.x_train.shape)
        self.swap                                            = swap
        self.rounds                                          = rounds
        self.want                                            = self.dynamic_engagement
        self.method                                          = method
        self.solution                                        = solution
        self.behaviors: Dict[str, Driver]                         = self.set_behaviors()
        self.drivers_results                                 = {b_name: 0 for b_name, _ in self.behaviors.items()}
        self.willing: float                                  = 0.0
        self.rounds_intention                                = 0
        self.exploitation                                    = exploitation
        self.exploration                                     = exploration
        self.least_select_factor                             = least_select_factor
        self.decay                                           = decay
        self.threshold                                       = threshold
        self.init_clients                                    = init_clients
        self.config_test                                     = config_test
        if self.dynamic_engagement:
            self.rounds_intention = self.rounds*0.1
            self.drivers_results['curiosity_driver'] = self.rounds_intention
    def set_behaviors(self):
        drivers: List[Driver] = [
            AccuracyDriver(input_shape = self.x_train.shape, model_type=self.model_type),
            CuriosityDriver(),
        ]
        return {
            driver.get_name(): driver
            for driver in drivers
        }

    def load_data(self):
        path = f'logs/{self.dataset.upper()}/non_iid-{self.non_iid}/clients-{self.num_clients}'
        folder = os.path.exists(f'{path}/{self.cid}-data-train')
        if not folder:
            logger.log(0, 'DOWNLOAD DATA FROM FLWR DATASET')
            if self.non_iid:
                logger.log(INFO, "LOAD DATASET WITH DIRICHLET PARTITIONER")
                partitioner_train = DirichletPartitioner(num_partitions=self.num_clients, partition_by="label",
                                        alpha=self.dirichlet_alpha,
                                        self_balancing=False)
            else:
                logger.log(INFO, "LOAD DATASET WITH IID PARTITIONER")
                partitioner_train =  IidPartitioner(num_partitions=self.num_clients)
            partitioner_test = IidPartitioner(num_partitions=self.num_clients)
            

            fds              = FederatedDataset(dataset=self.dataset, partitioners={"train": partitioner_train, "test": partitioner_test})

            train            = fds.load_partition(partition_id = self.cid, split = 'train').with_format("numpy")
            test             = fds.load_partition(partition_id = self.cid, split = 'test').with_format("numpy")
            train.save_to_disk(f'{path}/{self.cid}-data-train')
            test.save_to_disk(f'{path}/{self.cid}-data-test')
        else:
            print("LOAD DATA FROM LOCAL STORAGE")
            logger.log(INFO, 'LOAD DATA FROM LOCAL STORAGE')
            test = load_from_disk(f'{path}/{self.cid}-data-test')
            train = load_from_disk(f'{path}/{self.cid}-data-train')

        keys = list(test.features.keys())
        
        return train[keys[0]], train[keys[1]], test[keys[0]], test[keys[1]]

    def create_model(self, input_shape):
        if self.model_type == "cnn":
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape[1:]),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(10, activation='softmax'),
            ])
        else:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=input_shape[1:]),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64,  activation='relu'),
                tf.keras.layers.Dense(32,  activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax'),

            ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        if os.path.exists(self.model_path):
            print(f"Loading weights from {self.model_path}")
            model.load_weights(self.model_path)
        else:
            print(f"Saving weights to {self.model_path}")
            model.save_weights(self.model_path)
        return model

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        fit_response = {
            'cid': self.cid,
            "dynamic_engagement": self.dynamic_engagement,
        }
        history = self.debug_model.fit(self.x_train, self.y_train, epochs=self.epoch, verbose=0)
        self.l_fit_acc     = np.mean(history.history['accuracy'])
        self.l_fit_loss    = np.mean(history.history['loss'])

        if is_select_by_server(str(self.cid), config['selected_by_server'].split(',')):
            return self._engaged_fit(parameters=parameters, config=config), self.x_train.shape[0], fit_response

        return self._not_engaged_fit(parameters=parameters, config=config, server_selection='not_selected'), self.x_train.shape[0], fit_response
    
    def _engaged_fit(self, parameters, config):
        if not self.dynamic_engagement:
            return self._not_engaged_fit(parameters=parameters, config=config, server_selection='selected') 
        
        model_size = sum([layer.nbytes for layer in parameters])
        new_parameters, acc, loss, cost = self._global_fit(weights=parameters, server_round=config['rounds']) # Atualiza o modelo global com dados do cliente

        self.g_fit_acc = np.mean(acc)
        self.g_fit_loss = np.mean(loss)
        self.cost = cost
        self.model_size = model_size

        return new_parameters

    def _not_engaged_fit(self, parameters, config, server_selection):
        start_time = time.time()
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epoch, verbose=0)
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])
        end_time = time.time()
        cost = end_time - start_time

        model_size = sum([layer.nbytes for layer in self.model.get_weights()])

        self.g_fit_acc = np.mean(acc)
        self.g_fit_loss = np.mean(loss)
        self.cost = cost
        self.model_size = model_size
        return parameters

    def _global_fit(self, weights, server_round):
        start_time = time.time()
        self.model.set_weights(weights)
        history    = self.model.fit(self.x_train, self.y_train, epochs=self.epoch, verbose=0)
        acc        = np.mean(history.history['accuracy'])
        loss       = np.mean(history.history['loss'])
        parameters = self.model.get_weights()
        end_time = time.time()
        cost = end_time - start_time
        
        return parameters, acc, loss, cost

    def evaluate(self, parameters, config):
        loss = None
        shape = None
        eval_resp = None
        
        
        l_loss, l_acc = self.debug_model.evaluate(self.x_test, self.y_test)
        self.l_eval_acc = np.mean(l_acc)
        self.l_eval_loss = np.mean(l_loss)
        is_select = is_select_by_server(str(self.cid), config['selected_by_server'].split(','))
        if is_select:
            self.miss+=1
            loss, shape, eval_resp = self._engaged_evaluate(parameters=parameters, config=config)
        else:
            loss, shape, eval_resp = self._not_engaged_evaluate(parameters=parameters, config=config, server_selection='not_selected')

        my_logger.log(
            '/c-data.csv',
            data = {
                'rounds': config['rounds'], 
                'cid': self.cid, 
                'strategy': self.solution.lower(), 
                'select_client_method': self.method.lower(), 
                'model_type': self.model_type.lower(), 
                'g_eval_acc': self.g_eval_acc, 
                'g_eval_loss': self.g_eval_loss, 
                'l_eval_acc': self.l_eval_acc, 
                'l_eval_loss': self.l_eval_loss, 
                'g_fit_acc': self.g_fit_acc, 
                'g_fit_loss': self.g_fit_loss, 
                'l_fit_acc': self.l_fit_acc, 
                'l_fit_loss': self.l_fit_loss, 
                'dynamic_engagement': self.dynamic_engagement, 
                'old_dynamic_engagement': self.old_dynamic_engagement, 
                'is_selected': is_select, 
                'desire': self.want, 
                'size': self.model_size, 
                'cost': self.cost, 
                'willing': self.willing, 
                'r_intention': self.r_intention, 
                'miss': self.miss, 
                'local_epochs': self.epoch, 
                'dirichlet_alpha': self.dirichlet_alpha, 
                'non_iid': self.non_iid, 
                'dataset': self.dataset.lower(), 
                'exploitation': self.exploitation, 
                'exploration': self.exploration, 
                'least_select_factor': self.least_select_factor, 
                'decay': self.decay, 
                'threshold': self.threshold, 
                'init_clients': self.init_clients,
                'config_test': self.config_test,
            }
        )
        eval_resp['fit_acc'] = self.g_fit_acc

        return loss, shape, eval_resp

    def _engaged_evaluate(self, parameters, config):
        if not self.dynamic_engagement:
            return self._not_engaged_evaluate(parameters=parameters, config=config, server_selection='selected')

        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        
        old_dynamic_engagent = self.dynamic_engagement
        self.make_decision(parameters = parameters, config = config)
        
        evaluation_response = {
            "cid"               : self.cid,
            "dynamic_engagement": old_dynamic_engagent,
            'want'              : self.want,
            'acc'               : acc,
            'r_intention'       : self.rounds_intention
        }
        self.g_eval_loss = np.mean(loss)
        self.g_eval_acc = np.mean(acc)
        self.r_intention = self.rounds_intention
        self.old_dynamic_engagement = old_dynamic_engagent

        return loss, self.x_test.shape[0], evaluation_response

    def _not_engaged_evaluate(self, parameters, config, server_selection):
        l_loss, l_acc = self.model.evaluate(self.x_test, self.y_test)
        old_dynamic_engagent = self.dynamic_engagement
        self.make_decision(parameters = parameters, config = config)
        
        self.g_eval_acc = np.mean(l_acc)
        self.g_eval_loss = np.mean(l_loss)
        self.r_intention = self.rounds_intention
        self.old_dynamic_engagement = old_dynamic_engagent

        evaluation_response = {
            "cid"               : self.cid,
            "dynamic_engagement": old_dynamic_engagent,
            'want'              : self.want,
            'acc'               : l_acc,
            'r_intention'       : self.rounds_intention,
        }
        return l_loss, self.x_test.shape[0], evaluation_response

    def add_behavior(self, behavior: Driver):
        """
            In the current moment, the order of behaviros is importante, so take care what behavior you add
        """
        self.behaviors.append(behavior)

    def make_decision(self, parameters, config):
        if config['rounds'] == 1:
            self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)
            self.rounds_intention = self.behaviors['curiosity_driver'].current_round
            self.drivers_results['curiosity_driver'] = self.rounds_intention
            return
        # self.dynamic_engagement = True
        # self.want = True
        # return
        self.drivers_results = {}
        if is_select_by_server(str(self.cid), config['selected_by_server'].split(',')):
            value = self.behaviors["accuracy_driver"].analyze(self, parameters=parameters, config=config)
            self.drivers_results['accuracy_driver'] = value
            if value > self.threshold:
                self.dynamic_engagement = True
                self.want = True
                self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)
            else:
                self.dynamic_engagement = False
                self.want = False
                self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)

        return

    def bk_make_decision(self, parameters, config):
        if config['rounds'] == 1:
            self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)
            self.rounds_intention = self.behaviors['curiosity_driver'].current_round
            self.drivers_results['curiosity_driver'] = self.rounds_intention
            return
        self.drivers_results = {}
        if is_select_by_server(str(self.cid), config['selected_by_server'].split(',')):

            if self.behaviors['curiosity_driver'].state == EXPLORING:
                self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)
            elif self.behaviors['curiosity_driver'].state == IDLE:
                value = self.behaviors["accuracy_driver"].analyze(self, parameters=parameters, config=config)
                self.drivers_results['accuracy_driver'] = value
                # my_logger.log(
                #     "/c-curiosity-cp.csv",
                #     data={
                #         "round": config['rounds'],
                #         'cid': self.cid,
                #         'value': value,
                #         'threshold': self.threshold,
                #     },
                # )
                if value > self.threshold:
                    self.dynamic_engagement = True
                    self.want = True
                    self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)

            self.rounds_intention = self.behaviors['curiosity_driver'].current_round
            self.drivers_results['curiosity_driver'] = self.rounds_intention
        if self.behaviors['curiosity_driver'].state == EXPLORED:
            self.behaviors['curiosity_driver'].finish(self)
        # my_logger.log(
        #     "/d-curiosity.csv",
        #     data = {
        #         "round": config['rounds'],
        #         'cid': self.cid,
        #         'value': self.behaviors['curiosity_driver'].state,
        #         'threshold': self.behaviors['curiosity_driver'].current_round,
        #     },
        # )
        return

        values = []
        for b_name, behavior in self.behaviors.items():
            value = behavior.analyze(self, parameters, config)
            self.drivers_results[b_name] = value
            values.append(value)
        return
        if self.behaviors['curiosity_driver'].state == EXPLORING:
            return

        ## Se ficar aqui passa 1 round sem participar
        self.willing = sum(values)
        if self.willing >= self.threshold:
            self.want = True
            self.dynamic_engagement = True
            return 

        if self.behaviors['curiosity_driver'].state == EXPLORED:
            self.behaviors['curiosity_driver'].finish(self)

