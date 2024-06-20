import flwr as fl
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr.common.logger import logger
logger.disabled = True
from logging import INFO
from flwr_datasets.utils import divide_dataset
from datasets.utils.logging import disable_progress_bar
import os
import sys
sys.path.append(os.path.abspath('../../'))

from utils.select_by_server import is_select_by_server
from utils.logger import my_logger
from typing import List, Dict
from .drivers import Driver, CuriosityDriver, AccuracyDriver, IDLE, EXPLORING, EXPLORED
import numpy as np
import time
# disable_progress_bar()

TRESHOULD = 0.5

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
        swap:                 bool,
        rounds:               int,
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
        # Models
        self.model                                           = self.create_model(self.x_train.shape)
        self.debug_model                                     = self.create_model(self.x_train.shape)
        self.swap                                            = swap
        self.rounds                                          = rounds
        self.want                                            = self.dynamic_engagement

        self.behaviors: Dict[str, Driver]                         = self.set_behaviors()
        self.drivers_results                                 = {b_name: 0 for b_name, _ in self.behaviors.items()}
        self.willing: float                                  = 0.0
        self.rounds_intention                                = 0
    def set_behaviors(self):
        drivers: List[Driver] = [
            AccuracyDriver(input_shape = self.x_train.shape),
            CuriosityDriver(),
        ]
        return {
            driver.get_name(): driver
            for driver in drivers
        }

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

    def log_fit(self, data):
        my_logger.log(
            "/c-bw.csv",
            data = [
                data['server_round'], self.cid, data['size'], self.dynamic_engagement, data['server_selection']
            ],
            header = [
                'round', 'cid', 'size', 'dynamic_engagement', 'is_selected'
            ]
        )

        my_logger.fit("/c-fit.csv", data = [
            data['server_round'], self.cid, data['acc'], data['loss'], self.dynamic_engagement, data['server_selection']
        ])

        my_logger.log(
            '/c-cost-time.csv',
            data = [
                data['server_round'],
                self.cid,
                data['cost'],
                self.dynamic_engagement,
                data['server_selection']
            ],
            header = [
                'round',
                'cid',
                'cost',
                'dynamic_engagement',
                'is_selected']
        )

    def fit(self, parameters, config):
        fit_response = {
            'cid': self.cid,
            "dynamic_engagement": self.dynamic_engagement,
        }
        history = self.debug_model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])
        my_logger.log(
            "/c-local-model-fit.csv",
            data=[config['round'], self.cid, acc, loss],
            header=["round", 'cid', 'acc', 'loss'],
        )
        if is_select_by_server(str(self.cid), config['selected_by_server'].split(',')):
            return self._engaged_fit(parameters=parameters, config=config), self.x_train.shape[0], fit_response

        return self._not_engaged_fit(parameters=parameters, config=config, server_selection='not_selected'), self.x_train.shape[0], fit_response
    
    def _engaged_fit(self, parameters, config):
        if not self.dynamic_engagement:
            return self._not_engaged_fit(parameters=parameters, config=config, server_selection='selected') 
        
        model_size = sum([layer.nbytes for layer in parameters])
        new_parameters, acc, loss, cost = self._global_fit(weights=parameters, server_round=config['round']) # Atualiza o modelo global com dados do cliente

        self.log_fit({
            'server_round': config['round'],
            'size': model_size,
            'server_selection': 'selected',
            'acc': acc,
            'loss': loss,
            'cost': cost,
        })
        return new_parameters

    def _not_engaged_fit(self, parameters, config, server_selection):
        start_time = time.time()
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epoch, verbose=0)
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])
        end_time = time.time()
        cost = end_time - start_time

        model_size = sum([layer.nbytes for layer in self.model.get_weights()])

        self.log_fit({
            'server_round': config['round'],
            'size': model_size,
            'server_selection': server_selection,
            'acc': acc,
            'loss': loss,
            'cost': cost,
        })

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

    def log_evaluate(self, data):
        my_logger.evaluate("/c-eval.csv", [
            data['server_round'], self.cid, data['acc'], data['loss'], self.dynamic_engagement, data['server_selection']
        ])
        
        my_logger.drivers(
            "/c-drivers.csv",
            {
                "drivers": self.drivers_results,
                'server_round': data['server_round'],
                "cid": self.cid,
                "is_selected": data['server_selection'],
                'willing': self.willing
            }
        )
        if 'old_dynamic_engagement' in data:
            if data['old_dynamic_engagement'] != self.dynamic_engagement:
                my_logger.log(
                    filename = "/c-change-desired.csv",
                    data = [
                        data['server_round'], self.cid, data['old_dynamic_engagement'], self.want, data['acc'], data['r_intention'], self.miss
                    ],
                    header = ['round', 'cid', 'dynamic_engagement', 'desire', 'acc', 'r_intention', 'n_round_to_swap']
                )
                self.miss = 0

    def evaluate(self, parameters, config):
        loss, acc = self.debug_model.evaluate(self.x_test, self.y_test)
        my_logger.log(
            "/c-local-model-eval.csv",
            data=[config['round'], self.cid, acc, loss],
            header=['round', 'cid', 'acc', 'loss']
        )
        if is_select_by_server(str(self.cid), config['selected_by_server'].split(',')):
            self.miss+=1
            return self._engaged_evaluate(parameters=parameters, config=config)
        return self._not_engaged_evaluate(parameters=parameters, config=config, server_selection='not_selected')

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

        self.log_evaluate({
            'server_round': config['round'],
            'server_selection': 'selected',
            'acc': acc,
            'loss': loss,
            'r_intention'           : self.rounds_intention,
            'old_dynamic_engagement': old_dynamic_engagent,
        })

        return loss, self.x_test.shape[0], evaluation_response

    def _not_engaged_evaluate(self, parameters, config, server_selection):
        l_loss, l_acc = self.model.evaluate(self.x_test, self.y_test)
        old_dynamic_engagent = self.dynamic_engagement
        self.make_decision(parameters = parameters, config = config)

        self.log_evaluate({
            'server_round': config['round'],
            'server_selection': server_selection,
            'acc': l_acc,
            'loss': l_loss,
            'old_dynamic_engagement': old_dynamic_engagent,
            'r_intention'       : self.rounds_intention
        })

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
        self.drivers_results = {}
        if self.cid == 1 or self.cid == 2:
            self.dynamic_engagement = True
            self.want = True
            return

        if self.behaviors['curiosity_driver'].state == EXPLORING:
            if is_select_by_server(str(self.cid), config['selected_by_server'].split(',')):
                self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)
        elif self.behaviors['curiosity_driver'].state == IDLE:
            if is_select_by_server(str(self.cid), config['selected_by_server'].split(',')):
                value = self.behaviors["accuracy_driver"].analyze(self, parameters=parameters, config=config)
                self.drivers_results['accuracy_driver'] = value
                if value > TRESHOULD:
                    self.dynamic_engagement = True
                    self.want = True
                    self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)

        self.rounds_intention = self.behaviors['curiosity_driver'].current_round
        self.drivers_results['curiosity_driver'] = self.rounds_intention
        if self.behaviors['curiosity_driver'].state == EXPLORED:
            self.behaviors['curiosity_driver'].finish(self)

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
        if self.willing >= TRESHOULD:
            self.want = True
            self.dynamic_engagement = True
            return 

        if self.behaviors['curiosity_driver'].state == EXPLORED:
            self.behaviors['curiosity_driver'].finish(self)

