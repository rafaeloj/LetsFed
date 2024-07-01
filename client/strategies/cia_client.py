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
        solution:             str,
        method:               str,
        exploitation: float,
        exploration: float,
        least_select_factor: float,
        decay: float,
        threshold: float,
    ):
        self.miss                                            = 0 # Criar contador de quando mudou de estado
        self.cid                                             = cid
        self.num_clients                                     = num_clients
        self.epoch                                           = epoch
        self.no_iid                                          = no_iid
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
        if self.dynamic_engagement:
            self.rounds_intention = self.rounds*0.1
            self.drivers_results['curiosity_driver'] = self.rounds_intention
    def set_behaviors(self):
        drivers: List[Driver] = [
            AccuracyDriver(input_shape = self.x_train.shape, model=''),
            CuriosityDriver(),
        ]
        return {
            driver.get_name(): driver
            for driver in drivers
        }

    def load_data(self):
        if self.no_iid:
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
        if self.dataset.lower() == "cifar10":            
            deep_cnn = tf.keras.layers.Sequential()
            deep_cnn.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',kernel_initializer='he_uniform', input_shape=(input_shape[1], 1)))
            deep_cnn.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',kernel_initializer='he_uniform'))
            deep_cnn.add(tf.keras.layers.Dropout(0.6))
            deep_cnn.add(tf.keras.layers.MaxPooling1D(pool_size=2))
            deep_cnn.add(tf.keras.layers.Flatten())
            deep_cnn.add(tf.keras.layers.Dense(50, activation='relu'))
            deep_cnn.add(tf.keras.layers.Dense(10, activation='softmax'))
        
            deep_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            return deep_cnn
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
        new_parameters, acc, loss, cost = self._global_fit(weights=parameters, server_round=config['round']) # Atualiza o modelo global com dados do cliente

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
            header = [
                'round',
                'cid',
                'solution',
                'method',
                'g_eval_acc',
                'g_eval_loss',
                'l_eval_acc',
                'l_eval_loss',
                'g_fit_acc',
                'g_fit_loss',
                'l_fit_acc',
                'l_fit_loss',
                'dynamic_engagement',
                'old_dynamic_engagement',
                'is_selected',
                'desire',
                'size',
                'cost',
                'willing',
                'r_intention',
                'miss',
                'epoch',
                'dirichlet_alpha',
                'no_iid',
                'dataset',
                'exploitation',
                'exploration',
                'least_select_factor',
                'decay',
                'threshold',
            ],
            data = [
                config['round'],
                self.cid,
                self.solution,
                self.method,
                self.g_eval_acc,
                self.g_eval_loss,
                self.l_eval_acc,
                self.l_eval_loss,
                self.g_fit_acc,
                self.g_fit_loss,
                self.l_fit_acc,
                self.l_fit_loss,
                self.dynamic_engagement,
                self.old_dynamic_engagement,
                is_select,
                self.want,
                self.model_size,
                self.cost,
                self.willing,
                self.r_intention,
                self.miss,
                self.epoch,
                self.dirichlet_alpha,
                self.no_iid,
                self.dataset,
                self.exploitation,
                self.exploration,
                self.least_select_factor,
                self.decay,
                self.threshold,
            ]
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
        self.drivers_results = {}
        if is_select_by_server(str(self.cid), config['selected_by_server'].split(',')):

            if self.behaviors['curiosity_driver'].state == EXPLORING:
                self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)
            elif self.behaviors['curiosity_driver'].state == IDLE:
                value = self.behaviors["accuracy_driver"].analyze(self, parameters=parameters, config=config)
                self.drivers_results['accuracy_driver'] = value
                my_logger.log(
                    "/c-curiosity-cp.csv",
                    data=[config['round'], self.cid, value, self.threshold],
                    header=["round", 'cid', 'value', 'threshold'],
                )
                if value > self.threshold:
                    self.dynamic_engagement = True
                    self.want = True
                    self.behaviors['curiosity_driver'].analyze(self, parameters=parameters, config=config)

            self.rounds_intention = self.behaviors['curiosity_driver'].current_round
            self.drivers_results['curiosity_driver'] = self.rounds_intention
        if self.behaviors['curiosity_driver'].state == EXPLORED:
            self.behaviors['curiosity_driver'].finish(self)
        my_logger.log(
            "/d-curiosity.csv",
            data=[config['round'], self.cid, self.behaviors['curiosity_driver'].state, self.behaviors['curiosity_driver'].current_round],
            header=["round", 'cid', 'value', 'threshold'],
        )
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

