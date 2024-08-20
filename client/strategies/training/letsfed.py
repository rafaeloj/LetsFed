from .base import TrainingStrategy
from client.strategies.drivers import AccuracyDriver, CuriosityDriver
from utils import ModelManager, is_select_by_server
from typing import Dict, Tuple
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)
import time
import numpy as np

class LestFedClient(TrainingStrategy):
    def __init__(self, client: 'FederatedClient'):
        client.conf['miss'] = 0
        client.conf['desired_state'] = client.participating_state
        client.conf['rounds_intention'] = 1
        client.conf['diff'] = 0
        self.load_debug_model(client)
        client.add_drivers(
            self.__get_drivers(
                client.x_train.shape,
                client.conf['model_type'],
                client.conf['rounds_intention'],
                client.conf['threshold'],
            )
        )
    
    def __get_drivers(self, shape: Tuple[int], model_type: str, rounds_intention: int, threshold: float):
        drivers = [
            AccuracyDriver(shape, model_type, threshold),
            CuriosityDriver(rounds_intention)
        ]
        return drivers
    
    def load_debug_model(self, client: 'FederatedClient'):
        mm = ModelManager(
            model_type = client.model_type,
            input_shape = client.x_train.shape
        )
        client.debug_model = mm.get_model(client.model_type)

    def fit(self, client: 'FederatedClient', parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        fit_response = {
            'cid': client.conf['cid'],
            "participating_state": client.participating_state,
        }

        history = client.debug_model.fit(client.x_train, client.y_train, epochs=client.conf['epochs'], verbose=0)
        client.conf['l_fit_acc']     = np.mean(history.history['accuracy'])
        client.conf['l_fit_loss']    = np.mean(history.history['loss'])
        model_size = sum([layer.nbytes for layer in parameters])
        client.conf['model_size'] = model_size
        client.conf['selected'] = is_select_by_server(str(client.conf['cid']), config['selected_by_server'].split(','))
        if client.conf['selected']:
            return self.__participating_fit(client = client, parameters=parameters, config=config), client.x_train.shape[0], fit_response
        return self.__non_participating_fit(client =client, parameters=parameters, config=config), client.x_train.shape[0], fit_response
    
    def __participating_fit(self, client: 'FederatedClient', parameters: NDArrays, config: Config) -> NDArrays:
        if not client.participating_state:
            return self.__non_participating_fit(client, parameters=parameters, config=config) 
        
        client.model.set_weights(parameters)
        start_time = time.time()
        history    = client.model.fit(client.x_train, client.y_train, epochs=client.conf['epochs'], verbose=0)
        acc        = np.mean(history.history['accuracy'])
        loss       = np.mean(history.history['loss'])
        new_parameters = client.model.get_weights()
        end_time = time.time()
        cost = end_time - start_time

        client.conf['g_fit_acc'] = np.mean(acc)
        client.conf['g_fit_loss'] = np.mean(loss)
        client.conf['cost'] = cost

        return new_parameters
    
    def __non_participating_fit(self, client: 'FederatedClient', parameters: NDArrays, config: Config):
        start_time = time.time()
        history = client.model.fit(client.x_train, client.y_train, epochs=client.conf['epochs'], verbose=0)
        acc     = np.mean(history.history['accuracy'])
        loss    = np.mean(history.history['loss'])
        end_time = time.time()
        cost = end_time - start_time

        client.conf['g_fit_acc'] = np.mean(acc)
        client.conf['g_fit_loss'] = np.mean(loss)
        client.conf['cost'] = cost
        
        return parameters

    def evaluate(self, client: 'FederatedClient', parameters: NDArrays, config: Config)-> Tuple[NDArrays, int, Dict[str, Scalar]]:
        loss = None
        shape = None
        eval_resp = None

        l_loss, l_acc = client.debug_model.evaluate(client.x_test, client.y_test)
        client.conf['l_eval_acc'] = np.mean(l_acc)
        client.conf['l_eval_loss'] = np.mean(l_loss)

        client.conf['selected'] = is_select_by_server(str(client.conf['cid']), config['selected_by_server'].split(','))

        if client.conf['selected']:
            client.conf['miss']+=1
            loss, shape, eval_resp = self.__participating_evaluate(client = client, parameters=parameters, config=config)
        else:
            loss, shape, eval_resp = self.__non_participating_evaluate(client = client, parameters=parameters, config=config)

        eval_resp['fit_acc'] = client.conf['g_fit_acc']

        return loss, shape, eval_resp

    def __participating_evaluate(self, client: 'FederatedClient', parameters: NDArrays, config: Config) -> Tuple[float, int, Config]:
        if not client.participating_state:
            return self.__non_participating_evaluate(client, parameters=parameters, config=config)

        client.model.set_weights(parameters)
        loss, acc = client.model.evaluate(client.x_test, client.y_test)

        old_participating_state = client.participating_state
        self.__apply_drivers(client, parameters, config)

        evaluation_response = {
            "cid"                : client.conf['cid'],
            "participating_state": old_participating_state,
            'desired_state'      : client.participating_state,
            'acc'               : acc,
            'r_intention'        : client.conf['rounds_intention']
        }
        client.conf['g_eval_loss'] = np.mean(loss)
        client.conf['g_eval_acc'] = np.mean(acc)
        client.conf['r_intention'] = client.conf['rounds_intention']
        client.conf['old_participating_state'] = old_participating_state

        return np.mean(loss), client.x_test.shape[0], evaluation_response

    def __non_participating_evaluate(self, client: 'FederatedClient', parameters: NDArrays , config: Config) -> Tuple[float, int, Config]:
        loss, acc = client.model.evaluate(client.x_test, client.y_test)
        old_participating_state = client.participating_state
        self.__apply_drivers(client, parameters, config)
        
        client.conf['g_eval_acc'] = np.mean(acc)
        client.conf['g_eval_loss'] = np.mean(loss)
        client.conf['r_intention'] = client.conf['rounds_intention']
        client.conf['old_participating_state'] = old_participating_state

        evaluation_response = {
            "cid"                : client.conf['cid'],
            "participating_state": old_participating_state,
            'desired_state'      : client.participating_state,
            'acc'                : acc,
            'r_intention'        : client.conf['rounds_intention'],
        }
        return loss, client.x_test.shape[0], evaluation_response

    def __apply_drivers(self, client: 'FederatedClient', parameters, config):
        drivers_answers = client.apply_drivers(parameters, config)
        self.__manager_state(client, drivers_answers['accuracy_driver'])

    def __manager_state(self, client: 'FederatedClient', state: bool):
        if not state and client.participating_state:
            print(client.participating_state)
            client.state.non_participate(client)
            print(client.participating_state)
            return
        
        if state and not client.participating_state:
            print(client.participating_state)
            client.state.participate(client)
            print(client.participating_state)
            return
        return
        # raise ValueError(f"Client state error; State: {state} - Client state: {client.participating_state}")
