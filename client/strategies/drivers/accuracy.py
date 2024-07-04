from .driver import Driver
import tensorflow as tf
# from utils.select_by_server import is_select_by_server

WILLING_PERC = 1.0

class AccuracyDriver(Driver):
    def __init__(self, input_shape, model_type):
        self._create_model(input_shape = input_shape, model_type = model_type)
        self.history = []
    
    def get_name(self):
        return "accuracy_driver"

    def _create_model(self, input_shape, model_type):
        if model_type == "cnn":
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape[1:]),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation='softmax'),
            ])
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape[1:]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64,  activation='relu'),
            tf.keras.layers.Dense(32,  activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),

        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def analyze(self, client, parameters, config):       
        server_round = config['rounds']


        willing = 0
        if server_round == 1:
            self.history.append(willing)
            return willing
        
        self.model.set_weights(parameters)
        _, tmp_accuracy = self.model.evaluate(client.x_test, client.y_test,verbose=0)
        _, client_accuracy = client.model.evaluate(client.x_test, client.y_test,verbose=0)

        willing = self._better(global_acc = tmp_accuracy, client_acc = client_accuracy)
        self.history.append(willing)
        return willing
    
        return 0

    def _better(self, global_acc, client_acc):
        """
            ACC relativa entre o modelo global e o local
            se o valor relativo for > 1 significa que o global é melhor caso contrário é igual ou inferior ao local
        """
        # if global_acc > client_acc:
        #     return 1
        return global_acc / client_acc * WILLING_PERC
    
    def make_decision(self, client):
        client.want = True
        client.dynamic_engagement = True