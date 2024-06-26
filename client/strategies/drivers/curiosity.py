from .driver import Driver
from random import randint

IDLE = 0
EXPLORING = 1
EXPLORED = 2
CURIOSITY = 0.1
WILLING_PERC = 0.80
class CuriosityDriver(Driver):
    def __init__(self):
        self.history       = []
        self.state         = IDLE # Type: idle, exploring, explored
        self.current_round = 0
    
    def get_name(self):
        return "curiosity_driver"

    def analyze(self, client, parameters, config):
        """
        """
        if self.on_exploration(client = client):
            state = self.explore(client = client)
            return state

        if client.dynamic_engagement:
            state = self.start_exploration(client = client)
            return state
    
        # self.current_round = randint(1, int(client.rounds*0.1))
        return 0 #self.calc_value(client = client)


    def explore(self, client):
        """
            Regra:
                Caso ainda esteja no processo de exploração "executa" a exploração e retorna True
                Caso não esteja retorna False
        """
        self.current_round -= 1
        if not self.on_exploration(client):
            self.set_explored()
        return self.calc_value(client = client)

    def on_exploration(self, client):
        return self.current_round > 0 and self.state == EXPLORING

    def start_exploration(self, client):
        if self.current_round == 0:
            self.current_round = randint(1, int(client.rounds*CURIOSITY)+1)
        self.set_exploring()
        return self.calc_value(client = client)

    def set_explored(self):
        self.state = EXPLORED
    
    def set_exploring(self):
        self.state = EXPLORING
    
    def set_idle(self):
        self.current_round = 0
        self.sate = IDLE

    def finish(self, client):
        """
            Finaliza o processo de explocação
        """
        self.set_idle()
        client.dynamic_engagement = False
        client.want = False

    def calc_value(self, client):
        """ A vontate é com base na quantidade de rounds que o client deseja participar. Quanto mais rounds maior é a vontade """
        # return 1
        return self.current_round / (CURIOSITY*100) * WILLING_PERC