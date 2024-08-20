from .driver import Driver
from random import randint
IDLE = 0
EXPLORING = 1
class CuriosityDriver(Driver):
    def __init__(self, r_intention=0):
        self.state         = IDLE # Type: idle, exploring, explored
        self.current_round = r_intention
    
    def get_name(self):
        return "curiosity_driver"

    def run(self, client, parameters, config, selected=True):
        if not selected:
            return True if self.state == EXPLORING else False

        if self.on_exploration():
            state = self.explore()
            return state

        if client.participating_state:
            state = self.start_exploration(client = client)
            return state
    
        return False


    def explore(self):
        self.current_round -= 1
        if not self.on_exploration():
            self.set_idle()
            return False
        return True

    def on_exploration(self):
        return self.current_round > 0 and self.state == EXPLORING

    def start_exploration(self, client):
        if self.current_round == 0:
            self.current_round = randint(1, int(client.conf['rounds'])+1)
        self.set_exploring()
        return True
    
    def set_exploring(self):
        self.state = EXPLORING
    
    def set_idle(self):
        self.current_round = 0
        self.sate = IDLE
