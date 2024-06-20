from abc import ABC, abstractmethod

class Driver(ABC):
    @abstractmethod
    def analyze(self, client, parameters, config):
        pass
    def get_name(self) -> str:
        pass
    def make_decision(self, client):
        pass