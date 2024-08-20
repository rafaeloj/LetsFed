from abc import ABC, abstractmethod

class Driver(ABC):
    @abstractmethod
    def run(self, client, parameters, config):
        pass
    @abstractmethod
    def get_name(self) -> str:
        pass
    