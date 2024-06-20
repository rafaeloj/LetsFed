from abc import ABC, abstractmethod

class Driver(ABC):
    @abstractmethod
    def run(self, server, parameters, config):
        pass
    def get_name(self) -> str:
        pass