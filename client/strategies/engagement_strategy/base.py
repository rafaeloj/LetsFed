from abc import ABC, abstractmethod
import numpy as np

class EngagementStrategy(ABC):
    @abstractmethod
    def calculate_engagement(self):
        pass

def calculate(engagement: EngagementStrategy) -> float:
    return engagement.calculate_engagement()

calculate_criteria = np.vectorize(calculate)
