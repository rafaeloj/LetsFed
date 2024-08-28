from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from typing import List
if TYPE_CHECKING:
    from server.strategies import FLServer

class ClientSelectionMethod(ABC):
    @abstractmethod
    def select(
        self,
        server: 'FLServer',
        server_round: int,
        list_of_clients: List[str],
    ) -> List[str]:
        pass