from abc import ABC, abstractmethod

class ClientState(ABC):
    @abstractmethod
    def participate(self, client):
        pass

    @abstractmethod
    def non_participate(self, client):
        pass

class ParticipationState(ClientState):
    def participate(self, client):
        print("Client already participating")
        return

    def non_participate(self, client):
        client.participating_state = False
        client.set_state(client.non_participation_state)
        return

class NonParticipationState(ClientState):
    def participate(self, client):
        client.participating_state = True
        client.set_state(client.participation_state)
        return

    def non_participate(self, client):
        print("Client already non-participating")
        return
