import numpy as np
import random
import torch

from QNN import QNN

class AgentSimple():
    """Agent qui utilise la prédiction de son réseau de neurones pour choisir ses actions selon une stratégie d’exploration (pas d'apprentissage)."""

    def __init__(self, dim_etat: int, dim_action: int):
        """
            dim_etat: dimension de l'état
            dim_action: dimension de l'action
        """
        self.dim_etat = dim_etat
        self.dim_action = dim_action
        self.qnn = QNN(dim_etat, dim_action)

    def action_egreedy(self, etat : np.ndarray , eps: float = 0.0) -> int:
        """
            eps: probabilité d'exploration
        """
        if random.random() < eps: # exploration
            return random.randrange(self.dim_action) # action aléatoire dans [0, dim_action-1]
        else:
            self.qnn.eval() # mise en mode évaluation
            with torch.no_grad(): # exploitation
                q_values = self.qnn(etat) # prédiction des valeurs Q pour l'état donné
        return torch.argmax(q_values).item() # action avec la valeur Q maximale, convertie en entier
    
        
