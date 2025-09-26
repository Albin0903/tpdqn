import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class AgentDQNCible():
    """Agent qui utilise l'algorithme DQN avec réseau cible."""

    def __init__(self, dim_etat:int, dim_action:int, gamma=0.99):
        """Constructeur.
        

        """
        self.state_size = dim_etat
        self.action_size = dim_action
        

    def phase_echantillonage(self,etat : np.ndarray ,action : np.ndarray ,recompense: float,etat_suivant: np.ndarray ,terminaison: bool):
        return 0
        
    def phase_apprentissage(self):
        return 0
    
    
    def action_egreedy(self, etat : np.ndarray ,eps: float = 0.0) -> int:
        return 0