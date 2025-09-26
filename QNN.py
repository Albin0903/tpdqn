import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNN(nn.Module):
    """Reseau de neurones pour approximer la Q fonction."""

    def __init__(self,dim_entree:int, dim_sortie:int):
        """Initialisation des parametres ...
        """
        super(QNN, self).__init__()
        
        "*** TODO ***"
        self.fc1 = nn.Linear(dim_entree, 64) # Couche d'entrée
        self.fc2 = nn.Linear(64, 64) # Couche cachée
        self.fc3 = nn.Linear(64, dim_sortie) # Couche de sortie
        
        
    def forward(self, etat: np.ndarray) -> torch.Tensor :
        """Forward pass"""

        if isinstance(etat, np.ndarray):
            etat = torch.tensor(etat, dtype=torch.float)
            
        "*** TODO ***"
        x = F.relu(self.fc1(etat)) # Fonction d'activation ReLU
        x = F.relu(self.fc2(x)) # Fonction d'activation ReLU
        etat = self.fc3(x) # Aucune fonction (identité) sur la couche de sortie
        
        return etat


