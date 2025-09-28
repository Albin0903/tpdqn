import numpy as np
import random

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


class AgentDQN:
    """Agent DQN: Q-fonction approchée (QNN) + replay buffer + epsilon-greedy."""

    def __init__(self, dim_etat: int, dim_action: int, gamma=0.99, lr=5e-4, buffer_size=100000, batch_size=64, update_every=4):
        """
        dim_etat: dimension de l'état
        dim_action: nombre d'actions discrètes
        gamma: facteur d'actualisation
        lr: taux d'apprentissage
        buffer_size: capacité du replay buffer
        batch_size: taille des minibatchs
        update_every: fréquence (en pas) des mises à jour
        """
        self.state_size = dim_etat
        self.action_size = dim_action
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every

        self.network = QNN(dim_etat, dim_action)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.t_step = 0  # cadence d'apprentissage (tous les update_every pas)

    def phase_echantillonage(self, etat: np.ndarray, action: int, recompense: float, etat_suivant: np.ndarray, terminaison: bool):
        """Mémorise la transition et déclenche l'apprentissage selon la cadence."""
        self.memory.add(etat, action, recompense, etat_suivant, terminaison)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.phase_apprentissage()

    def phase_apprentissage(self):
        """Un pas d'optimisation sur un minibatch échantillonné (i.i.d.)."""
        states, actions, rewards, next_states, dones = self.memory.sample() # échantillonnage

        # Cible de Bellman
        with torch.no_grad():
            next_q_values = self.network(next_states).max(1, keepdim=True)[0] # max_a' Q(s',a')
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones) # y = r + γ max_a' Q(s',a') si non terminal, sinon y = r

        # Q(s,a) pour les actions réellement prises
        q_values = self.network(states).gather(1, actions) 

        # Descente de gradient sur MSE(Q(s,a), y)
        loss = F.mse_loss(q_values, target_q_values) 
        self.optimizer.zero_grad() 
        loss.backward() 
        self.optimizer.step() 

    def action_egreedy(self, etat: np.ndarray, eps: float = 0.0) -> int: 
        """Politique epsilon-greedy: exploite si rand>eps, sinon explore."""
        if random.random() > eps:
            state = torch.tensor(etat, dtype=torch.float).unsqueeze(0)
            self.network.eval()
            with torch.no_grad():
                q_vals = self.network(state)
            self.network.train()
            return int(np.argmax(q_vals.cpu().numpy()))
        else:
            return random.randrange(self.action_size)