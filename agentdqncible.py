import numpy as np
import random

from QNN import QNN
from replaybuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


class AgentDQNCible:
    """Agent DQN avec réseau cible (QNN + replay buffer + epsilon-greedy)."""

    def __init__(
        self,
        dim_etat: int,
        dim_action: int,
        gamma=0.99,
        lr=5e-4,
        buffer_size=100000,
        batch_size=64,
        update_every=4,
        target_update_every=500,  # fréquence de copie du réseau vers le réseau cible (en pas d'apprentissage)
    ):
        # Hyperparamètres et buffers
        self.state_size = dim_etat
        self.action_size = dim_action
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update_every = target_update_every

        # Réseau en ligne (policy) et réseau cible (target)
        self.network = QNN(dim_etat, dim_action)
        self.target_network = QNN(dim_etat, dim_action)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Compteurs
        self.t_step = 0 # cadence d'appel à l'apprentissage
        self.learn_step = 0 # pas d'apprentissage cumulé (pour MAJ réseau cible)

    def phase_echantillonage(self, etat: np.ndarray, action: int, recompense: float, etat_suivant: np.ndarray, terminaison: bool):
        """Stocke la transition; déclenche l'apprentissage périodiquement."""
        self.memory.add(etat, action, recompense, etat_suivant, terminaison)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.phase_apprentissage()

    def phase_apprentissage(self):
        """Un pas d'optimisation sur un minibatch; cibles calculées avec le réseau cible."""
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Cible de Bellman avec le réseau cible (stabilise l'apprentissage)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Q(s,a) du réseau en ligne pour les actions réellement prises
        q_values = self.network(states).gather(1, actions)

        # Optimisation MSE(Q(s,a), cible)
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour périodique "hard" du réseau cible
        self.learn_step += 1
        if self.learn_step % self.target_update_every == 0:
            for p_tgt, p_src in zip(self.target_network.parameters(), self.network.parameters()):
                p_tgt.data.copy_(p_src.data)

    def action_egreedy(self, etat: np.ndarray, eps: float = 0.0) -> int:
        """Politique epsilon-greedy: exploitation sinon exploration."""
        if random.random() > eps:
            state = torch.tensor(etat, dtype=torch.float).unsqueeze(0)
            self.network.eval()
            with torch.no_grad():
                q_vals = self.network(state)
            self.network.train()
            return int(np.argmax(q_vals.cpu().numpy()))
        else:
            return random.randrange(self.action_size)