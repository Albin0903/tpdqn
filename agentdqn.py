import numpy as np
import random
from collections import namedtuple, deque

from QNN import QNN  # Assurez-vous que ce fichier est dans le même dossier
from replaybuffer import ReplayBuffer # Assurez-vous que ce fichier est dans le même dossier

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class AgentDQN():
    """Agent qui utilise l'algorithme de deep QLearning avec replaybuffer."""

    def __init__(self, dim_etat:int, dim_action:int, gamma=0.99, lr=5e-4, buffer_size=100000, batch_size=64, update_every=4):
        """
        Constructeur de l'agent DQN.
        
        Args:
            dim_etat (int): Dimension de l'espace d'état.
            dim_action (int): Dimension de l'espace d'action.
            gamma (float): Facteur d'actualisation (discount factor).
            lr (float): Taux d'apprentissage (learning rate) pour l'optimiseur.
            buffer_size (int): Taille maximale du replay buffer.
            batch_size (int): Taille du minibatch à échantillonner pour l'apprentissage.
            update_every (int): Fréquence (en nombre de pas) à laquelle le réseau est mis à jour.
        """
        self.state_size = dim_etat
        self.action_size = dim_action
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        
        # --- Réseau de neurones (Q-Network) ---
        # Le réseau principal qui apprend les valeurs Q.
        self.network = QNN(dim_etat, dim_action)
        
        # --- Optimiseur ---
        # Adam est un optimiseur robuste qui ajuste le taux d'apprentissage.
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr) 
        
        # --- Replay Buffer ---
        # La mémoire qui stocke les transitions (état, action, récompense, ...).
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
        # --- Compteur ---
        # Compteur pour déclencher la phase d'apprentissage tous les `update_every` pas.
        self.t_step = 0

    def phase_echantillonage(self, etat: np.ndarray, action: int, recompense: float, etat_suivant: np.ndarray, terminaison: bool):
        """
        Étape 1: Stocke une transition dans le replay buffer.
        Étape 2: Déclenche la phase d'apprentissage si les conditions sont remplies.
        """
        # Ajoute la nouvelle transition à la mémoire.
        self.memory.add(etat, action, recompense, etat_suivant, terminaison)
        
        # Incrémente le compteur de pas.
        self.t_step = (self.t_step + 1) % self.update_every
        
        # Si le compteur atteint 0, il est temps d'apprendre.
        if self.t_step == 0:
            # On vérifie qu'il y a assez d'échantillons dans la mémoire pour former un batch.
            if len(self.memory) > self.batch_size:
                self.phase_apprentissage()
        
    def phase_apprentissage(self):
        """
        Met à jour les poids du réseau de neurones en utilisant un batch
        d'expériences tirées du replay buffer.
        """
        # Échantillonne un batch de transitions depuis la mémoire.
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # --- Calcul des Q-Valeurs Cibles (Y_j) ---
        with torch.no_grad(): # On ne calcule pas les gradients pour les cibles.
            # 1. Prédire les Q-valeurs pour les états suivants.
            #    Le .max(1)[0] sélectionne la meilleure Q-valeur pour chaque état suivant.
            #    Le .unsqueeze(1) redimensionne le tenseur pour les calculs suivants.
            next_q_values = self.network(next_states).max(1)[0].unsqueeze(1)
            
            # 2. Calculer la Q-valeur cible selon la formule de Bellman.
            #    Si l'état est terminal (done=1), la cible est juste la récompense.
            #    Sinon, c'est recompense + gamma * (meilleure Q-valeur future).
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # --- Calcul des Q-Valeurs Attendues (Q(S_j, A_j)) ---
        # 1. Obtenir les Q-valeurs prédites pour les actions qui ont été réellement prises.
        #    Le .gather(1, actions) sélectionne la Q-valeur correspondant à l'action prise.
        q_values = self.network(states).gather(1, actions)
        
        # --- Calcul de la Perte ---
        # On utilise la Mean Squared Error (MSE) entre les valeurs cibles et les valeurs prédites.
        loss = F.mse_loss(q_values, target_q_values)
        
        # --- Optimisation (Mise à jour des poids) ---
        # 1. Remettre les gradients à zéro.
        self.optimizer.zero_grad()
        # 2. Rétropropager l'erreur (calculer les gradients).
        loss.backward()
        # 3. Mettre à jour les poids du réseau.
        self.optimizer.step()
    
    
    def action_egreedy(self, etat: np.ndarray, eps: float = 0.0) -> int:
        """
        Sélectionne une action en utilisant la stratégie epsilon-greedy.
        
        Args:
            etat (np.ndarray): L'état actuel de l'environnement.
            eps (float): La probabilité de choisir une action aléatoire (exploration).
        
        Returns:
            int: L'action choisie.
        """
        # Décide s'il faut explorer ou exploiter.
        if random.random() > eps:
            # --- Exploitation ---
            # Convertit l'état (numpy array) en tenseur PyTorch.
            # .unsqueeze(0) ajoute une dimension pour simuler un "batch" de taille 1.
            state = torch.from_numpy(etat).float().unsqueeze(0)
            
            # Passe le réseau en mode évaluation.
            self.network.eval()
            with torch.no_grad():
                # Prédit les Q-valeurs pour toutes les actions possibles depuis l'état actuel.
                action_values = self.network(state)
            # Repasse le réseau en mode entraînement.
            self.network.train()
            
            # Choisit l'action avec la plus grande Q-valeur.
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # --- Exploration ---
            # Choisit une action au hasard dans l'espace d'actions.
            return random.randrange(self.action_size)